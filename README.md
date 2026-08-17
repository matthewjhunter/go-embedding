# go-embedding

A small, reusable Go module for vector embeddings: a single `Embedder`
interface, two HTTP backends (Ollama and any OpenAI-compatible
`/v1/embeddings` server), per-model byte limits, and a fingerprint
contract that catches model swaps before they silently corrupt your
search results.

> Maintained as a personal utility shared across my projects (memstore,
> herald, …). Issues and PRs are welcome but I may not respond quickly.

```go
import "github.com/matthewjhunter/go-embedding"
```

## Quick start

```go
e, err := embedding.New(embedding.Config{
    Backend: embedding.BackendOllama,
    BaseURL: "http://localhost:11434",
    Model:   "nomic-embed-text",
})
if err != nil {
    log.Fatal(err)
}

vec, err := embedding.Single(ctx, e, "hello, world")
```

For a one-line ecosystem default:

```go
e, _ := embedding.New(embedding.DefaultConfig())
```

`DefaultConfig` currently aliases `OllamaLocalNomic`. There is also
`LemonadeNomic` (Lemonade Server on its default port, OpenAI protocol).
External callers should prefer constructing `Config` explicitly so a
default change in this module doesn't surprise them on a `go get -u`.

## Env-driven configuration

To share one embedding configuration across multiple apps, set the
canonical env vars once and have every app read them:

```sh
export EMBEDDING_BACKEND=ollama
export EMBEDDING_BASE_URL=http://gpu-host:11434
export EMBEDDING_MODEL=nomic-embed-text
```

```go
cfg, err := embedding.ConfigFromEnv()
if err != nil { log.Fatal(err) }
e, err := embedding.New(cfg)
```

Recognised vars: `EMBEDDING_BACKEND`, `EMBEDDING_BASE_URL`,
`EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_STRICT`,
`EMBEDDING_TIMEOUT`, `EMBEDDING_PER_INPUT_TIMEOUT`,
`EMBEDDING_MAX_BYTES`, `EMBEDDING_MAX_TOKENS`. Unset (or empty) vars
fall back to `DefaultConfig`. Unknown backend names, unparseable bools,
durations, and budgets return an error.

For per-app namespaces use a custom prefix:

```go
cfg, _ := embedding.ConfigFromEnvPrefix("MEMSTORE_EMBED")
// reads MEMSTORE_EMBED_BACKEND, MEMSTORE_EMBED_BASE_URL, …
```

`ConfigFromEnvPrefix` cascades per field: prefixed key → canonical
`EMBEDDING_*` key → `DefaultConfig`. So you can set
`EMBEDDING_BASE_URL` once for every app and override only `MEMSTORE_EMBED_MODEL`
for the one app that needs a different model — all the other fields
still come from the shared canonical env.

## Backends

| Backend | Endpoint | Authentication |
|---|---|---|
| `BackendOllama` | `POST {BaseURL}/api/embed` | none |
| `BackendOpenAI` | `POST {BaseURL}/v1/embeddings` | optional `Bearer {APIKey}` |

`BackendOpenAI` works against OpenAI itself, LiteLLM, vLLM, Ollama
(>=0.1.24), Lemonade, and anything else speaking the same protocol.

## Fingerprint check

Two model versions can share a name while producing incompatible
vectors (e.g. `nomic-embed-text` v1 and v2 — same name, different
internal weights, mixed rankings come out as silent garbage). A
fingerprint pairs the model name with the vector dimension, which is
filled in after the first `Embed` call.

Persist the fingerprint when you write your first vector, then check
it on every subsequent open:

```go
current := e.Fingerprint()
if err := embedding.CheckFingerprint(stored, current); err != nil {
    var mismatch *embedding.MismatchError
    if errors.As(err, &mismatch) {
        // re-embed your corpus, or refuse to serve stale vectors
    }
}
```

## Request deadlines

Every HTTP request carries a deadline, applied as a context timeout so
a failure arrives as `context.DeadlineExceeded` and composes with the
caller's own cancellation. Without one, a backend that accepts a
connection and then stops responding hangs the caller forever, which
is worse than an error: an error retries, a hang stalls the queue.

The budget scales with the request, because request cost does:
`Timeout + PerInputTimeout * len(texts)`. A single flat number would
either trip on a large batch or be too loose to catch a hang on a
small one.

```go
e, _ := embedding.New(embedding.Config{
    Backend:         embedding.BackendOllama,
    BaseURL:         "http://gpu-host:11434",
    Model:           "nomic-embed-text",
    Timeout:         30 * time.Second, // base, zero uses DefaultTimeout
    PerInputTimeout: 2 * time.Second,  // per input, zero uses DefaultPerInputTimeout
})
```

Defaults are `DefaultTimeout` (30s) and `DefaultPerInputTimeout` (2s),
so a 25-input batch gets 80s. They are deliberately generous: the job
is catching a wedged backend, not policing a slow one. Tune them down
only with measurements from your own hardware.

Zero means "use the default" rather than "disable", so a caller who
never thinks about timeouts still gets one. To opt out, set
`Timeout: embedding.NoTimeout` for unbounded requests, or
`PerInputTimeout: embedding.NoTimeout` to keep a flat budget that
doesn't scale with batch size.

From the environment, both take a duration string (`45s`, `500ms`) or
the word `none` to disable. A bare number is an error rather than a
guess -- `30` is as likely to mean seconds as milliseconds, and
choosing wrong silently turns the deadline into either a hang or a
stream of spurious timeouts.

`RerankConfig` carries the same two fields, scaling on the document
count.

## Limits

`Embed` consults a per-model byte limit registered for nomic-embed-text
and a few siblings. Oversize input is truncated to the limit (with a
`log.Printf` so the truncation is visible in logs). Set `Strict: true`
on `Config` to make oversize input an error instead.

`LookupLimits` (and the related task / document prompter lookups) fall
back to the bare model name when a tagged variant is not registered.
That means `nomic-embed-text:latest` and `nomic-embed-text:q4_0` get the
base model's limits automatically — limits are an architectural
property and don't change with a tag. **Storage keys are NOT
canonicalised this way** (see `Config.Model` doc): vectors from
different tags can be incompatible.

Register custom models at startup:

```go
embedding.RegisterLimits("my-custom-embedder", embedding.Limits{MaxBytes: 4096})
```

Or override the budget per embedder, without touching the registry:

```go
e, _ := embedding.New(embedding.Config{
    // ...
    MaxTokens: 8192,  // or EMBEDDING_MAX_TOKENS
})
```

A model the registry has never heard of becomes enforceable from
`MaxTokens` alone -- a byte budget is derived from it conservatively --
so a new model does not need a library edit. Models with neither a
registry entry nor an override get no enforcement.

## Clipped input

A byte budget is a stand-in for a token budget, and the conversion is
approximate. When it guesses high, the input overruns the model's
context, and the two backend families fail differently: llama.cpp and
Lemonade reject the request outright, while **Ollama silently truncates**
-- returning a normal-looking vector computed from a document whose tail
was thrown away. Nothing about the response says so.

Both protocols do report the token count they processed
(`prompt_eval_count`, `usage.prompt_tokens`), and a count that has
reached the budget is the fingerprint of an input that was cut. Set
`OnUsage` to see it:

```go
e, _ := embedding.New(embedding.Config{
    // ...
    OnUsage: embedding.UsageFunc(func(u embedding.Usage) {
        if u.Clipped {
            clipped.Add(1)  // measure your real clip rate
        }
    }),
})
```

`Strict: true` turns a clipped result into an error instead, on the same
reasoning that makes strict mode refuse to truncate before sending.

Two things `Clipped` deliberately does not claim. It stays false for a
multi-input request, because both protocols report one total for the
whole batch and an over-budget total across several inputs is the normal
shape rather than evidence about any one of them -- single-input requests
are what this catches, including every one-by-one fallback
`BatchEmbedResults` performs. And it cannot tell a clipped input from one
that legitimately lands within a few tokens of the budget; the budget
sits below the true context window precisely so nothing lands there.

## Structured input: fields and task prompts

Most production embedding work isn't "embed this raw string." It's
"embed this article along with its author, feed name, categories, and
content," or "embed this fact along with its subject and category."
The library provides two layers for assembling that input:

```go
// Caller-controlled metadata (stable, ordered key-value lines):
type Field struct{ Key, Value string }
text := embedding.FormatRecord(
    []embedding.Field{
        {"feed",       "Schneier on Security"},
        {"author",     "Bruce Schneier"},
        {"categories", "cryptography, surveillance"},
        {"title",      "How AI Will Change Cyber Defense"},
    },
    "Full article body…",
)
```

Produces:

```
feed: Schneier on Security
author: Bruce Schneier
categories: cryptography, surveillance
title: How AI Will Change Cyber Defense

Full article body…
```

Empty values (and empty keys) are skipped. `Field` is a slice, not a
map, on purpose — Go map iteration is non-deterministic, and the
embedder learns recurring positional patterns, so two calls with the
same data must produce identical text.

To wrap that in a model-specific task prompt:

```go
text := embedding.FormatRecordForTask(
    "nomic-embed-text:latest",
    embedding.TaskClustering,
    fields,
    body,
)
// → "clustering:\nfeed: …\nauthor: …\n…\n\nbody"

text = embedding.FormatRecordForTask(
    "embeddinggemma",
    embedding.TaskClustering,
    fields,
    body,
)
// → "task: clustering | query:\nfeed: …\n…\n\nbody"
```

The task prefix is followed by a newline so structured field labels
start at column 0 below it. This keeps recurring `key:` patterns
positionally stable across the corpus regardless of which model's
prefix is in use.

Built-in conventions cover `nomic-embed-text` and `embeddinggemma`
across `TaskRetrievalDocument`, `TaskRetrievalQuery`, `TaskClustering`,
`TaskClassification`, `TaskSimilarity`, `TaskQuestionAnswering`,
`TaskFactChecking`, and `TaskCodeRetrieval` (each model implements the
subset it documents). Models without a registered convention pass the
text through unchanged. Add your own:

```go
embedding.RegisterTaskPrompter("my-model", func(task embedding.Task, text string) string {
    if task == embedding.TaskClustering {
        return "[CLUSTER] " + text
    }
    return text
})
```

### Retrieval-document with a real title

EmbeddingGemma's retrieval-document prompt has a structural title slot
(`title: <T> | text: <body>`). nomic-embed-text doesn't —
`search_document:` wraps the entire input as one blob. To use the title
slot when the model supports it, and gracefully fall back to a regular
metadata field when it doesn't:

```go
text := embedding.FormatRetrievalDocument(
    "embeddinggemma",
    "How AI Will Change Cyber Defense",  // title slot
    fields,
    body,
)
// → "title: How AI Will Change Cyber Defense | text:\nfeed: …\n…\n\nbody"

text = embedding.FormatRetrievalDocument(
    "nomic-embed-text",
    "How AI Will Change Cyber Defense",  // promoted to a leading field
    fields,
    body,
)
// → "search_document:\ntitle: How AI Will Change Cyber Defense\nfeed: …\n\nbody"
```

Register a custom title-aware prompter via `RegisterDocumentPrompter`.

### Stripping non-semantic content

`StripNonsemantic` removes URLs, HTML tags, markdown link/image syntax,
and runs of whitespace from a body string. The intent is to fit more
real prose into a model's context window — URLs and markup tokenize
densely but contribute no semantic signal to a clustering or
retrieval embedder.

```go
clean := embedding.StripNonsemantic(article.Content)
text  := embedding.FormatRecordForTask(model, embedding.TaskClustering, fields, clean)
```

What's stripped (full table in the godoc): bare URLs, HTML tags,
`![alt](url)` → `alt`, `[text](url)` → `text`, whitespace runs
collapsed. Markdown emphasis markers (`*`, `_`) are preserved —
they're 1-2 bytes each and removing them safely across `snake_case`
identifiers is more error-prone than the savings justify.

This is a lossy transform. Don't apply it to inputs where URLs or
markup carry meaning (e.g. a search-indexed README). For RSS article
bodies feeding a clustering or retrieval embedder, the loss is
intentional.

## Batch helper

`BatchEmbed` issues batch embed calls and falls back to one-by-one when
a backend returns either an error or fewer vectors than inputs (some
servers return 200 with a partial response). The result slice is
always the same length as the input; failed entries are nil so the
caller keeps index alignment.

```go
vectors, err := embedding.BatchEmbed(ctx, e, texts, 25, func(done, total int) {
    log.Printf("embedded %d/%d", done, total)
})
```

`BatchEmbed` only returns a non-nil error if every input failed.

## Compatibility

`NewOllamaEmbedder` and `NewOpenAIEmbedder` are still exported but
marked `Deprecated`. They will be removed in v1.0; new code should use
`New(Config)`.
