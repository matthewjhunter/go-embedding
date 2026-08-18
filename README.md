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
durations, and budgets return an error. A budget must be positive: `0` is
rejected rather than read as "no budget", since once it reaches `Config` it
is indistinguishable from unset. Unset the variable to use the model's
registered limits.

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

## Calibration: what the ratio really is

Every byte budget in this library is a token budget in disguise, converted
by a constant. Measured against real corpora that constant is wrong often
enough to matter -- text that was assumed to run ~3 bytes per token
measures 1.7-2.0, and denser content (CJK, base64, source, URL-heavy
text) runs lower still.

Rather than guess, watch. Every response reports the tokens it processed,
so pairing that with the bytes sent gives a real observation per request,
free and without a tokenizer:

```go
c, ok := embedding.CalibrationFor("nomic-embed-text")
if ok {
    log.Printf("bytes/token: min %.2f p10 %.2f median %.2f (n=%d, %d clipped)",
        c.Min, c.P10, c.Median, c.Samples, c.Clipped)
}
```

This is **measurement only**. Nothing derives a budget from it; the static
ratio still converts tokens to bytes. Read it to find out what the ratio
is for your corpus and model before deciding what to do about it.

Two things to know when reading a snapshot:

**Check `Clipped` first.** Requests the backend clipped are counted but
not measured -- their token count covers only the bytes that got through,
so the ratio would come out inflated. But clipped inputs are the *densest*
documents in a corpus, so a high count means the retained sample is
biased toward text that tokenizes easily, and the estimate is optimistic.

**Use the low end, not the middle.** Underestimating bytes per token
clips input; overestimating wastes a little of the window. `Min` and
`P10` are the conservative figures a budget should be derived from.

Observations are in-process, per model, and not persisted, over a sliding
window of the most recent few thousand requests. `ResetCalibration` clears
them, which is useful for scoping a window to one specific run such as a
full re-embed.

## Chunking

A vector has fixed capacity. Pouring a whole long document into one
embedding averages away the specificity retrieval depends on, and
anything past the context window is silently discarded on top of that
(see "Clipped input" above). `Split` cuts a document into segments that
each fit a byte budget:

```go
chunks := embedding.Split("nomic-embed-text", article, embedding.SplitOptions{
    MaxBytes: 1024,  // ~512 tokens, the usual retrieval working range
    Overlap:  128,   // carry context across the boundary
    MinBytes: 200,   // no trailing slivers
})
for _, c := range chunks {
    store(c.Ordinal, c.Start, c.End, embed(c.Text))
}
```

Boundaries are chosen in descending preference: paragraph, line,
sentence, word, and only a hard cut when a single unbroken run leaves no
choice. Cuts land on rune boundaries.

### Markdown structure

Set `Structure: StructureMarkdown` and `Split` reads the document's
headings: it prefers to break where a section does, and records the
heading path in force on each chunk.

```go
chunks := embedding.Split("nomic-embed-text", doc, embedding.SplitOptions{
    MaxBytes:  1024,
    Structure: embedding.StructureMarkdown,
})
// chunks[7].Headings == []string{"Deployment", "Rollback", "Manual steps"}
```

That path is what makes an isolated chunk searchable. A chunk reading
"this requires careful tuning of the retry budget" matches almost
nothing on its own; prefixed with its section path it matches what it is
actually about. Prepending it before embedding is measurably worth doing
-- [Anthropic reports a 35% reduction in retrieval failure](https://www.anthropic.com/engineering/contextual-retrieval)
from adding chunk context -- but it is left to the caller, because
splicing it into `Text` would break the guarantee that `Text` is exactly
`source[Start:End]`.

ATX (`## Section`) and setext (underlined) headings are both read, and
fenced code blocks are skipped so a shell comment or a diff marker does
not register as a heading.

Markdown is the only structured format, deliberately: it is trivial to
scan, structured enough to break on, and everything else converts to it.
Convert HTML or PDF text to markdown first rather than expecting this
library to grow a parser per format.

`Chunk.Text` is exactly `source[Start:End]`, so provenance -- which span
of which document a vector came from -- survives without recomputing
offsets that overlap makes impossible to derive.

**Pass `MaxBytes`.** It defaults to the model's full byte budget, which
is the only figure the library knows and almost never the one you want:
256-512 tokens per chunk is the usual working range, well under a
2048-token window. Long-context embedders mostly save you from building
a chunker; they do not generally retrieve better.

**Pooling is deliberately absent.** Whether to store one vector per
chunk or average them into one per document is a retrieval-design
decision with real consequences for scoring and dedup, and it belongs to
the caller's schema rather than to this library.

Chunking multiplies embed calls -- one long article can become dozens of
chunks -- so pair it with `BatchEmbedResults` rather than embedding
chunks one at a time. Chunks are uniform by construction, which makes
them a far better batching unit than whole documents.

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

`BatchEmbedResults` issues batch embed calls and falls back to
one-by-one when a backend returns either an error or fewer vectors
than inputs (some servers return 200 with a partial response). The
result slice is always the same length as the input, so the caller
keeps index alignment.

```go
results, err := embedding.BatchEmbedResults(ctx, e, texts, 25, func(done, total int) {
    log.Printf("embedded %d/%d", done, total)
})
for i, r := range results {
    if r.Err != nil {
        log.Printf("input %d failed: %v (retryable=%v)", i, r.Err, embedding.IsRetryable(r.Err))
        continue
    }
    store(texts[i], r.Vector)
}
```

It only returns a non-nil error if every input failed.

Each failure is an `*ItemError`, which unwraps to the cause of that
input's individual attempt and separately records `Batch` — the
batch-level failure, when the whole batch request failed rather than
just this input. That distinction is what tells one poisoned input
apart from a failing backend: an oversized input fails while its
neighbours succeed, whereas a backend that is down fails every input
in the batch with the same `Batch` cause. A caller draining a long
queue can stop on the latter instead of grinding through the rest to
fail identically.

The older `BatchEmbed` returns bare `[][]float32` with nil for failed
entries. It is deprecated: a nil entry can't be told apart from an
input the caller meant to skip, and it discards the cause needed to
decide whether a retry is worthwhile.

## Exact token counts

Every token budget in this library is otherwise converted to a byte
budget through a ratio -- a guess that is wrong by 50-70% on real text,
differs per corpus, and changes with every model. Supplying a
`TokenCounter` removes the ratio: input is truncated to the token budget
it actually has, and `Split` sizes chunks in the unit a caller reasons
in.

```go
cfg := embedding.Config{
    Backend:   embedding.BackendOllama,
    BaseURL:   "http://localhost:11434",
    Model:     "embeddinggemma",
    MaxTokens: 480,
    Tokenizer: myTokenizer, // implements CountTokens(string) int
}
```

No tokenizer ships with this library, deliberately. Vocabularies belong
to models, the Go bindings vary in weight and licence, and a vendored
one would have to be swapped in lockstep with every model change. Pure
Go implementations exist for the common families; wire one in through
the interface. `TokenCountFunc` adapts an ordinary function.

Implementations must be safe for concurrent use and monotonic over
prefixes -- a longer prefix of the same string never counts fewer
tokens. The truncation search relies on that and nothing else.

Without a tokenizer, `BytesPerToken` lets a caller supply a ratio
measured from their own corpus rather than the conservative built-in
guess, and `CalibrationFor` reports the ratio actually observed.

## Model names, aliases, and unknown models

Model identity decides two things: which task prefixes wrap the text,
and how large an input the model accepts. Both are looked up by name,
and both silently do nothing when the lookup misses -- a miss yields a
passthrough prompter and a zero `Limits`, so text reaches the model
unwrapped *and* nothing enforces a budget.

The second half is the dangerous one. A caller sizing chunks from the
registered budget will build inputs far past the model's context, and a
backend that rejects oversize input hard rather than truncating
(llama.cpp returns HTTP 500 at its physical batch limit) then fails
every long document, with nothing naming the cause.

Misses are the normal case, because serving runtimes rename models.
Ollama appends a `:tag`; Lemonade appends `-GGUF`, adds quantisation
markers, and takes a `user.` prefix on registration. None of those
describe a different model, but each defeats an exact-match lookup.

Packaging affixes are stripped automatically -- letter case, a tag, a
`user.` prefix, a `-GGUF`/`.gguf` suffix. Version and mixture markers
are **not**: `-v2` and `-moe` denote a different model, and resolving
one model to another's prefixes and budget is worse than not resolving
at all. Anything beyond packaging needs an explicit alias, because only
the operator knows whether two names mean the same weights:

```go
embedding.RegisterModelAlias("nomic-embed-text-v1-GGUF", "nomic-embed-text")
```

or, without a code change:

```
EMBEDDING_MODEL_ALIAS=nomic-embed-text-v1-GGUF=nomic-embed-text,EmbeddingGemma-300M-GGUF=embeddinggemma
```

An alias resolves prefixes and limits only. It does not rename the
model: `Embedder.Model()` keeps reporting the served name, because
stored vectors are keyed by it and vectors from different quantisations
are not interchangeable.

`LookupModel` reports what a name actually resolved to, so "which prefix
and which budget am I using?" is answerable at startup rather than
inferred from bad results later:

```go
info, ok := embedding.LookupModel(cfg.Model)
// info.Canonical, info.Limits, info.HasPrompts; ok == false when the
// library knows nothing about the model.
```

`Config.StrictModel` turns that check into a construction error, so an
unrecognised model is a boot failure rather than a quiet quality
regression.

### Task prefixes by family

Registered models use different schemes, and swapping models without
swapping prefixes is not a small mistake:

| task | `nomic-embed-text` | `embeddinggemma` |
|---|---|---|
| RetrievalDocument | `search_document:` | `title: none \| text:` |
| RetrievalQuery | `search_query:` | `task: search result \| query:` |
| Clustering | `clustering:` | `task: clustering \| query:` |
| Classification | `classification:` | `task: classification \| query:` |
| Similarity | *(none)* | `task: sentence similarity \| query:` |
| QuestionAnswering | *(none)* | `task: question answering \| query:` |
| FactChecking | *(none)* | `task: fact checking \| query:` |
| CodeRetrieval | *(none)* | `task: code retrieval \| query:` |

Both sides of a comparison must use the same convention: a query
embedded with no prefix, or with a different task, is being compared
across a boundary the model was trained to distinguish.


## Compatibility

`NewOllamaEmbedder` and `NewOpenAIEmbedder` are still exported but
marked `Deprecated`. They will be removed in v1.0; new code should use
`New(Config)`.

`BatchEmbed` is likewise deprecated in favour of `BatchEmbedResults`.
Its behaviour is unchanged — it is now a thin wrapper that drops the
per-input errors — so existing callers keep working until v1.0.
