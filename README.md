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

## Limits

`Embed` consults a per-model byte limit registered for nomic-embed-text
and a few siblings. Oversize input is truncated to the limit (with a
`log.Printf` so the truncation is visible in logs). Set `Strict: true`
on `Config` to make oversize input an error instead.

Register custom models at startup:

```go
embedding.RegisterLimits("my-custom-embedder", embedding.Limits{MaxBytes: 4096})
```

Models not in the registry get no enforcement.

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
