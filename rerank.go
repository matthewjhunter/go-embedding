package embedding

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
)

// =============================================================================
// SPECULATIVE — DO NOT TREAT AS STABLE.
//
// This Reranker interface, its result type, and its configuration are a first
// sketch. They exist before any consumer has wired rerank into a real
// retrieval pipeline, so the shapes below are guesses about what callers will
// need, not contracts. Expect breaking revision as the first consumers (the
// search module, possibly memstore recall) land and tell us what they
// actually want.
//
// Several shape questions have been resolved against the needs of the first
// consumer (a two-stage search function); the contracts they settled are now
// stated on Rerank, RerankResult, and RerankConfig below:
//   - Return shape: sorted by descending Score, each carrying its input Index.
//     Aligned 1:1 with input order was rejected because topN > 0 returns fewer
//     results than inputs, and the rerank backends return sorted-with-index
//     natively; Index already gives callers full composability.
//   - Call shape: Rerank takes a RerankRequest struct, not positional args, so
//     optional knobs (TopN, Instruction) and future additions don't force a
//     breaking signature change each time. This intentionally diverges from
//     Embedder.Embed's positional args — Embed has one input; Rerank has
//     several optional ones.
//   - TopN cutoff: a RerankRequest field, not a Config field, because the
//     cutoff is intrinsically per-query.
//   - Instruction: an optional RerankRequest field for instruction-tuned
//     rerankers (Qwen3-Reranker, mxbai-rerank-v2). Cross-encoders such as
//     bge-reranker-v2-m3 ignore it. It is the natural-language task string, NOT
//     a prompt template — the serving stack owns the model's template. How the
//     string reaches the model is backend-specific (the Cohere/Jina wire
//     protocol has no instruction field, so an implementation typically folds
//     it into the query).
//   - Over-length input: Strict on RerankConfig mirrors Config, choosing
//     truncate (default) vs. error for query+document pairs exceeding the
//     model's max sequence length.
//   - Large candidate sets: Rerank accepts an arbitrarily long documents slice
//     and fans out internally; callers need not pre-chunk.
//   - Graceful degradation: the library does NOT silently fake a passthrough
//     when the backend is down — that would fabricate scores and hide outages.
//     Rerank returns a real error; an availability failure (refused/timed-out
//     connection, HTTP 5xx, HTTP 429, a per-call deadline) wraps
//     ErrRerankUnavailable, recognised by IsRerankUnavailable. The consumer
//     owns the fallback (only it holds the first-stage order and scores) and,
//     on an unavailable error, keeps that first-stage order — collapsing into
//     the same path it uses when no Reranker is configured. A 4xx request error
//     is a caller bug and surfaces (not unavailable), so a broken deploy is not
//     mistaken for a healthy one with poor relevance.
//
// Known-open questions, to resolve against real usage rather than in advance:
//   - Score semantics: raw model logits are not comparable across models and
//     are not bounded to [0,1]. Do callers need a normalized score, or is
//     "higher is better, scale is model-specific" enough? Matters only for
//     score fusion or thresholding; pure reordering does not need it.
//   - Env wiring: a ConfigFromEnvPrefix analog (RerankConfigFromEnvPrefix) is
//     intentionally deferred until the constructor shape settles.
//
// Backend protocol note: unlike embeddings (which standardized on the OpenAI
// shape), there is NO OpenAI rerank endpoint. Reranking standardized on the
// Cohere/Jina shape ({query, documents, top_n} -> {results:[{index,
// relevance_score}]}), which llama.cpp, vLLM, Infinity, TEI (a near variant),
// and the Cohere/Jina clouds all implement. Supporting that one shape is the
// broad option. Note that Ollama and Lemonade expose NO rerank endpoint at all
// — they are not rerank backends, so a card serving only Ollama cannot rerank
// without also running one of the servers above.
//
// Unlike Embedder, Reranker has no Fingerprint. A reranker is stateless at the
// corpus level: it produces a per-query score and persists nothing, so there
// is no stored artifact a model swap could silently corrupt. Model-swap safety
// is the embedder's problem, not the reranker's.
// =============================================================================

// RerankBackend identifies which reranking backend a Reranker uses.
//
// Speculative: the set and the names will change as backends are implemented.
type RerankBackend string

const (
	// RerankBackendJina dispatches to a Cohere/Jina-style rerank endpoint
	// (POST {BaseURL}/v1/rerank or /rerank), the de-facto standard rerank
	// protocol. This is the broad backend: llama.cpp (--reranking), vLLM,
	// Infinity, and the Cohere and Jina clouds all speak it. Prefer this
	// unless a server needs the TEI variant.
	RerankBackendJina RerankBackend = "jina"
	// RerankBackendTEI dispatches to a Text Embeddings Inference rerank
	// server (POST {BaseURL}/rerank). TEI's request/response differ enough
	// from the Cohere/Jina shape to warrant a separate path.
	RerankBackendTEI RerankBackend = "tei"
)

// RerankResult is one scored document from a Rerank call. A Rerank call
// returns these sorted by descending Score, each carrying the Index of the
// document in the slice originally passed to Rerank, so the caller can map a
// result back to its source document. To recover input order, or to blend
// Score with a first-stage score, index into the original slice by Index.
type RerankResult struct {
	// Index is the position of this document in the documents slice passed
	// to Rerank.
	Index int
	// Score is the model's relevance score for (query, document). Higher is
	// more relevant. The scale is model-specific: scores are not bounded to
	// a fixed range and are not comparable across different rerank models.
	Score float64
}

// Reranker rescores a set of candidate documents against a query using a
// cross-encoder model. It runs at query time over a shortlist already
// retrieved by a cheaper first stage (vector / full-text search); it is not a
// retriever and does not index anything.
//
// Speculative interface — see the banner at the top of this file.
type Reranker interface {
	// Rerank scores each document in req.Documents against req.Query and
	// returns the results sorted by descending Score. See RerankRequest for
	// the per-call options (TopN, Instruction) and their semantics.
	//
	// An empty Documents slice returns nil and no error, with no round-trip to
	// the backend. Documents may be arbitrarily long: implementations fan out
	// across multiple backend requests as needed and merge, so callers need
	// not pre-chunk to a backend's per-request limit.
	//
	// A query+document pair exceeding the model's maximum sequence length is
	// truncated by default; see RerankConfig.Strict to make that an error
	// instead.
	//
	// When the backend is unavailable (connection refused or timed out, HTTP
	// 5xx or 429, or ctx's deadline elapsed), Rerank returns an error for which
	// IsRerankUnavailable reports true; the caller should degrade to its
	// first-stage ordering rather than fail the query. Other errors — notably a
	// 4xx request error — are not unavailable and should surface.
	Rerank(ctx context.Context, req RerankRequest) ([]RerankResult, error)
	// Model returns a stable identifier for the rerank model (e.g.
	// "bge-reranker-v2-m3"). It need not — and generally will not — match the
	// embedding model used for first-stage retrieval; rerankers and embedders
	// are chosen independently and share no vector space.
	Model() string
}

// RerankRequest is a single Rerank call. Query and Documents are required;
// TopN and Instruction are optional. It is a struct rather than positional
// arguments so that new optional fields can be added without breaking the
// Reranker interface.
type RerankRequest struct {
	// Query is the search query the documents are scored against.
	Query string
	// Documents is the candidate shortlist to rescore — for hybrid search,
	// the merged-and-deduped union of the first-stage (vector + full-text)
	// results. Each RerankResult.Index refers back into this slice.
	Documents []string
	// TopN, if > 0, limits the result to the TopN highest-scoring documents.
	// TopN <= 0 returns a result for every document. Either way each result's
	// Index refers to the document's position in Documents.
	TopN int
	// Instruction is an optional natural-language task description for
	// instruction-tuned rerankers (e.g. Qwen3-Reranker, mxbai-rerank-v2),
	// such as "Given a support question, rank the most relevant docs". Plain
	// cross-encoders (e.g. bge-reranker-v2-m3) ignore it. It is NOT a prompt
	// template: the serving stack owns the model's template, and the backend
	// decides how the string reaches the model (the Cohere/Jina wire protocol
	// has no instruction field, so it is typically folded into the query).
	Instruction string
}

// RerankConfig configures a new Reranker. It mirrors Config (the embedder
// config) on purpose so that callers configuring both halves of a retrieval
// pipeline see a consistent shape.
//
// Backend, BaseURL, and Model are required. APIKey is optional and only
// meaningful for backends that authenticate.
type RerankConfig struct {
	Backend RerankBackend
	BaseURL string
	APIKey  string
	Model   string
	// Strict is intended to control how the reranker reacts to a query+document
	// pair that exceeds the model's maximum sequence length: truncate when
	// false (default), error when true, mirroring Config.Strict.
	//
	// Not yet enforced: the Jina backend leaves truncation to the serving
	// stack because reranker sequence budgets are not registered in limits.go
	// and the Cohere/Jina wire protocol reports no truncation. The field is
	// reserved so enabling client-side enforcement later is not a breaking
	// change. Keep rerank shortlists chunked upstream until then.
	Strict bool
}

// NewReranker constructs a Reranker from cfg. Returns an error if any required
// field is missing or if Backend is not recognised.
//
// RerankBackendJina is implemented (see JinaReranker); RerankBackendTEI is not
// yet, since one Cohere/Jina implementation already covers the broadest set of
// servers. cfg.Strict is carried on RerankConfig but not yet enforced by the
// Jina backend — see JinaReranker.
func NewReranker(cfg RerankConfig) (Reranker, error) {
	switch {
	case cfg.Backend == "":
		return nil, fmt.Errorf("embedding: rerank Backend is required")
	case cfg.BaseURL == "":
		return nil, fmt.Errorf("embedding: rerank BaseURL is required")
	case cfg.Model == "":
		return nil, fmt.Errorf("embedding: rerank Model is required")
	}

	switch cfg.Backend {
	case RerankBackendJina:
		return NewJinaReranker(cfg.BaseURL, cfg.APIKey, cfg.Model), nil
	case RerankBackendTEI:
		return nil, fmt.Errorf("embedding: rerank backend %q not yet implemented", cfg.Backend)
	default:
		return nil, fmt.Errorf("embedding: unknown rerank backend %q", cfg.Backend)
	}
}

// ErrRerankUnavailable marks a rerank failure caused by the backend being
// unreachable or unhealthy — a refused or timed-out connection, an HTTP 5xx, or
// an HTTP 429 — rather than by a malformed request. It is the signal a consumer
// uses to degrade gracefully: when Rerank fails with an error for which
// IsRerankUnavailable reports true, the consumer keeps its first-stage (e.g.
// RRF-fused) ordering instead of failing the query. Errors that do not wrap it
// (notably a 4xx request error, surfaced as a *PermanentError) are caller bugs
// and must surface rather than silently degrade.
//
// A backend wraps it at two sites: transport failures from the HTTP round trip
// (use fmt.Errorf("%w: %w", ErrRerankUnavailable, err) so a refused connection
// — which is not a timeout and so not caught by IsRerankUnavailable on its own
// — is recognised), and 429/5xx responses via classifyRerankHTTPError.
var ErrRerankUnavailable = errors.New("embedding: rerank backend unavailable")

// IsRerankUnavailable reports whether err indicates the rerank backend was
// unavailable, so the caller should degrade to first-stage ordering instead of
// failing the query. It is true for errors wrapping ErrRerankUnavailable and
// for the transport-level failures a backend may surface unwrapped: a per-call
// deadline (context.DeadlineExceeded) or any net.Error reporting a timeout. A
// 4xx request error (a *PermanentError) is not unavailable — that is a caller
// bug and must surface. context.Canceled is likewise not unavailable: a
// cancelled call was abandoned by the caller, not failed by the backend.
func IsRerankUnavailable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrRerankUnavailable) {
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// classifyRerankHTTPError classifies a non-2xx rerank response. A 429 or 5xx
// means the backend is reachable but unhealthy and wraps ErrRerankUnavailable
// so the consumer can degrade to first-stage ordering. A 4xx is the caller's
// problem (bad request, unknown model, auth, oversize pair) and surfaces as a
// *PermanentError, with TooLong set when the body indicates the query+document
// pair exceeded the model's max sequence length. It mirrors classifyHTTPError;
// the 429 split is rerank-specific — backpressure from a rerank sidecar is an
// availability signal to retry/degrade on, not a permanent rejection.
func classifyRerankHTTPError(err error, status int, body []byte) error {
	switch {
	case status == http.StatusTooManyRequests, status >= 500:
		return fmt.Errorf("%w: %w", ErrRerankUnavailable, err)
	case status >= 400:
		return &PermanentError{Err: err, TooLong: isContextLengthError(status, body)}
	}
	return err
}
