package embedding

import (
	"context"
	"errors"
	"fmt"
	"math"
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
//     ErrRerankUnavailable, for which IsRerankAvailable reports false. The consumer
//     owns the fallback (only it holds the first-stage order and scores) and,
//     on an unavailable error, keeps that first-stage order — collapsing into
//     the same path it uses when no Reranker is configured. A 4xx request error
//     is a caller bug and surfaces (not unavailable), so a broken deploy is not
//     mistaken for a healthy one with poor relevance.
//
// Env wiring is resolved: RerankConfigFromEnv / RerankConfigFromEnvPrefix read
// the RERANK_* namespace, mirroring ConfigFromEnv over EMBEDDING_* but kept
// separate (the reranker endpoint and model are chosen independently).
//
// Score normalization is OFF by default but available as an opt-in. Scores
// pass through raw unless RerankConfig.NormalizeScores is set, because their
// scale depends on the serving stack as much as the model (Cohere/Jina/TEI
// return [0,1]; llama.cpp returns raw logits), and the wire protocol carries no
// flag distinguishing the two — so a default sigmoid would double-normalize the
// already-bounded backends, and min-max would destroy the absolute signal a
// relevance floor needs. The bit the wire cannot supply is operator-declared:
// a consumer running a raw-logit stack (e.g. llama.cpp --reranking) sets
// NormalizeScores, and the library applies a sigmoid in one place
// (normalizingReranker), so every consumer of that deployment shares the
// transform instead of reimplementing it. The transform is keyed off this
// declared bit, NOT the model name: every cross-encoder reranker emits a
// relevance logit whose canonical map to [0,1] is the same sigmoid, and the
// model name does not reveal whether the stack already applied it (the same
// model returns raw on llama.cpp and bounded on TEI). The sigmoid is
// order-preserving, so it never changes ranking — only the scale. Thresholding,
// when wanted, is still the consumer's policy: an operator-calibrated floor,
// per deployment, not a portable constant (see RerankResult.Score).
//
// Known-open questions, to resolve against real usage rather than in advance:
//   - Strict enforcement: client-side over-length truncation is deferred until
//     reranker sequence budgets are registered; the serving stack truncates
//     for now (see RerankConfig.Strict, JinaReranker).
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
	// more relevant, so sorting by descending Score is always meaningful.
	//
	// The scale, however, is not: it depends on both the model AND the serving
	// stack. The same model returns a sigmoid-bounded [0,1] score on one server
	// (Cohere/Jina, TEI by default) and an unbounded raw logit on another
	// (llama.cpp --reranking), and the wire protocol carries no flag saying
	// which. So Score is comparable WITHIN one Rerank call but not across
	// models or backends. A consumer applying a relevance floor (threshold)
	// must therefore calibrate it per deployment and treat it as operator
	// config — do NOT hardcode a portable cutoff. Pure reordering needs none of
	// this; only thresholding or score fusion does. See the banner's score-
	// semantics note for why the library does not normalize by default.
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
	// IsRerankAvailable reports false; the caller should degrade to its
	// first-stage ordering rather than fail the query. Other errors — notably a
	// 4xx request error — leave IsRerankAvailable true and should surface.
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
	// MaxDocumentBytes, when > 0, truncates each document to this many bytes
	// before scoring, overriding the model's registered byte budget for this
	// call. Cross-encoder latency is superlinear in sequence length, so a caller
	// on a tight budget (e.g. a per-prompt recall path) can truncate hard while a
	// caller that values relevance over latency leaves it 0 (use the registered
	// budget). Truncation ranks each document on its lead content; it is lossy
	// for signal buried in long tails.
	MaxDocumentBytes int
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
	// Strict controls how the Jina backend reacts to a document that exceeds the
	// model's registered byte budget (see limits.go, e.g. bge-reranker-v2-m3):
	// truncate to the budget and log when false (default), or reject with a
	// *PermanentError when true, mirroring Config.Strict. A model with no
	// registered budget is never truncated regardless of this flag.
	Strict bool
	// NormalizeScores, when true, maps each raw relevance score to [0,1] via a
	// sigmoid (1/(1+e^-x)) before returning it, by wrapping the backend in a
	// normalizingReranker. It is operator-declared rather than inferred because
	// the wire protocol cannot distinguish a raw-logit serving stack (llama.cpp
	// --reranking) from one that already bounds scores (Cohere/Jina/TEI): set it
	// only for a raw-logit deployment, or it will double-normalize an
	// already-bounded one. The sigmoid is monotonic, so it never reorders
	// results — it only puts the score on a bounded scale a consumer can fuse
	// with a first-stage score or threshold against. Default false (raw
	// passthrough), which preserves the score the backend returned verbatim.
	NormalizeScores bool
}

// NewReranker constructs a Reranker from cfg. Returns an error if any required
// field is missing or if Backend is not recognised.
//
// RerankBackendJina is implemented (see JinaReranker); RerankBackendTEI is not
// yet, since one Cohere/Jina implementation already covers the broadest set of
// servers. cfg.Strict is carried on RerankConfig but not yet enforced by the
// Jina backend — see JinaReranker. When cfg.NormalizeScores is set, the
// constructed backend is wrapped in a normalizingReranker so the [0,1] sigmoid
// applies to every backend uniformly.
func NewReranker(cfg RerankConfig) (Reranker, error) {
	switch {
	case cfg.Backend == "":
		return nil, fmt.Errorf("embedding: rerank Backend is required")
	case cfg.BaseURL == "":
		return nil, fmt.Errorf("embedding: rerank BaseURL is required")
	case cfg.Model == "":
		return nil, fmt.Errorf("embedding: rerank Model is required")
	}

	var rr Reranker
	switch cfg.Backend {
	case RerankBackendJina:
		jr := NewJinaReranker(cfg.BaseURL, cfg.APIKey, cfg.Model)
		jr.strict = cfg.Strict
		rr = jr
	case RerankBackendTEI:
		return nil, fmt.Errorf("embedding: rerank backend %q not yet implemented", cfg.Backend)
	default:
		return nil, fmt.Errorf("embedding: unknown rerank backend %q", cfg.Backend)
	}

	if cfg.NormalizeScores {
		rr = normalizingReranker{inner: rr}
	}
	return rr, nil
}

// normalizingReranker wraps a Reranker and maps each raw relevance score to
// [0,1] via a sigmoid. It is backend-agnostic by design: the transform lives
// here once rather than in each backend, so any Reranker (Jina today, TEI
// later) gets it for free. See RerankConfig.NormalizeScores for why this is
// opt-in and operator-declared.
type normalizingReranker struct {
	inner Reranker
}

// Rerank delegates to the wrapped Reranker, then sigmoid-maps each Score.
// Because sigmoid is monotonic, the descending-Score order the inner Reranker
// already established is preserved, so no re-sort is needed. An error (including
// an ErrRerankUnavailable) passes through untouched so the consumer's
// degradation logic still sees it.
func (n normalizingReranker) Rerank(ctx context.Context, req RerankRequest) ([]RerankResult, error) {
	results, err := n.inner.Rerank(ctx, req)
	if err != nil {
		return nil, err
	}
	for i := range results {
		results[i].Score = sigmoid(results[i].Score)
	}
	return results, nil
}

// Model reports the wrapped reranker's model; normalization does not change it.
func (n normalizingReranker) Model() string { return n.inner.Model() }

// sigmoid maps a raw cross-encoder logit to a [0,1] relevance score. It is the
// canonical transform for rerankers that emit unbounded logits (e.g. bge,
// mxbai, Qwen3-Reranker served by llama.cpp --reranking).
func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// ErrRerankUnavailable marks a rerank failure caused by the backend being
// unreachable or unhealthy — a refused or timed-out connection, an HTTP 5xx, or
// an HTTP 429 — rather than by a malformed request. It is the signal a consumer
// uses to degrade gracefully: when Rerank fails with an error for which
// IsRerankAvailable reports false, the consumer keeps its first-stage (e.g.
// RRF-fused) ordering instead of failing the query. Errors that do not wrap it
// (notably a 4xx request error, surfaced as a *PermanentError) are caller bugs
// and must surface rather than silently degrade.
//
// A backend wraps it at two sites: transport failures from the HTTP round trip
// (use fmt.Errorf("%w: %w", ErrRerankUnavailable, err) so a refused connection
// — which is not a timeout and so not flagged by IsRerankAvailable on its own
// — is recognised), and 429/5xx responses via classifyRerankHTTPError.
var ErrRerankUnavailable = errors.New("embedding: rerank backend unavailable")

// IsRerankAvailable reports whether err indicates the rerank backend was
// reachable and healthy. It is the predicate a consumer inverts to decide
// between surfacing and degrading: when Rerank fails with an error for which
// IsRerankAvailable reports false, the caller should degrade to first-stage
// ordering instead of failing the query.
//
// It returns false for errors wrapping ErrRerankUnavailable and for the
// transport-level failures a backend may surface unwrapped: a per-call deadline
// (context.DeadlineExceeded) or any net.Error reporting a timeout. It returns
// true otherwise, including for: a nil error (the call succeeded); a 4xx request
// error (a *PermanentError — the backend was reached, the request was malformed,
// a caller bug that must surface, not an outage); and context.Canceled (the
// caller abandoned the call, so nothing indicates the backend is down).
func IsRerankAvailable(err error) bool {
	if err == nil {
		return true
	}
	if errors.Is(err, ErrRerankUnavailable) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var ne net.Error
	return !errors.As(err, &ne) || !ne.Timeout()
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
