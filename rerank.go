package embedding

import (
	"context"
	"fmt"
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
	// Strict controls how the reranker reacts to a query+document pair that
	// exceeds the model's maximum sequence length. When false (default), the
	// over-length pair is truncated to fit. When true, Rerank returns an error
	// instead of scoring a truncated pair. Mirrors Config.Strict.
	Strict bool
}

// NewReranker constructs a Reranker from cfg. Returns an error if any required
// field is missing or if Backend is not recognised.
//
// Speculative: no backend is implemented yet. The cases below document the
// intended dispatch; each currently returns a not-implemented error so the
// interface can be designed against before the HTTP plumbing is written. The
// real implementations should reuse the existing backend machinery
// (classifyHTTPError, the retry/limits helpers) rather than duplicating it.
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
		return nil, fmt.Errorf("embedding: rerank backend %q not yet implemented", cfg.Backend)
	case RerankBackendTEI:
		return nil, fmt.Errorf("embedding: rerank backend %q not yet implemented", cfg.Backend)
	default:
		return nil, fmt.Errorf("embedding: unknown rerank backend %q", cfg.Backend)
	}
}
