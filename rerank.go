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
// Known-open questions, to resolve against real usage rather than in advance:
//   - Return shape: results sorted by score (as drafted) vs. aligned 1:1 with
//     the input order. Sorted matches Cohere/Jina; aligned is more composable.
//   - Score semantics: raw model logits are not comparable across models and
//     are not bounded to [0,1]. Do callers need a normalized score, or is
//     "higher is better, scale is model-specific" enough?
//   - TopN: push the cutoff into the backend request (Cohere/Jina support it)
//     vs. return all and let the caller slice. Drafted as a Config field; may
//     belong on a per-call options arg instead.
//   - Backends: rerank endpoints differ from embed endpoints and Ollama has no
//     native rerank, so RerankBackend is deliberately a separate enum from
//     Backend rather than a reuse of it.
//   - Env wiring: a ConfigFromEnvPrefix analog (RerankConfigFromEnvPrefix) is
//     intentionally deferred until the constructor shape settles.
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
	// RerankBackendTEI dispatches to a Text Embeddings Inference rerank
	// server (POST {BaseURL}/rerank). Also covers servers that copy its
	// protocol.
	RerankBackendTEI RerankBackend = "tei"
	// RerankBackendJina dispatches to a Jina-style rerank endpoint
	// (POST {BaseURL}/v1/rerank), which Cohere's API also closely matches.
	RerankBackendJina RerankBackend = "jina"
)

// RerankResult is one scored document from a Rerank call.
//
// Speculative: see the open question on return shape above. As drafted, a
// Rerank call returns these sorted by descending Score, each carrying the
// Index of the document in the slice originally passed to Rerank, so the
// caller can map a result back to its source document.
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
	// Rerank scores each document in documents against query and returns the
	// results sorted by descending Score. The returned slice has one entry
	// per input document (subject to the open TopN question above).
	Rerank(ctx context.Context, query string, documents []string) ([]RerankResult, error)
	// Model returns a stable identifier for the rerank model (e.g.
	// "bge-reranker-v2-m3"). It need not — and generally will not — match the
	// embedding model used for first-stage retrieval; rerankers and embedders
	// are chosen independently and share no vector space.
	Model() string
}

// RerankConfig configures a new Reranker. It mirrors Config (the embedder
// config) on purpose so that callers configuring both halves of a retrieval
// pipeline see a consistent shape.
//
// Backend, BaseURL, and Model are required. APIKey is optional and only
// meaningful for backends that authenticate.
//
// Speculative — TopN in particular may move to a per-call argument.
type RerankConfig struct {
	Backend RerankBackend
	BaseURL string
	APIKey  string
	Model   string
	// TopN, if > 0, asks the backend to return only its top N results.
	// Zero means return a score for every document. Speculative: see the
	// TopN open question above.
	TopN int
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
	case RerankBackendTEI:
		return nil, fmt.Errorf("embedding: rerank backend %q not yet implemented", cfg.Backend)
	case RerankBackendJina:
		return nil, fmt.Errorf("embedding: rerank backend %q not yet implemented", cfg.Backend)
	default:
		return nil, fmt.Errorf("embedding: unknown rerank backend %q", cfg.Backend)
	}
}
