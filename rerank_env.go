package embedding

import (
	"fmt"
	"strconv"
	"strings"
)

// DefaultRerankEnvPrefix is the canonical prefix RerankConfigFromEnv reads. It
// is deliberately separate from DefaultEnvPrefix (EMBEDDING): the reranker runs
// on a different endpoint and model than the embedder, so the two must not
// share BASE_URL/MODEL/API_KEY.
const DefaultRerankEnvPrefix = "RERANK"

// RerankConfigFromEnv reads RerankConfig from RERANK_BACKEND, RERANK_BASE_URL,
// RERANK_API_KEY, RERANK_MODEL, RERANK_STRICT, and RERANK_NORMALIZE_SCORES.
//
// Backend defaults to the Cohere/Jina shape (RerankBackendJina). Unlike
// embeddings there is no ecosystem default endpoint or model — those are
// operator config — so BaseURL and Model are not defaulted here; NewReranker
// rejects them if still empty. Missing vars are not errors; only parse
// failures (unknown backend, malformed bool) are.
func RerankConfigFromEnv() (RerankConfig, error) {
	return RerankConfigFromEnvPrefix(DefaultRerankEnvPrefix)
}

// RerankConfigFromEnvPrefix is RerankConfigFromEnv with a caller-supplied
// prefix and a per-field fallback chain: {prefix}_FOO → RERANK_FOO. It mirrors
// ConfigFromEnvPrefix over the RERANK_* namespace.
//
// E.g. RerankConfigFromEnvPrefix("SEARCH_RERANK") reads SEARCH_RERANK_BASE_URL
// first, then RERANK_BASE_URL. The chain stops at RERANK_*; it never falls back
// to EMBEDDING_*, because the rerank endpoint and model are chosen
// independently of the embedder.
func RerankConfigFromEnvPrefix(prefix string) (RerankConfig, error) {
	cfg := RerankConfig{Backend: RerankBackendJina}

	if v, src := envCascadeTo(prefix, DefaultRerankEnvPrefix, envSuffixBackend); v != "" {
		switch strings.ToLower(v) {
		case string(RerankBackendJina):
			cfg.Backend = RerankBackendJina
		case string(RerankBackendTEI):
			cfg.Backend = RerankBackendTEI
		default:
			return RerankConfig{}, fmt.Errorf(
				"embedding: unknown rerank backend %q in %s (want jina|tei)",
				v, src,
			)
		}
	}
	if v, _ := envCascadeTo(prefix, DefaultRerankEnvPrefix, envSuffixBaseURL); v != "" {
		cfg.BaseURL = v
	}
	if v, _ := envCascadeTo(prefix, DefaultRerankEnvPrefix, envSuffixAPIKey); v != "" {
		cfg.APIKey = v
	}
	if v, _ := envCascadeTo(prefix, DefaultRerankEnvPrefix, envSuffixModel); v != "" {
		cfg.Model = v
	}
	if v, src := envCascadeTo(prefix, DefaultRerankEnvPrefix, envSuffixStrict); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return RerankConfig{}, fmt.Errorf(
				"embedding: invalid %s value %q: %w",
				src, v, err,
			)
		}
		cfg.Strict = b
	}
	if v, src := envCascadeTo(prefix, DefaultRerankEnvPrefix, envSuffixNormalizeScores); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return RerankConfig{}, fmt.Errorf(
				"embedding: invalid %s value %q: %w",
				src, v, err,
			)
		}
		cfg.NormalizeScores = b
	}
	return cfg, nil
}
