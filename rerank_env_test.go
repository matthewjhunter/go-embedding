package embedding

import (
	"strings"
	"testing"
)

// clearRerankEnv blanks every canonical RERANK_* key so a test is hermetic
// regardless of the caller's environment. t.Setenv handles cleanup.
func clearRerankEnv(t *testing.T) {
	t.Helper()
	for _, suffix := range []string{
		envSuffixBackend, envSuffixBaseURL, envSuffixAPIKey, envSuffixModel, envSuffixStrict,
		envSuffixNormalizeScores,
	} {
		t.Setenv(DefaultRerankEnvPrefix+suffix, "")
	}
}

func TestRerankConfigFromEnv_NoVars_DefaultsBackendOnly(t *testing.T) {
	clearRerankEnv(t)

	cfg, err := RerankConfigFromEnv()
	if err != nil {
		t.Fatalf("RerankConfigFromEnv: %v", err)
	}
	want := RerankConfig{Backend: RerankBackendJina}
	if cfg != want {
		t.Errorf("got %+v, want %+v", cfg, want)
	}
}

func TestRerankConfigFromEnv_OverridesAllFields(t *testing.T) {
	t.Setenv("RERANK_BACKEND", "tei")
	t.Setenv("RERANK_BASE_URL", "http://cube:8080")
	t.Setenv("RERANK_API_KEY", "rk-test")
	t.Setenv("RERANK_MODEL", "bge-reranker-v2-m3")
	t.Setenv("RERANK_STRICT", "true")
	t.Setenv("RERANK_NORMALIZE_SCORES", "true")

	cfg, err := RerankConfigFromEnv()
	if err != nil {
		t.Fatalf("RerankConfigFromEnv: %v", err)
	}
	want := RerankConfig{
		Backend:         RerankBackendTEI,
		BaseURL:         "http://cube:8080",
		APIKey:          "rk-test",
		Model:           "bge-reranker-v2-m3",
		Strict:          true,
		NormalizeScores: true,
	}
	if cfg != want {
		t.Errorf("got %+v, want %+v", cfg, want)
	}
}

func TestRerankConfigFromEnvPrefix_CascadesToCanonical(t *testing.T) {
	clearRerankEnv(t)
	// Canonical fallback set; prefix-specific overrides only some fields.
	t.Setenv("RERANK_BASE_URL", "http://canonical:8080")
	t.Setenv("RERANK_MODEL", "canonical-model")
	t.Setenv("SEARCH_RERANK_MODEL", "search-model")

	cfg, err := RerankConfigFromEnvPrefix("SEARCH_RERANK")
	if err != nil {
		t.Fatalf("RerankConfigFromEnvPrefix: %v", err)
	}
	if cfg.BaseURL != "http://canonical:8080" {
		t.Errorf("BaseURL = %q, want canonical fallback", cfg.BaseURL)
	}
	if cfg.Model != "search-model" {
		t.Errorf("Model = %q, want prefix override", cfg.Model)
	}
}

func TestRerankConfigFromEnv_DoesNotInheritEmbeddingNamespace(t *testing.T) {
	clearRerankEnv(t)
	// An EMBEDDING_* endpoint must never leak into the rerank config: the
	// sidecar is a different server than the embedder.
	t.Setenv("EMBEDDING_BASE_URL", "http://embed-host:11434")
	t.Setenv("EMBEDDING_MODEL", "nomic-embed-text")

	cfg, err := RerankConfigFromEnv()
	if err != nil {
		t.Fatalf("RerankConfigFromEnv: %v", err)
	}
	if cfg.BaseURL != "" {
		t.Errorf("BaseURL = %q, want empty (EMBEDDING_* leaked in)", cfg.BaseURL)
	}
	if cfg.Model != "" {
		t.Errorf("Model = %q, want empty (EMBEDDING_* leaked in)", cfg.Model)
	}
}

func TestRerankConfigFromEnv_UnknownBackend(t *testing.T) {
	clearRerankEnv(t)
	t.Setenv("RERANK_BACKEND", "cohere-v3")

	_, err := RerankConfigFromEnv()
	if err == nil {
		t.Fatal("expected an error for unknown backend")
	}
	if !strings.Contains(err.Error(), "unknown rerank backend") {
		t.Errorf("error = %v, want 'unknown rerank backend'", err)
	}
}

func TestRerankConfigFromEnv_InvalidStrict(t *testing.T) {
	clearRerankEnv(t)
	t.Setenv("RERANK_STRICT", "yep")

	_, err := RerankConfigFromEnv()
	if err == nil {
		t.Fatal("expected an error for malformed bool")
	}
	if !strings.Contains(err.Error(), "RERANK_STRICT") {
		t.Errorf("error = %v, want it to name RERANK_STRICT", err)
	}
}

func TestRerankConfigFromEnv_InvalidNormalizeScores(t *testing.T) {
	clearRerankEnv(t)
	t.Setenv("RERANK_NORMALIZE_SCORES", "maybe")

	_, err := RerankConfigFromEnv()
	if err == nil {
		t.Fatal("expected an error for malformed bool")
	}
	if !strings.Contains(err.Error(), "RERANK_NORMALIZE_SCORES") {
		t.Errorf("error = %v, want it to name RERANK_NORMALIZE_SCORES", err)
	}
}
