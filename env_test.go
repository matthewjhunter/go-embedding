package embedding

import (
	"strings"
	"testing"
)

func TestConfigFromEnv_NoVars_ReturnsDefault(t *testing.T) {
	// t.Setenv guarantees cleanup; explicit unsets here would silently
	// leak the *parent* env. Override each canonical key with empty so
	// the test is hermetic regardless of caller env.
	t.Setenv("EMBEDDING_BACKEND", "")
	t.Setenv("EMBEDDING_BASE_URL", "")
	t.Setenv("EMBEDDING_API_KEY", "")
	t.Setenv("EMBEDDING_MODEL", "")
	t.Setenv("EMBEDDING_STRICT", "")

	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatalf("ConfigFromEnv: %v", err)
	}
	// Empty-string env values should be treated as "not set" so we get
	// the same Config as DefaultConfig.
	if cfg != DefaultConfig() {
		t.Errorf("got %+v, want DefaultConfig %+v", cfg, DefaultConfig())
	}
}

func TestConfigFromEnv_OverridesAllFields(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND", "openai")
	t.Setenv("EMBEDDING_BASE_URL", "http://gpu-host:13305")
	t.Setenv("EMBEDDING_API_KEY", "sk-test")
	t.Setenv("EMBEDDING_MODEL", "text-embedding-3-small")
	t.Setenv("EMBEDDING_STRICT", "true")

	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatalf("ConfigFromEnv: %v", err)
	}
	want := Config{
		Backend: BackendOpenAI,
		BaseURL: "http://gpu-host:13305",
		APIKey:  "sk-test",
		Model:   "text-embedding-3-small",
		Strict:  true,
	}
	if cfg != want {
		t.Errorf("got %+v, want %+v", cfg, want)
	}
}

func TestConfigFromEnv_PartialOverride_KeepsDefaults(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND", "")
	t.Setenv("EMBEDDING_BASE_URL", "http://other:11434")
	t.Setenv("EMBEDDING_API_KEY", "")
	t.Setenv("EMBEDDING_MODEL", "")
	t.Setenv("EMBEDDING_STRICT", "")

	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatalf("ConfigFromEnv: %v", err)
	}
	if cfg.BaseURL != "http://other:11434" {
		t.Errorf("BaseURL: got %q, want override", cfg.BaseURL)
	}
	if cfg.Backend != BackendOllama {
		t.Errorf("Backend: got %q, want default ollama", cfg.Backend)
	}
	if cfg.Model != "nomic-embed-text" {
		t.Errorf("Model: got %q, want default nomic-embed-text", cfg.Model)
	}
}

func TestConfigFromEnv_BackendCaseInsensitive(t *testing.T) {
	cases := []string{"ollama", "Ollama", "OLLAMA", "oLLaMa"}
	for _, v := range cases {
		t.Run(v, func(t *testing.T) {
			t.Setenv("EMBEDDING_BACKEND", v)
			cfg, err := ConfigFromEnv()
			if err != nil {
				t.Fatalf("ConfigFromEnv: %v", err)
			}
			if cfg.Backend != BackendOllama {
				t.Errorf("Backend: got %q, want ollama", cfg.Backend)
			}
		})
	}
}

func TestConfigFromEnv_InvalidBackend(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND", "cohere")
	_, err := ConfigFromEnv()
	if err == nil {
		t.Fatal("expected error for unknown backend")
	}
	if !strings.Contains(err.Error(), "EMBEDDING_BACKEND") {
		t.Errorf("error should name the env var: %v", err)
	}
}

func TestConfigFromEnv_StrictParsing(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"true", true},
		{"false", false},
		{"1", true},
		{"0", false},
		{"TRUE", true},
		{"f", false},
	}
	for _, tc := range cases {
		t.Run(tc.val, func(t *testing.T) {
			t.Setenv("EMBEDDING_STRICT", tc.val)
			cfg, err := ConfigFromEnv()
			if err != nil {
				t.Fatalf("ConfigFromEnv: %v", err)
			}
			if cfg.Strict != tc.want {
				t.Errorf("Strict: got %v, want %v", cfg.Strict, tc.want)
			}
		})
	}
}

func TestConfigFromEnv_InvalidStrict(t *testing.T) {
	t.Setenv("EMBEDDING_STRICT", "banana")
	_, err := ConfigFromEnv()
	if err == nil {
		t.Fatal("expected error for unparseable bool")
	}
	if !strings.Contains(err.Error(), "EMBEDDING_STRICT") {
		t.Errorf("error should name the env var: %v", err)
	}
}

func TestConfigFromEnvPrefix_UsesGivenPrefix(t *testing.T) {
	// Set both EMBEDDING_* and MEMSTORE_EMBED_* to confirm the prefixed
	// call only reads MEMSTORE_EMBED_* and ignores the canonical set.
	t.Setenv("EMBEDDING_BASE_URL", "http://default:11434")
	t.Setenv("MEMSTORE_EMBED_BASE_URL", "http://app-specific:11434")

	cfg, err := ConfigFromEnvPrefix("MEMSTORE_EMBED")
	if err != nil {
		t.Fatalf("ConfigFromEnvPrefix: %v", err)
	}
	if cfg.BaseURL != "http://app-specific:11434" {
		t.Errorf("BaseURL: got %q, want app-specific override", cfg.BaseURL)
	}
}

func TestConfigFromEnvPrefix_FallsBackToCanonical(t *testing.T) {
	// No prefixed value set; ConfigFromEnvPrefix should pick up the canonical
	// EMBEDDING_BACKEND so a single shared env can drive every app.
	t.Setenv("EMBEDDING_BACKEND", "openai")
	t.Setenv("MYAPP_EMBED_BACKEND", "")

	cfg, err := ConfigFromEnvPrefix("MYAPP_EMBED")
	if err != nil {
		t.Fatalf("ConfigFromEnvPrefix: %v", err)
	}
	if cfg.Backend != BackendOpenAI {
		t.Errorf("Backend: got %q, want openai (canonical fallback should apply)", cfg.Backend)
	}
}

func TestConfigFromEnvPrefix_PrefixWinsOverCanonical(t *testing.T) {
	// Both set; the prefixed value must override the canonical.
	t.Setenv("EMBEDDING_BACKEND", "ollama")
	t.Setenv("MYAPP_EMBED_BACKEND", "openai")

	cfg, err := ConfigFromEnvPrefix("MYAPP_EMBED")
	if err != nil {
		t.Fatalf("ConfigFromEnvPrefix: %v", err)
	}
	if cfg.Backend != BackendOpenAI {
		t.Errorf("Backend: got %q, want openai (prefix should override canonical)", cfg.Backend)
	}
}

func TestConfigFromEnvPrefix_PerFieldCascade(t *testing.T) {
	// Mix sources: BASE_URL only set in canonical, MODEL only set in prefix.
	// Result should combine both — proving the cascade is per-field, not
	// all-or-nothing.
	t.Setenv("EMBEDDING_BASE_URL", "http://canonical-host:11434")
	t.Setenv("EMBEDDING_MODEL", "")
	t.Setenv("MYAPP_EMBED_BASE_URL", "")
	t.Setenv("MYAPP_EMBED_MODEL", "custom-model")

	cfg, err := ConfigFromEnvPrefix("MYAPP_EMBED")
	if err != nil {
		t.Fatalf("ConfigFromEnvPrefix: %v", err)
	}
	if cfg.BaseURL != "http://canonical-host:11434" {
		t.Errorf("BaseURL: got %q, want canonical fallback", cfg.BaseURL)
	}
	if cfg.Model != "custom-model" {
		t.Errorf("Model: got %q, want prefix override", cfg.Model)
	}
}

func TestConfigFromEnvPrefix_BadPrefixedBackend_ErrorMessageNamesPrefixedKey(t *testing.T) {
	// When a parse error involves the prefixed value, the error should
	// name the prefixed key so the user knows which to fix.
	t.Setenv("EMBEDDING_BACKEND", "ollama")
	t.Setenv("MYAPP_EMBED_BACKEND", "cohere")

	_, err := ConfigFromEnvPrefix("MYAPP_EMBED")
	if err == nil {
		t.Fatal("expected error for bad prefixed backend")
	}
	if !strings.Contains(err.Error(), "MYAPP_EMBED_BACKEND") {
		t.Errorf("error should name MYAPP_EMBED_BACKEND, got: %v", err)
	}
}

func TestConfigFromEnvPrefix_BadCanonicalBackend_ErrorMessageNamesCanonicalKey(t *testing.T) {
	// When the cascade falls through to a bad canonical value, the error
	// should name EMBEDDING_BACKEND so the user fixes the right one.
	t.Setenv("EMBEDDING_BACKEND", "cohere")
	t.Setenv("MYAPP_EMBED_BACKEND", "")

	_, err := ConfigFromEnvPrefix("MYAPP_EMBED")
	if err == nil {
		t.Fatal("expected error for bad canonical backend")
	}
	if !strings.Contains(err.Error(), "EMBEDDING_BACKEND") {
		t.Errorf("error should name EMBEDDING_BACKEND, got: %v", err)
	}
}

func TestConfigFromEnv_ResultValidates(t *testing.T) {
	// Ensure a fully env-overridden Config produces a working Embedder via
	// New — closes the loop on the canonical "set env, just New it" flow.
	t.Setenv("EMBEDDING_BACKEND", "ollama")
	t.Setenv("EMBEDDING_BASE_URL", "http://localhost:11434")
	t.Setenv("EMBEDDING_MODEL", "nomic-embed-text")

	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatalf("ConfigFromEnv: %v", err)
	}
	if _, err := New(cfg); err != nil {
		t.Errorf("New(env-derived Config): %v", err)
	}
}

func TestDefaultEnvPrefix_Constant(t *testing.T) {
	// Pin the constant value — callers may rely on this for diagnostics
	// or for constructing related env keys.
	if DefaultEnvPrefix != "EMBEDDING" {
		t.Errorf("DefaultEnvPrefix: got %q, want EMBEDDING", DefaultEnvPrefix)
	}
}
