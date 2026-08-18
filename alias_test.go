package embedding

import (
	"os"
	"strings"
	"testing"
)

// The names a model is actually served under differ per runtime, and a lookup
// miss disables both prefixes and limits silently. These are the real names
// observed across an Ollama + Lemonade fleet.
func TestCanonicalModel_BuiltinAffixes(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"plain name is unchanged", "embeddinggemma", "embeddinggemma"},
		{"ollama tag", "nomic-embed-text:latest", "nomic-embed-text"},
		{"ollama quantisation tag", "nomic-embed-text:q4_0", "nomic-embed-text"},
		{"lemonade registration prefix", "user.embeddinggemma", "embeddinggemma"},
		{"gguf format suffix", "embeddinggemma-GGUF", "embeddinggemma"},
		{"lowercase folding", "EmbeddingGemma", "embeddinggemma"},
		{"prefix, suffix and tag together", "user.EmbeddingGemma-GGUF:latest", "embeddinggemma"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := canonicalModel(tt.in); got != tt.want {
				t.Errorf("canonicalModel(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// A model variant is not a repackaging. Stripping a version or mixture marker
// would silently resolve one model to another's prefixes and budget, which is
// worse than not resolving at all.
func TestCanonicalModel_DoesNotStripVariantMarkers(t *testing.T) {
	for _, in := range []string{"nomic-embed-text-v2-moe", "nomic-embed-text-v1"} {
		if got := canonicalModel(in); got == "nomic-embed-text" {
			t.Errorf("canonicalModel(%q) = %q -- a variant marker was stripped", in, got)
		}
	}
}

func TestRegisterModelAlias_ResolvesPromptsAndLimits(t *testing.T) {
	ResetModelAliases()
	const served = "nomic-embed-text-v1-GGUF"

	if l := LookupLimits(served); l.MaxBytes != 0 {
		t.Fatalf("precondition: %q already resolves", served)
	}
	RegisterModelAlias(served, "nomic-embed-text")

	if got := LookupLimits(served).MaxBytes; got != LookupLimits("nomic-embed-text").MaxBytes {
		t.Errorf("aliased MaxBytes = %d, want the canonical model's", got)
	}
	want := FormatForTask("nomic-embed-text", TaskClustering, "hello")
	if got := FormatForTask(served, TaskClustering, "hello"); got != want {
		t.Errorf("aliased prompt = %q, want %q", got, want)
	}
}

// Aliases resolve prefixes and limits. They must not be used as storage keys:
// vectors from different quantisations are not interchangeable, the same
// reasoning already documented for tag stripping.
func TestRegisterModelAlias_DoesNotRewriteTheEmbedderModelName(t *testing.T) {
	ResetModelAliases()
	const served = "EmbeddingGemma-300M-GGUF"
	RegisterModelAlias(served, "embeddinggemma")

	e, err := New(Config{Backend: BackendOllama, BaseURL: "http://example.invalid", Model: served})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if e.Model() != served {
		t.Errorf("Model() = %q, want the served name %q -- storage keys must not be canonicalised", e.Model(), served)
	}
}

func TestLookupModel_ReportsWhatWillActuallyBeUsed(t *testing.T) {
	ResetModelAliases()

	known, ok := LookupModel("nomic-embed-text:latest")
	if !ok {
		t.Fatal("a registered model reported as unknown")
	}
	if known.Canonical != "nomic-embed-text" {
		t.Errorf("Canonical = %q, want nomic-embed-text", known.Canonical)
	}
	if !known.HasPrompts || known.Limits.MaxBytes == 0 {
		t.Errorf("HasPrompts=%v MaxBytes=%d, want both populated", known.HasPrompts, known.Limits.MaxBytes)
	}

	if _, ok := LookupModel("no-such-model-anywhere"); ok {
		t.Error("an unregistered model reported as known")
	}
}

// The whole point: an unrecognised model can be made a boot error instead of a
// silent quality regression.
func TestStrictModel_RejectsUnknownModelAtConstruction(t *testing.T) {
	ResetModelAliases()
	_, err := New(Config{
		Backend: BackendOllama, BaseURL: "http://example.invalid",
		Model: "totally-unknown-model", StrictModel: true,
	})
	if err == nil {
		t.Fatal("StrictModel accepted an unknown model")
	}
	if !strings.Contains(err.Error(), "totally-unknown-model") {
		t.Errorf("error does not name the model: %v", err)
	}
}

func TestStrictModel_AcceptsAnAliasedModel(t *testing.T) {
	ResetModelAliases()
	RegisterModelAlias("weird-vendor-name", "embeddinggemma")
	if _, err := New(Config{
		Backend: BackendOllama, BaseURL: "http://example.invalid",
		Model: "weird-vendor-name", StrictModel: true,
	}); err != nil {
		t.Errorf("StrictModel rejected an aliased model: %v", err)
	}
}

// Operators name models arbitrarily, so aliases have to be settable without a
// code change.
func TestConfigFromEnv_ModelAliases(t *testing.T) {
	ResetModelAliases()
	t.Setenv("EMBEDDING_MODEL_ALIAS", "nomic-embed-text-v1-GGUF=nomic-embed-text, EmbeddingGemma-300M-GGUF = embeddinggemma")
	defer os.Unsetenv("EMBEDDING_MODEL_ALIAS")

	if _, err := ConfigFromEnv(); err != nil {
		t.Fatalf("ConfigFromEnv: %v", err)
	}
	if got := LookupLimits("nomic-embed-text-v1-GGUF").MaxBytes; got == 0 {
		t.Error("first alias did not take effect")
	}
	if got := LookupLimits("EmbeddingGemma-300M-GGUF").MaxBytes; got == 0 {
		t.Error("second alias did not take effect (whitespace around = or , not tolerated?)")
	}
}

func TestConfigFromEnv_MalformedAliasIsAnError(t *testing.T) {
	ResetModelAliases()
	t.Setenv("EMBEDDING_MODEL_ALIAS", "no-equals-sign-here")
	defer os.Unsetenv("EMBEDDING_MODEL_ALIAS")

	if _, err := ConfigFromEnv(); err == nil {
		t.Error("a malformed alias was accepted silently")
	}
}

func TestConfigFromEnv_StrictModel(t *testing.T) {
	t.Setenv("EMBEDDING_STRICT_MODEL", "true")
	defer os.Unsetenv("EMBEDDING_STRICT_MODEL")
	cfg, err := ConfigFromEnv()
	if err != nil {
		t.Fatalf("ConfigFromEnv: %v", err)
	}
	if !cfg.StrictModel {
		t.Error("EMBEDDING_STRICT_MODEL=true did not set StrictModel")
	}
}
