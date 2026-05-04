package embedding

import "testing"

func TestDefaultConfig_ValidatesAndReturnsEmbedder(t *testing.T) {
	cfg := DefaultConfig()
	if _, err := New(cfg); err != nil {
		t.Fatalf("DefaultConfig must produce a valid Config, got error: %v", err)
	}
	if cfg.Model == "" {
		t.Error("DefaultConfig.Model is empty")
	}
}

func TestOllamaLocalNomic(t *testing.T) {
	cfg := OllamaLocalNomic()
	if cfg.Backend != BackendOllama {
		t.Errorf("Backend: got %q, want %q", cfg.Backend, BackendOllama)
	}
	if cfg.Model != "nomic-embed-text" {
		t.Errorf("Model: got %q, want nomic-embed-text", cfg.Model)
	}
	if cfg.BaseURL != "http://localhost:11434" {
		t.Errorf("BaseURL: got %q, want http://localhost:11434", cfg.BaseURL)
	}
	if _, err := New(cfg); err != nil {
		t.Errorf("preset must validate: %v", err)
	}
}

func TestLemonadeNomic(t *testing.T) {
	cfg := LemonadeNomic()
	// Lemonade speaks the OpenAI protocol, not Ollama's native API.
	if cfg.Backend != BackendOpenAI {
		t.Errorf("Backend: got %q, want %q (Lemonade is OpenAI-compatible)", cfg.Backend, BackendOpenAI)
	}
	if cfg.Model != "nomic-embed-text" {
		t.Errorf("Model: got %q, want nomic-embed-text", cfg.Model)
	}
	if cfg.BaseURL != "http://localhost:13305" {
		t.Errorf("BaseURL: got %q, want http://localhost:13305", cfg.BaseURL)
	}
	if _, err := New(cfg); err != nil {
		t.Errorf("preset must validate: %v", err)
	}
}

func TestDefaultConfig_MatchesOllamaLocalNomic(t *testing.T) {
	// DefaultConfig is currently an alias for OllamaLocalNomic. If this
	// changes, update the godoc on DefaultConfig.
	if DefaultConfig() != OllamaLocalNomic() {
		t.Error("DefaultConfig diverged from OllamaLocalNomic; update godoc")
	}
}
