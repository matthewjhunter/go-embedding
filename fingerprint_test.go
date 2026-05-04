package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFingerprint_BeforeEmbed_DimZero(t *testing.T) {
	e, err := New(Config{Backend: BackendOllama, BaseURL: "http://x", Model: "nomic-embed-text"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	fp := e.Fingerprint()
	if fp.Model != "nomic-embed-text" {
		t.Errorf("Model: got %s, want nomic-embed-text", fp.Model)
	}
	if fp.Dim != 0 {
		t.Errorf("Dim before any Embed: got %d, want 0", fp.Dim)
	}
}

func TestFingerprint_AfterEmbed_DimPopulated_Ollama(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ollamaEmbedResponse{Embeddings: [][]float32{make([]float32, 768)}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, _ := New(Config{Backend: BackendOllama, BaseURL: server.URL, Model: "nomic-embed-text"})
	if _, err := e.Embed(context.Background(), []string{"hi"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	fp := e.Fingerprint()
	if fp.Dim != 768 {
		t.Errorf("Dim after Embed: got %d, want 768", fp.Dim)
	}
}

func TestFingerprint_AfterEmbed_DimPopulated_OpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openAIEmbedResponse{Data: []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{{Embedding: make([]float32, 1536), Index: 0}}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, _ := New(Config{Backend: BackendOpenAI, BaseURL: server.URL, Model: "text-embedding-3-small"})
	if _, err := e.Embed(context.Background(), []string{"hi"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	fp := e.Fingerprint()
	if fp.Dim != 1536 {
		t.Errorf("Dim after Embed: got %d, want 1536", fp.Dim)
	}
}

func TestCheckFingerprint_Match(t *testing.T) {
	a := Fingerprint{Model: "nomic", Dim: 768}
	b := Fingerprint{Model: "nomic", Dim: 768}
	if err := CheckFingerprint(a, b); err != nil {
		t.Errorf("matching fingerprints: got error %v, want nil", err)
	}
}

func TestCheckFingerprint_ModelMismatch(t *testing.T) {
	stored := Fingerprint{Model: "nomic-embed-text", Dim: 768}
	current := Fingerprint{Model: "embeddinggemma", Dim: 768}

	err := CheckFingerprint(stored, current)
	if err == nil {
		t.Fatal("expected mismatch error, got nil")
	}
	var mm *MismatchError
	if !errors.As(err, &mm) {
		t.Fatalf("expected *MismatchError, got %T", err)
	}
	if mm.Stored != stored || mm.Current != current {
		t.Errorf("MismatchError: got Stored=%v Current=%v", mm.Stored, mm.Current)
	}
	msg := err.Error()
	if !strings.Contains(msg, "nomic-embed-text") || !strings.Contains(msg, "embeddinggemma") {
		t.Errorf("error message missing model names: %s", msg)
	}
}

func TestCheckFingerprint_DimMismatch(t *testing.T) {
	// Same model name, different dim — the v1/v2 silent-corruption case.
	stored := Fingerprint{Model: "nomic-embed-text", Dim: 768}
	current := Fingerprint{Model: "nomic-embed-text", Dim: 1024}

	err := CheckFingerprint(stored, current)
	if err == nil {
		t.Fatal("expected mismatch error, got nil")
	}
	var mm *MismatchError
	if !errors.As(err, &mm) {
		t.Fatalf("expected *MismatchError, got %T", err)
	}
	msg := err.Error()
	if !strings.Contains(msg, "768") || !strings.Contains(msg, "1024") {
		t.Errorf("error message missing dims: %s", msg)
	}
}

func TestFingerprint_DeprecatedConstructorsImplementInterface(t *testing.T) {
	// Adding Fingerprint to Embedder is a breaking change for any external
	// implementers. Verify our shipped concrete types still satisfy it.
	var _ Embedder = NewOllamaEmbedder("http://x", "m")
	var _ Embedder = NewOpenAIEmbedder("http://x", "", "m")
}
