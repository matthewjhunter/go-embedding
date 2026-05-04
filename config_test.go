package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNew_OllamaBackend(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		resp := ollamaEmbedResponse{Embeddings: [][]float32{{0.1, 0.2}}}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, err := New(Config{
		Backend: BackendOllama,
		BaseURL: server.URL,
		Model:   "nomic-embed-text",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, ok := e.(*OllamaEmbedder); !ok {
		t.Errorf("expected *OllamaEmbedder, got %T", e)
	}
	if got := e.Model(); got != "nomic-embed-text" {
		t.Errorf("Model: got %s, want nomic-embed-text", got)
	}

	vecs, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 2 {
		t.Errorf("unexpected embed shape: %v", vecs)
	}
}

func TestNew_OpenAIBackend(t *testing.T) {
	gotAuth := ""
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		resp := openAIEmbedResponse{Data: []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{{Embedding: []float32{0.5, 0.6}, Index: 0}}}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, err := New(Config{
		Backend: BackendOpenAI,
		BaseURL: server.URL,
		APIKey:  "secret-key",
		Model:   "text-embedding-3-small",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, ok := e.(*OpenAIEmbedder); !ok {
		t.Errorf("expected *OpenAIEmbedder, got %T", e)
	}

	if _, err := e.Embed(context.Background(), []string{"hi"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if gotAuth != "Bearer secret-key" {
		t.Errorf("Authorization: got %q, want %q", gotAuth, "Bearer secret-key")
	}
}

func TestNew_Validation(t *testing.T) {
	cases := []struct {
		name    string
		cfg     Config
		wantErr string
	}{
		{
			name:    "missing backend",
			cfg:     Config{BaseURL: "http://x", Model: "m"},
			wantErr: "Backend",
		},
		{
			name:    "missing base url",
			cfg:     Config{Backend: BackendOllama, Model: "m"},
			wantErr: "BaseURL",
		},
		{
			name:    "missing model",
			cfg:     Config{Backend: BackendOllama, BaseURL: "http://x"},
			wantErr: "Model",
		},
		{
			name:    "unknown backend",
			cfg:     Config{Backend: "bogus", BaseURL: "http://x", Model: "m"},
			wantErr: "unknown backend",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(tc.cfg)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error: got %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestNew_DeprecatedConstructorsStillWork(t *testing.T) {
	// Verify the deprecated constructors continue to function so existing
	// callers (herald v0.2.x) keep working without modification.
	o := NewOllamaEmbedder("http://localhost:11434", "nomic-embed-text")
	if o.Model() != "nomic-embed-text" {
		t.Errorf("Ollama Model: got %s", o.Model())
	}
	p := NewOpenAIEmbedder("http://localhost:11434", "", "nomic-embed-text")
	if p.Model() != "nomic-embed-text" {
		t.Errorf("OpenAI Model: got %s", p.Model())
	}
}

func TestNew_OllamaIgnoresAPIKey(t *testing.T) {
	// APIKey is meaningless for Ollama; New should not error on its presence.
	_, err := New(Config{
		Backend: BackendOllama,
		BaseURL: "http://localhost:11434",
		APIKey:  "ignored",
		Model:   "nomic-embed-text",
	})
	if err != nil {
		t.Errorf("APIKey on Ollama should be tolerated, got: %v", err)
	}
}
