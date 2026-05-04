package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAIEmbedder_Embed_HappyPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("path: got %s, want /v1/embeddings", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("method: got %s, want POST", r.Method)
		}
		var req openAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Model != "text-embedding-3-small" {
			t.Errorf("model: got %s, want text-embedding-3-small", req.Model)
		}
		resp := openAIEmbedResponse{Data: []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{
			{Embedding: []float32{0.1, 0.2}, Index: 0},
			{Embedding: []float32{0.3, 0.4}, Index: 1},
		}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewOpenAIEmbedder(server.URL, "key", "text-embedding-3-small")
	results, err := e.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results): got %d, want 2", len(results))
	}
}

func TestOpenAIEmbedder_ReordersByIndex(t *testing.T) {
	// OpenAI spec says data may be returned in any order; the embedder must
	// reorder by Index so the caller's slot positions are correct.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openAIEmbedResponse{Data: []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{
			{Embedding: []float32{2.0}, Index: 2},
			{Embedding: []float32{0.0}, Index: 0},
			{Embedding: []float32{1.0}, Index: 1},
		}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewOpenAIEmbedder(server.URL, "", "m")
	results, err := e.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	for i, want := range []float32{0.0, 1.0, 2.0} {
		if len(results[i]) != 1 || results[i][0] != want {
			t.Errorf("results[%d]: got %v, want [%f]", i, results[i], want)
		}
	}
}

func TestOpenAIEmbedder_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
	}))
	defer server.Close()

	e := NewOpenAIEmbedder(server.URL, "", "missing")
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error on HTTP 404")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("error should mention status code: %v", err)
	}
}

func TestOpenAIEmbedder_EmptyResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(openAIEmbedResponse{})
	}))
	defer server.Close()

	e := NewOpenAIEmbedder(server.URL, "", "m")
	_, err := e.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error on empty response")
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Errorf("error should mention 'empty': %v", err)
	}
}

func TestOpenAIEmbedder_AuthorizationHeader_Omitted_WhenNoKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization should be empty when APIKey is unset, got %q", got)
		}
		_ = json.NewEncoder(w).Encode(openAIEmbedResponse{Data: []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{{Embedding: []float32{1}, Index: 0}}})
	}))
	defer server.Close()

	e := NewOpenAIEmbedder(server.URL, "", "m")
	if _, err := e.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
}

func TestOpenAIEmbedder_BaseURLTrailingSlashTrimmed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Without trim, the URL becomes "{base}//v1/embeddings" which most
		// servers redirect; httptest serves both, so we confirm via the
		// path that no double-slash leaked into routing.
		if strings.HasPrefix(r.URL.Path, "//") {
			t.Errorf("path has leading double slash: %q", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(openAIEmbedResponse{Data: []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{{Embedding: []float32{1}, Index: 0}}})
	}))
	defer server.Close()

	e := NewOpenAIEmbedder(server.URL+"/", "", "m")
	if _, err := e.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
}

func TestOpenAIEmbedder_Model(t *testing.T) {
	e := NewOpenAIEmbedder("http://x", "", "text-embedding-3-large")
	if got := e.Model(); got != "text-embedding-3-large" {
		t.Errorf("Model: got %s", got)
	}
}
