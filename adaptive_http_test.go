package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// contextLengthServer rejects any embed request whose first input exceeds
// fitsAt bytes with an HTTP 400 carrying a context-length message; smaller
// inputs get a stub embedding. It records the smallest input length it
// accepted and how many requests it saw.
func contextLengthServer(t *testing.T, fitsAt int) (*httptest.Server, *int) {
	t.Helper()
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var req ollamaEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Input) > 0 && len(req.Input[0]) > fitsAt {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"input exceeds the maximum context length"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{Embeddings: [][]float32{{0.1, 0.2}}})
	}))
	t.Cleanup(srv.Close)
	return srv, &requests
}

func TestEmbed_AdaptiveShrinkOnReject_Ollama(t *testing.T) {
	// fitsAt is below nomic's 6000-byte budget, so the statically truncated
	// input still 400s and adaptive shrink must kick in.
	srv, requests := contextLengthServer(t, 3000)

	e, _ := New(Config{Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text"})
	out, err := e.Embed(context.Background(), []string{strings.Repeat("a", 9000)})
	if err != nil {
		t.Fatalf("Embed should succeed after adaptive shrink: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("want 1 embedding, got %d", len(out))
	}
	if *requests < 2 {
		t.Errorf("expected at least one reject + one accepted request, got %d", *requests)
	}
}

func TestEmbed_PermanentWhenNeverFits_Ollama(t *testing.T) {
	srv, _ := contextLengthServer(t, 0) // rejects everything as too long

	e, _ := New(Config{Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text"})
	_, err := e.Embed(context.Background(), []string{strings.Repeat("a", 9000)})
	if err == nil {
		t.Fatal("expected a permanent error when input never fits")
	}
	if IsRetryable(err) {
		t.Error("expected non-retryable error after exhausting adaptive shrink")
	}
}

func TestEmbed_AuthErrorIsPermanentNotShrunk_Ollama(t *testing.T) {
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"invalid api key"}`))
	}))
	defer srv.Close()

	e, _ := New(Config{Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text"})
	_, err := e.Embed(context.Background(), []string{"short query"})
	if err == nil {
		t.Fatal("expected an error")
	}
	if IsRetryable(err) {
		t.Error("4xx auth error should be permanent")
	}
	if requests != 1 {
		t.Errorf("auth error should not trigger shrink retries; got %d requests", requests)
	}
}

func TestEmbedWithRetry_StopsOnPermanent(t *testing.T) {
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("nope"))
	}))
	defer srv.Close()

	e, _ := New(Config{Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text"})
	_, err := EmbedWithRetry(context.Background(), e, []string{"q"})
	if err == nil {
		t.Fatal("expected error")
	}
	// One attempt only: a permanent error must not be retried.
	if requests != 1 {
		t.Errorf("EmbedWithRetry made %d requests, want 1 (no retry on permanent)", requests)
	}
}
