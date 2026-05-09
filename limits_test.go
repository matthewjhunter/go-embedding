package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestLookupLimits_PreRegisteredModels(t *testing.T) {
	cases := []struct {
		model        string
		wantMaxBytes int
	}{
		{"nomic-embed-text", 6000},
		{"nomic-embed-text-v2", 6000},
		{"embeddinggemma", 6000},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			got := LookupLimits(tc.model)
			if got.MaxBytes != tc.wantMaxBytes {
				t.Errorf("MaxBytes for %s: got %d, want %d", tc.model, got.MaxBytes, tc.wantMaxBytes)
			}
		})
	}
}

func TestLookupLimits_Unknown(t *testing.T) {
	got := LookupLimits("entirely-fictional-model")
	if got != (Limits{}) {
		t.Errorf("unknown model: got %v, want zero Limits", got)
	}
}

func TestLookupLimits_TagSuffixFallsBackToBase(t *testing.T) {
	// Tagged variants ("nomic-embed-text:latest", ":q4_0", ":v1.5", ...) share
	// the base model's architectural limits. The lookup should fall back to
	// the bare name when no exact match is registered.
	cases := []string{
		"nomic-embed-text:latest",
		"nomic-embed-text:q4_0",
		"nomic-embed-text:v1.5",
		"nomic-embed-text:fp16",
	}
	for _, model := range cases {
		t.Run(model, func(t *testing.T) {
			got := LookupLimits(model)
			if got.MaxBytes != 6000 {
				t.Errorf("MaxBytes for %s: got %d, want 6000 (via base-name fallback)", model, got.MaxBytes)
			}
		})
	}
}

func TestLookupLimits_TagSuffixOnUnknownBase(t *testing.T) {
	// Tag-stripping should not invent limits for an unregistered base model.
	got := LookupLimits("entirely-fictional-model:latest")
	if got != (Limits{}) {
		t.Errorf("unknown base with tag: got %v, want zero Limits", got)
	}
}

func TestLookupLimits_ExactMatchWinsOverTagFallback(t *testing.T) {
	// If a tagged variant is explicitly registered with different limits, the
	// exact match must win — base-name fallback only fires on miss.
	const tagged = "nomic-embed-text:tiny"
	t.Cleanup(func() { unregisterLimits(tagged) })
	RegisterLimits(tagged, Limits{MaxBytes: 1000})
	if got := LookupLimits(tagged); got.MaxBytes != 1000 {
		t.Errorf("explicit tagged registration: got %d, want 1000", got.MaxBytes)
	}
	// Bare name still returns the registry value, not the tagged override.
	if got := LookupLimits("nomic-embed-text"); got.MaxBytes != 6000 {
		t.Errorf("bare name unaffected by tagged registration: got %d, want 6000", got.MaxBytes)
	}
}

func TestRegisterLimits(t *testing.T) {
	const name = "test-only-custom-embedder"
	t.Cleanup(func() { unregisterLimits(name) })
	RegisterLimits(name, Limits{MaxBytes: 1234})
	if got := LookupLimits(name); got.MaxBytes != 1234 {
		t.Errorf("after RegisterLimits: got %d, want 1234", got.MaxBytes)
	}
}

func TestRegisterLimits_ConcurrentReadWrite(t *testing.T) {
	// Without the mutex this fails under -race. With it, every goroutine
	// observes a consistent map.
	const name = "test-only-race-embedder"
	t.Cleanup(func() { unregisterLimits(name) })

	const goroutines = 8
	const iterations = 200
	var wg sync.WaitGroup
	wg.Add(goroutines * 2)

	for i := 0; i < goroutines; i++ {
		go func(seed int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				RegisterLimits(name, Limits{MaxBytes: seed*1000 + j})
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = LookupLimits(name)
			}
		}()
	}
	wg.Wait()
}

func TestEmbed_TruncatesByDefault_Ollama(t *testing.T) {
	var receivedLen int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Input) > 0 {
			receivedLen = len(req.Input[0])
		}
		resp := ollamaEmbedResponse{Embeddings: [][]float32{{0.1, 0.2}}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, _ := New(Config{Backend: BackendOllama, BaseURL: server.URL, Model: "nomic-embed-text"})
	overlength := strings.Repeat("a", 9000)
	if _, err := e.Embed(context.Background(), []string{overlength}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if receivedLen != 6000 {
		t.Errorf("server saw %d bytes, want 6000 (truncated)", receivedLen)
	}
}

func TestEmbed_StrictReturnsError_Ollama(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be called when Strict rejects input")
	}))
	defer server.Close()

	e, _ := New(Config{
		Backend: BackendOllama,
		BaseURL: server.URL,
		Model:   "nomic-embed-text",
		Strict:  true,
	})
	overlength := strings.Repeat("a", 9000)
	_, err := e.Embed(context.Background(), []string{overlength})
	if err == nil {
		t.Fatal("expected error in Strict mode, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("error: got %q, want substring 'exceeds'", err.Error())
	}
}

func TestEmbed_NoLimitsForUnknownModel(t *testing.T) {
	// An unregistered model has no limits; even Strict mode passes oversize text.
	var receivedLen int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Input) > 0 {
			receivedLen = len(req.Input[0])
		}
		resp := ollamaEmbedResponse{Embeddings: [][]float32{{0.1}}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, _ := New(Config{
		Backend: BackendOllama,
		BaseURL: server.URL,
		Model:   "unknown-model",
		Strict:  true,
	})
	input := strings.Repeat("a", 50000)
	if _, err := e.Embed(context.Background(), []string{input}); err != nil {
		t.Fatalf("Embed should succeed for unregistered model even in Strict mode: %v", err)
	}
	if receivedLen != 50000 {
		t.Errorf("server saw %d bytes, want 50000 (no limit applied)", receivedLen)
	}
}

func TestTruncateToBytes_ASCII(t *testing.T) {
	got := truncateToBytes("abcdef", 3)
	if got != "abc" {
		t.Errorf("got %q, want abc", got)
	}
}

func TestTruncateToBytes_UTF8Safety(t *testing.T) {
	// "héllo" — é is 2 bytes (0xC3 0xA9). Truncating at byte 2 must not
	// leave a half-rune at the boundary.
	s := "héllo"
	if len(s) != 6 {
		t.Fatalf("setup wrong: len=%d", len(s))
	}
	got := truncateToBytes(s, 2)
	// We expect "h" (1 byte), backing off the multi-byte boundary.
	if got != "h" {
		t.Errorf("got %q (len=%d), want \"h\" (len=1) — must back off multi-byte boundary", got, len(got))
	}
}

func TestTruncateToBytes_NoTruncationNeeded(t *testing.T) {
	got := truncateToBytes("short", 100)
	if got != "short" {
		t.Errorf("got %q, want short", got)
	}
}
