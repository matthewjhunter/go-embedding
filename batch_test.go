package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// fakeEmbedder is a programmable Embedder used to drive BatchEmbed scenarios
// without an HTTP server.
type fakeEmbedder struct {
	model string
	// embed is called for each Embed invocation. It returns the vectors
	// (which may be shorter than texts to simulate partial responses) and
	// an optional error.
	embed func(texts []string) ([][]float32, error)
	calls atomic.Int32
}

func (f *fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.calls.Add(1)
	return f.embed(texts)
}

func (f *fakeEmbedder) Model() string            { return f.model }
func (f *fakeEmbedder) Fingerprint() Fingerprint { return Fingerprint{Model: f.model} }

func TestBatchEmbed_HappyPath(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			out := make([][]float32, len(texts))
			for i := range texts {
				out[i] = []float32{float32(i)}
			}
			return out, nil
		},
	}
	texts := []string{"a", "b", "c", "d", "e", "f", "g"}
	got, err := BatchEmbed(context.Background(), f, texts, 3, nil)
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if len(got) != len(texts) {
		t.Fatalf("len(got)=%d, want %d", len(got), len(texts))
	}
	for i, v := range got {
		if v == nil {
			t.Errorf("vector %d is nil", i)
		}
	}
	// 7 texts in batches of 3 = 3 batch calls (3+3+1)
	if f.calls.Load() != 3 {
		t.Errorf("expected 3 batch calls, got %d", f.calls.Load())
	}
}

func TestBatchEmbed_PartialBatchTriggersFallback(t *testing.T) {
	var batchCalls, singleCalls atomic.Int32
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			if len(texts) > 1 {
				batchCalls.Add(1)
				// Return one fewer vector than requested — partial response.
				return [][]float32{{1.0}}, nil
			}
			singleCalls.Add(1)
			return [][]float32{{2.0}}, nil
		},
	}
	got, err := BatchEmbed(context.Background(), f, []string{"a", "b", "c"}, 3, nil)
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if batchCalls.Load() != 1 {
		t.Errorf("expected 1 batch call, got %d", batchCalls.Load())
	}
	if singleCalls.Load() != 3 {
		t.Errorf("expected 3 single-call fallbacks, got %d", singleCalls.Load())
	}
	for i, v := range got {
		if v == nil || v[0] != 2.0 {
			t.Errorf("vector %d: got %v, want [2.0] (from fallback)", i, v)
		}
	}
}

func TestBatchEmbed_BatchErrorTriggersFallback(t *testing.T) {
	var singleCalls atomic.Int32
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			if len(texts) > 1 {
				return nil, &fakeError{"boom"}
			}
			singleCalls.Add(1)
			return [][]float32{{0.5}}, nil
		},
	}
	got, err := BatchEmbed(context.Background(), f, []string{"a", "b"}, 2, nil)
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if singleCalls.Load() != 2 {
		t.Errorf("expected 2 single-call fallbacks, got %d", singleCalls.Load())
	}
	for i, v := range got {
		if v == nil {
			t.Errorf("vector %d is nil", i)
		}
	}
}

func TestBatchEmbed_AllFailReturnsError(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			return nil, &fakeError{"always fails"}
		},
	}
	_, err := BatchEmbed(context.Background(), f, []string{"a", "b"}, 2, nil)
	if err == nil {
		t.Fatal("expected error when all embeddings fail")
	}
}

func TestBatchEmbed_ProgressInvokedPerBatch(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			out := make([][]float32, len(texts))
			for i := range texts {
				out[i] = []float32{1}
			}
			return out, nil
		},
	}

	type progressCall struct{ done, total int }
	var calls []progressCall
	progress := func(done, total int) {
		calls = append(calls, progressCall{done, total})
	}

	texts := []string{"a", "b", "c", "d", "e"}
	if _, err := BatchEmbed(context.Background(), f, texts, 2, progress); err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	// 5 texts, batchSize 2 → batches end at 2, 4, 5
	want := []progressCall{{2, 5}, {4, 5}, {5, 5}}
	if len(calls) != len(want) {
		t.Fatalf("progress calls: got %d, want %d (%v)", len(calls), len(want), calls)
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Errorf("call %d: got %v, want %v", i, calls[i], want[i])
		}
	}
}

func TestBatchEmbed_ZeroBatchSizeUsesDefault(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			out := make([][]float32, len(texts))
			for i := range texts {
				out[i] = []float32{1}
			}
			return out, nil
		},
	}
	// 30 texts with batchSize 0 → uses default (25), so 2 batches
	texts := make([]string, 30)
	for i := range texts {
		texts[i] = "x"
	}
	if _, err := BatchEmbed(context.Background(), f, texts, 0, nil); err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if f.calls.Load() != 2 {
		t.Errorf("expected 2 batch calls with default size, got %d", f.calls.Load())
	}
}

func TestBatchEmbed_EmptyTexts(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			t.Error("Embed should not be called for empty input")
			return nil, nil
		},
	}
	got, err := BatchEmbed(context.Background(), f, nil, 10, nil)
	if err != nil {
		t.Fatalf("BatchEmbed on empty: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d vectors, want 0", len(got))
	}
}

// Smoke test that BatchEmbed works against a real httptest backend, not just
// the fake.
func TestBatchEmbed_AgainstHTTPServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		out := make([][]float32, len(req.Input))
		for i := range req.Input {
			out[i] = []float32{float32(len(req.Input[i]))}
		}
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{Embeddings: out})
	}))
	defer server.Close()

	e, _ := New(Config{Backend: BackendOllama, BaseURL: server.URL, Model: "test"})
	texts := []string{"a", "bb", "ccc", "dddd"}
	got, err := BatchEmbed(context.Background(), e, texts, 2, nil)
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("got %d vectors, want 4", len(got))
	}
	wantLens := []float32{1, 2, 3, 4}
	for i, v := range got {
		if len(v) != 1 || v[0] != wantLens[i] {
			t.Errorf("vector %d: got %v, want [%f]", i, v, wantLens[i])
		}
	}
}

type fakeError struct{ msg string }

func (e *fakeError) Error() string { return e.msg }
