package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

// capturedRerank records what the fake server received, for assertions.
type capturedRerank struct {
	req        jinaRerankRequest
	rawBody    string
	authHeader string
}

// jinaTestServer spins an httptest server speaking the Cohere/Jina rerank
// shape. scoreFn maps a received document to its relevance score; the handler
// returns results in the order received (the backend must sort them itself).
func jinaTestServer(t *testing.T, scoreFn func(doc string) float64) (*httptest.Server, *capturedRerank) {
	t.Helper()
	cap := &capturedRerank{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &cap.req); err != nil {
			t.Errorf("server: bad request body: %v", err)
		}
		cap.rawBody = string(body)
		cap.authHeader = r.Header.Get("Authorization")

		results := make([]map[string]any, 0, len(cap.req.Documents))
		for i, doc := range cap.req.Documents {
			results = append(results, map[string]any{
				"index":           i,
				"relevance_score": scoreFn(doc),
			})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func TestJinaRerankSortsAndPreservesIndex(t *testing.T) {
	t.Parallel()
	// Score = the integer the document contains, so we know the expected order.
	srv, _ := jinaTestServer(t, func(doc string) float64 {
		n, _ := strconv.Atoi(doc)
		return float64(n)
	})
	rr := NewJinaReranker(srv.URL, "", "bge-reranker-v2-m3")

	got, err := rr.Rerank(context.Background(), RerankRequest{
		Query:     "q",
		Documents: []string{"3", "1", "9", "5"}, // scores 3,1,9,5
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	// Sorted desc by score: doc "9"(idx2), "5"(idx3), "3"(idx0), "1"(idx1).
	wantIdx := []int{2, 3, 0, 1}
	if len(got) != len(wantIdx) {
		t.Fatalf("got %d results, want %d", len(got), len(wantIdx))
	}
	for i, w := range wantIdx {
		if got[i].Index != w {
			t.Errorf("result[%d].Index = %d, want %d", i, got[i].Index, w)
		}
	}
	if got[0].Score != 9 {
		t.Errorf("top score = %v, want 9", got[0].Score)
	}
}

func TestJinaRerankEmptyDocumentsNoRoundTrip(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("backend was called for empty documents")
	}))
	t.Cleanup(srv.Close)
	rr := NewJinaReranker(srv.URL, "", "m")

	got, err := rr.Rerank(context.Background(), RerankRequest{Query: "q"})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

func TestJinaRerankTopN(t *testing.T) {
	t.Parallel()
	srv, cap := jinaTestServer(t, func(doc string) float64 {
		n, _ := strconv.Atoi(doc)
		return float64(n)
	})
	rr := NewJinaReranker(srv.URL, "", "m")

	got, err := rr.Rerank(context.Background(), RerankRequest{
		Query:     "q",
		Documents: []string{"3", "1", "9", "5"},
		TopN:      2,
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
	if got[0].Index != 2 || got[1].Index != 3 {
		t.Errorf("top-2 indices = %d,%d, want 2,3", got[0].Index, got[1].Index)
	}
	// TopN is applied client-side; it must not leak into the wire request,
	// which would break correctness under fan-out.
	if strings.Contains(cap.rawBody, "top_n") {
		t.Errorf("request body should not contain top_n, got %q", cap.rawBody)
	}
}

func TestJinaRerankInstructionFoldedIntoQuery(t *testing.T) {
	t.Parallel()
	srv, cap := jinaTestServer(t, func(string) float64 { return 1 })
	rr := NewJinaReranker(srv.URL, "", "m")

	_, err := rr.Rerank(context.Background(), RerankRequest{
		Query:       "what is foo",
		Documents:   []string{"a"},
		Instruction: "Rank docs by support relevance",
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if !strings.Contains(cap.req.Query, "Rank docs by support relevance") ||
		!strings.Contains(cap.req.Query, "what is foo") {
		t.Errorf("query %q should contain both instruction and query", cap.req.Query)
	}
}

func TestJinaRerankSendsModelAndAuth(t *testing.T) {
	t.Parallel()
	srv, cap := jinaTestServer(t, func(string) float64 { return 1 })
	rr := NewJinaReranker(srv.URL, "secret-key", "bge-reranker-v2-m3")

	_, err := rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{"a"}})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if cap.req.Model != "bge-reranker-v2-m3" {
		t.Errorf("model = %q, want bge-reranker-v2-m3", cap.req.Model)
	}
	if cap.authHeader != "Bearer secret-key" {
		t.Errorf("auth = %q, want Bearer secret-key", cap.authHeader)
	}
}

func TestJinaRerankServerErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		status      int
		wantUnavail bool
		wantPerm    bool
	}{
		{name: "503 down", status: 503, wantUnavail: true},
		{name: "429 backpressure", status: 429, wantUnavail: true},
		{name: "400 bad request", status: 400, wantPerm: true},
		{name: "404 unknown model", status: 404, wantPerm: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte("error body"))
			}))
			t.Cleanup(srv.Close)
			rr := NewJinaReranker(srv.URL, "", "m")

			_, err := rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{"a"}})
			if err == nil {
				t.Fatal("expected an error")
			}
			if gotUnavail := !IsRerankAvailable(err); gotUnavail != tt.wantUnavail {
				t.Errorf("!IsRerankAvailable = %v, want %v (err: %v)", gotUnavail, tt.wantUnavail, err)
			}
			var pe *PermanentError
			if errors.As(err, &pe) != tt.wantPerm {
				t.Errorf("PermanentError = %v, want %v (err: %v)", errors.As(err, &pe), tt.wantPerm, err)
			}
		})
	}
}

func TestJinaRerankTransportFailureIsUnavailable(t *testing.T) {
	t.Parallel()
	// Start a server, capture its URL, then close it so the connection refuses.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()
	rr := NewJinaReranker(url, "", "m")

	_, err := rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{"a"}})
	if err == nil {
		t.Fatal("expected an error")
	}
	if IsRerankAvailable(err) {
		t.Errorf("a refused connection should be unavailable, got %v", err)
	}
}

func TestJinaRerankRejectsOutOfRangeIndex(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":99,"relevance_score":0.5}]}`))
	}))
	t.Cleanup(srv.Close)
	rr := NewJinaReranker(srv.URL, "", "m")

	_, err := rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{"a", "b"}})
	if err == nil {
		t.Fatal("expected an error for out-of-range index")
	}
	// A malformed server response is a real error, not a degrade signal.
	if !IsRerankAvailable(err) {
		t.Errorf("out-of-range index should not be unavailable, got %v", err)
	}
}

func TestJinaRerankFansOutAcrossBatches(t *testing.T) {
	t.Parallel()
	// Score each doc by the integer it contains, so a result's Score equals
	// its global document index — letting us verify cross-batch index mapping.
	srv, _ := jinaTestServer(t, func(doc string) float64 {
		n, _ := strconv.Atoi(doc)
		return float64(n)
	})
	rr := NewJinaReranker(srv.URL, "", "m")

	const n = maxRerankBatch*2 + 7 // force three batches
	docs := make([]string, n)
	for i := range docs {
		docs[i] = strconv.Itoa(i)
	}

	got, err := rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: docs})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != n {
		t.Fatalf("got %d results, want %d", len(got), n)
	}
	for i, res := range got {
		if res.Score != float64(res.Index) {
			t.Errorf("result %d: Score %v != Index %d (index mapping broke across batches)", i, res.Score, res.Index)
		}
		if i > 0 && got[i-1].Score < res.Score {
			t.Errorf("results not sorted descending at %d", i)
		}
	}
	if got[0].Index != n-1 {
		t.Errorf("top result Index = %d, want %d", got[0].Index, n-1)
	}
}

func TestNewRerankerJinaConstructs(t *testing.T) {
	t.Parallel()
	rr, err := NewReranker(RerankConfig{
		Backend: RerankBackendJina,
		BaseURL: "http://localhost:9999",
		Model:   "bge-reranker-v2-m3",
	})
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if rr.Model() != "bge-reranker-v2-m3" {
		t.Errorf("Model = %q, want bge-reranker-v2-m3", rr.Model())
	}
}

func TestNewRerankerNormalizeScoresWrapsBackend(t *testing.T) {
	t.Parallel()
	// Server echoes each document's text as its raw logit score.
	srv, _ := jinaTestServer(t, func(doc string) float64 {
		f, _ := strconv.ParseFloat(doc, 64)
		return f
	})
	rr, err := NewReranker(RerankConfig{
		Backend:         RerankBackendJina,
		BaseURL:         srv.URL,
		Model:           "bge-reranker-v2-m3",
		NormalizeScores: true,
	})
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	got, err := rr.Rerank(context.Background(), RerankRequest{
		Query:     "q",
		Documents: []string{"1.7297048568725586", "-11.033184051513672"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
	for i, r := range got {
		if r.Score <= 0 || r.Score >= 1 {
			t.Errorf("result[%d].Score = %v, want normalized into (0,1)", i, r.Score)
		}
	}
}

func TestNewRerankerRawScoresByDefault(t *testing.T) {
	t.Parallel()
	srv, _ := jinaTestServer(t, func(doc string) float64 {
		f, _ := strconv.ParseFloat(doc, 64)
		return f
	})
	rr, err := NewReranker(RerankConfig{
		Backend: RerankBackendJina,
		BaseURL: srv.URL,
		Model:   "bge-reranker-v2-m3",
	})
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	got, err := rr.Rerank(context.Background(), RerankRequest{
		Query:     "q",
		Documents: []string{"1.7297048568725586"},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if got[0].Score != 1.7297048568725586 {
		t.Errorf("Score = %v, want raw passthrough (no normalization by default)", got[0].Score)
	}
}

func TestJinaRerankTruncatesOversizeDocuments(t *testing.T) {
	// A small registered budget makes truncation observable without huge strings.
	RegisterLimits("test-rerank-trunc", Limits{MaxBytes: 10})
	t.Cleanup(func() { unregisterLimits("test-rerank-trunc") })

	srv, cap := jinaTestServer(t, func(string) float64 { return 1 })
	rr := NewJinaReranker(srv.URL, "", "test-rerank-trunc")

	long := strings.Repeat("x", 50)
	got, err := rr.Rerank(context.Background(), RerankRequest{
		Query:     "q",
		Documents: []string{"short", long},
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	// The server must have received the long doc truncated to the budget, and the
	// short doc untouched. Order/count are preserved so indices still map back.
	if len(cap.req.Documents) != 2 {
		t.Fatalf("server received %d docs, want 2", len(cap.req.Documents))
	}
	if cap.req.Documents[0] != "short" {
		t.Errorf("doc 0 = %q, want unchanged %q", cap.req.Documents[0], "short")
	}
	if l := len(cap.req.Documents[1]); l != 10 {
		t.Errorf("doc 1 sent with %d bytes, want truncated to 10", l)
	}
	if len(got) != 2 {
		t.Errorf("got %d results, want 2 (indices must survive truncation)", len(got))
	}
}

func TestJinaRerankStrictRejectsOversize(t *testing.T) {
	RegisterLimits("test-rerank-strict", Limits{MaxBytes: 10})
	t.Cleanup(func() { unregisterLimits("test-rerank-strict") })

	srv, _ := jinaTestServer(t, func(string) float64 { return 1 })
	rr, err := NewReranker(RerankConfig{
		Backend: RerankBackendJina,
		BaseURL: srv.URL,
		Model:   "test-rerank-strict",
		Strict:  true,
	})
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	_, err = rr.Rerank(context.Background(), RerankRequest{
		Query:     "q",
		Documents: []string{strings.Repeat("x", 50)},
	})
	if err == nil {
		t.Fatal("expected error for oversize doc in strict mode")
	}
	var perr *PermanentError
	if !errors.As(err, &perr) {
		t.Errorf("expected *PermanentError in strict mode, got %T: %v", err, err)
	}
	// A strict-mode size rejection is a caller bug, not an outage: it must NOT be
	// classified as a transient unavailability (which would silently degrade).
	// IsRerankAvailable is true here, so the consumer surfaces the error instead.
	if !IsRerankAvailable(err) {
		t.Error("strict oversize error should surface (available=true), not degrade as a transient outage")
	}
}

func TestJinaRerankNoTruncationForUnregisteredModel(t *testing.T) {
	srv, cap := jinaTestServer(t, func(string) float64 { return 1 })
	rr := NewJinaReranker(srv.URL, "", "totally-unknown-model")

	long := strings.Repeat("y", 9000)
	if _, err := rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{long}}); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(cap.req.Documents[0]) != 9000 {
		t.Errorf("unregistered model should not truncate; sent %d bytes, want 9000", len(cap.req.Documents[0]))
	}
}
