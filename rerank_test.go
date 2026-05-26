package embedding

import (
	"context"
	"errors"
	"fmt"
	"net"
	"testing"
	"time"
)

// fakeTimeoutErr implements net.Error and reports a timeout, standing in for a
// transport-level deadline a backend might surface unwrapped.
type fakeTimeoutErr struct{}

func (fakeTimeoutErr) Error() string   { return "fake timeout" }
func (fakeTimeoutErr) Timeout() bool   { return true }
func (fakeTimeoutErr) Temporary() bool { return true }

// fakeNetErr implements net.Error but is NOT a timeout (e.g. connection
// refused surfaces this way before any wrapping).
type fakeNetErr struct{}

func (fakeNetErr) Error() string   { return "connection refused" }
func (fakeNetErr) Timeout() bool   { return false }
func (fakeNetErr) Temporary() bool { return false }

var (
	_ net.Error = fakeTimeoutErr{}
	_ net.Error = fakeNetErr{}
)

// stubReranker is a Reranker that returns canned results, for exercising
// wrappers (e.g. normalizingReranker) without a backend.
type stubReranker struct {
	results []RerankResult
	err     error
	model   string
}

func (s stubReranker) Rerank(context.Context, RerankRequest) ([]RerankResult, error) {
	return s.results, s.err
}
func (s stubReranker) Model() string { return s.model }

var _ Reranker = stubReranker{}

func TestNormalizingRerankerAppliesSigmoid(t *testing.T) {
	t.Parallel()
	// Raw logits as llama.cpp --reranking returns them (measured against the
	// bge-reranker-v2-m3 sidecar): unbounded, already sorted descending.
	inner := stubReranker{
		model: "bge-reranker-v2-m3",
		results: []RerankResult{
			{Index: 0, Score: 1.7297048568725586},
			{Index: 2, Score: -2.306173801422119},
			{Index: 3, Score: -11.033184051513672},
		},
	}
	nr := normalizingReranker{inner: inner}

	got, err := nr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{"a", "b", "c"}})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}

	for i, r := range got {
		if r.Score <= 0 || r.Score >= 1 {
			t.Errorf("result[%d].Score = %v, want in (0,1)", i, r.Score)
		}
	}
	// Sigmoid is monotonic, so index order and descending order must survive.
	wantIdx := []int{0, 2, 3}
	for i, w := range wantIdx {
		if got[i].Index != w {
			t.Errorf("result[%d].Index = %d, want %d (order must be preserved)", i, got[i].Index, w)
		}
	}
	for i := 1; i < len(got); i++ {
		if got[i-1].Score < got[i].Score {
			t.Errorf("results not descending at %d: %v < %v", i, got[i-1].Score, got[i].Score)
		}
	}
	// Spot-check the transform: sigmoid(1.7297...) ≈ 0.8494.
	if d := got[0].Score - 0.8494; d > 1e-3 || d < -1e-3 {
		t.Errorf("sigmoid(top logit) = %v, want ≈0.8494", got[0].Score)
	}
}

func TestNormalizingRerankerPropagatesError(t *testing.T) {
	t.Parallel()
	sentinel := errors.New("boom")
	nr := normalizingReranker{inner: stubReranker{err: sentinel}}

	_, err := nr.Rerank(context.Background(), RerankRequest{Documents: []string{"a"}})
	if !errors.Is(err, sentinel) {
		t.Errorf("err = %v, want sentinel error", err)
	}
}

func TestNormalizingRerankerModelPassthrough(t *testing.T) {
	t.Parallel()
	nr := normalizingReranker{inner: stubReranker{model: "bge-reranker-v2-m3"}}
	if nr.Model() != "bge-reranker-v2-m3" {
		t.Errorf("Model() = %q, want bge-reranker-v2-m3", nr.Model())
	}
}

func TestClassifyRerankHTTPError(t *testing.T) {
	t.Parallel()

	base := errors.New("rerank: HTTP error body")

	tests := []struct {
		name          string
		status        int
		body          string
		wantUnavail   bool // IsRerankAvailable should report the inverse
		wantPermanent bool // result should be a *PermanentError
		wantTooLong   bool // PermanentError.TooLong, when permanent
	}{
		{name: "503 unhealthy", status: 503, wantUnavail: true},
		{name: "500 internal", status: 500, wantUnavail: true},
		{name: "502 bad gateway", status: 502, wantUnavail: true},
		{name: "429 backpressure", status: 429, wantUnavail: true},
		{name: "400 bad request", status: 400, wantPermanent: true},
		{name: "401 auth", status: 401, wantPermanent: true},
		{name: "404 unknown model", status: 404, wantPermanent: true},
		{name: "413 oversize pair", status: 413, wantPermanent: true, wantTooLong: true},
		{
			name:          "400 context length body",
			status:        400,
			body:          "input exceeds the maximum context length",
			wantPermanent: true,
			wantTooLong:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := classifyRerankHTTPError(base, tt.status, []byte(tt.body))
			if got == nil {
				t.Fatal("classifyRerankHTTPError returned nil for a non-2xx status")
			}

			if gotUnavail := !IsRerankAvailable(got); gotUnavail != tt.wantUnavail {
				t.Errorf("!IsRerankAvailable = %v, want %v", gotUnavail, tt.wantUnavail)
			}

			var pe *PermanentError
			gotPermanent := errors.As(got, &pe)
			if gotPermanent != tt.wantPermanent {
				t.Errorf("errors.As(*PermanentError) = %v, want %v", gotPermanent, tt.wantPermanent)
			}
			if gotPermanent && pe.TooLong != tt.wantTooLong {
				t.Errorf("PermanentError.TooLong = %v, want %v", pe.TooLong, tt.wantTooLong)
			}

			// A permanent (request) error must never look unavailable: silently
			// degrading on a caller bug is exactly what we want to avoid.
			if tt.wantPermanent && !IsRerankAvailable(got) {
				t.Error("a PermanentError must not report as unavailable")
			}
		})
	}
}

func TestIsRerankAvailable(t *testing.T) {
	t.Parallel()

	// want is the availability verdict: false means "degrade to first stage".
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: true},
		{name: "sentinel", err: ErrRerankUnavailable, want: false},
		{
			name: "wrapped sentinel",
			err:  fmt.Errorf("rerank: %w", ErrRerankUnavailable),
			want: false,
		},
		{
			name: "double-wrapped sentinel",
			err:  fmt.Errorf("%w: %w", ErrRerankUnavailable, errors.New("HTTP 503")),
			want: false,
		},
		{name: "context deadline", err: context.DeadlineExceeded, want: false},
		{
			name: "wrapped context deadline",
			err:  fmt.Errorf("rerank: %w", context.DeadlineExceeded),
			want: false,
		},
		{name: "net timeout", err: fakeTimeoutErr{}, want: false},
		{name: "net non-timeout", err: fakeNetErr{}, want: true},
		{name: "plain error", err: errors.New("boom"), want: true},
		{name: "context canceled", err: context.Canceled, want: true},
		{
			name: "permanent error",
			err:  &PermanentError{Err: errors.New("HTTP 400")},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := IsRerankAvailable(tt.err); got != tt.want {
				t.Errorf("IsRerankAvailable(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

// Guard the contract that a sub-deadline trips degradation: a context that has
// already expired, surfaced unwrapped, reports unavailable (false).
func TestIsRerankAvailableExpiredContext(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	<-ctx.Done()
	if IsRerankAvailable(ctx.Err()) {
		t.Errorf("expired context error should be unavailable, got err=%v", ctx.Err())
	}
}
