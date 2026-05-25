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

func TestClassifyRerankHTTPError(t *testing.T) {
	t.Parallel()

	base := errors.New("rerank: HTTP error body")

	tests := []struct {
		name          string
		status        int
		body          string
		wantUnavail   bool // IsRerankUnavailable should report this
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

			if gotUnavail := IsRerankUnavailable(got); gotUnavail != tt.wantUnavail {
				t.Errorf("IsRerankUnavailable = %v, want %v", gotUnavail, tt.wantUnavail)
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
			if tt.wantPermanent && IsRerankUnavailable(got) {
				t.Error("a PermanentError must not report as unavailable")
			}
		})
	}
}

func TestIsRerankUnavailable(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "sentinel", err: ErrRerankUnavailable, want: true},
		{
			name: "wrapped sentinel",
			err:  fmt.Errorf("rerank: %w", ErrRerankUnavailable),
			want: true,
		},
		{
			name: "double-wrapped sentinel",
			err:  fmt.Errorf("%w: %w", ErrRerankUnavailable, errors.New("HTTP 503")),
			want: true,
		},
		{name: "context deadline", err: context.DeadlineExceeded, want: true},
		{
			name: "wrapped context deadline",
			err:  fmt.Errorf("rerank: %w", context.DeadlineExceeded),
			want: true,
		},
		{name: "net timeout", err: fakeTimeoutErr{}, want: true},
		{name: "net non-timeout", err: fakeNetErr{}, want: false},
		{name: "plain error", err: errors.New("boom"), want: false},
		{name: "context canceled", err: context.Canceled, want: false},
		{
			name: "permanent error",
			err:  &PermanentError{Err: errors.New("HTTP 400")},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := IsRerankUnavailable(tt.err); got != tt.want {
				t.Errorf("IsRerankUnavailable(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

// Guard the contract that a sub-deadline trips degradation: a context that has
// already expired, surfaced unwrapped, is unavailable.
func TestIsRerankUnavailableExpiredContext(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	<-ctx.Done()
	if !IsRerankUnavailable(ctx.Err()) {
		t.Errorf("expired context error should be unavailable, got err=%v", ctx.Err())
	}
}
