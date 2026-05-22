package embedding

import (
	"errors"
	"fmt"
	"net/http"
	"testing"
)

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"plain error", errors.New("boom"), true},
		{"permanent", &PermanentError{Err: errors.New("bad input")}, false},
		{"permanent too long", &PermanentError{Err: errors.New("too long"), TooLong: true}, false},
		{"wrapped permanent", fmt.Errorf("ctx: %w", &PermanentError{Err: errors.New("x")}), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsRetryable(tt.err); got != tt.want {
				t.Errorf("IsRetryable(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestIsContextLengthError(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"413 regardless of body", http.StatusRequestEntityTooLarge, "", true},
		{"ollama context length", http.StatusBadRequest, `{"error":"input exceeds the maximum context length"}`, true},
		{"too long", http.StatusBadRequest, "input is too long", true},
		{"number of tokens", http.StatusBadRequest, "requested number of tokens exceeds limit", true},
		{"auth not length", http.StatusUnauthorized, "invalid token provided", false},
		{"model not found", http.StatusNotFound, "model 'foo' not found", false},
		{"generic bad request", http.StatusBadRequest, "malformed json", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isContextLengthError(tt.status, []byte(tt.body)); got != tt.want {
				t.Errorf("isContextLengthError(%d, %q) = %v, want %v", tt.status, tt.body, got, tt.want)
			}
		})
	}
}

func TestClassifyHTTPError(t *testing.T) {
	base := errors.New("HTTP error")

	t.Run("4xx too long is permanent and TooLong", func(t *testing.T) {
		err := classifyHTTPError(base, http.StatusBadRequest, []byte("context length exceeded"))
		var pe *PermanentError
		if !errors.As(err, &pe) {
			t.Fatalf("expected *PermanentError, got %T", err)
		}
		if !pe.TooLong {
			t.Error("expected TooLong=true")
		}
	})

	t.Run("4xx auth is permanent not TooLong", func(t *testing.T) {
		err := classifyHTTPError(base, http.StatusUnauthorized, []byte("bad api key"))
		var pe *PermanentError
		if !errors.As(err, &pe) {
			t.Fatalf("expected *PermanentError, got %T", err)
		}
		if pe.TooLong {
			t.Error("expected TooLong=false for auth error")
		}
	})

	t.Run("5xx is transient", func(t *testing.T) {
		err := classifyHTTPError(base, http.StatusInternalServerError, []byte("upstream down"))
		if !IsRetryable(err) {
			t.Error("expected 5xx to be retryable")
		}
	})
}
