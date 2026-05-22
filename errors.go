package embedding

import (
	"errors"
	"net/http"
	"strings"
)

// PermanentError marks an embedding failure that will recur for identical
// input, so retrying it is futile. A background embed queue should quarantine
// the input rather than spin on it. Transient failures (timeouts, 5xx,
// connection resets) are deliberately NOT wrapped this way.
type PermanentError struct {
	// Err is the underlying cause.
	Err error
	// TooLong indicates the backend rejected the input as exceeding its
	// context window. The adaptive embed path truncates and retries such
	// inputs; a TooLong error that still escapes means truncation reached the
	// floor without fitting.
	TooLong bool
}

func (e *PermanentError) Error() string { return e.Err.Error() }
func (e *PermanentError) Unwrap() error { return e.Err }

// IsRetryable reports whether a non-nil error from an embed call is worth
// retrying. Permanent failures — HTTP 4xx, input rejected as too large,
// strict-mode oversize — return false; transient failures (timeouts, 5xx,
// connection errors) return true.
func IsRetryable(err error) bool {
	var pe *PermanentError
	return !errors.As(err, &pe)
}

// classifyHTTPError wraps a non-2xx embed response. A 4xx is permanent — the
// request itself is the problem — and is additionally flagged TooLong when the
// body indicates the input exceeded the model's context window, so the
// adaptive path can truncate and retry. 5xx and anything else are treated as
// transient and returned unwrapped.
func classifyHTTPError(err error, status int, body []byte) error {
	if status >= 400 && status < 500 {
		return &PermanentError{Err: err, TooLong: isContextLengthError(status, body)}
	}
	return err
}

// isContextLengthError reports whether a 4xx embed rejection looks like the
// input exceeding the model's context window, rather than an auth, model, or
// malformed-request error. HTTP 413 matches outright; otherwise it scans the
// response body for the phrasings Ollama and OpenAI-compatible backends use.
// The bare word "token" is intentionally excluded — it collides with auth
// errors ("invalid token") that have nothing to do with length.
func isContextLengthError(status int, body []byte) bool {
	if status == http.StatusRequestEntityTooLarge {
		return true
	}
	s := strings.ToLower(string(body))
	for _, kw := range []string{
		"context length", "context window", "maximum context",
		"too long", "too large", "exceeds", "exceed the",
		"input length", "number of tokens", "maximum tokens",
	} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}
