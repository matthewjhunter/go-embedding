package embedding

import (
	"errors"
	"strings"
	"testing"
)

func TestShrinkTexts(t *testing.T) {
	t.Run("shrinks by ~20%", func(t *testing.T) {
		in := []string{strings.Repeat("x", 1000)}
		out, changed := shrinkTexts(in)
		if !changed {
			t.Fatal("expected changed=true")
		}
		if len(out[0]) != 800 {
			t.Errorf("len = %d, want 800", len(out[0]))
		}
	})

	t.Run("floors at minEmbedBytes", func(t *testing.T) {
		in := []string{strings.Repeat("x", minEmbedBytes+10)}
		out, changed := shrinkTexts(in)
		if !changed {
			t.Fatal("expected changed=true")
		}
		if len(out[0]) != minEmbedBytes {
			t.Errorf("len = %d, want floor %d", len(out[0]), minEmbedBytes)
		}
	})

	t.Run("no change when already at or below floor", func(t *testing.T) {
		in := []string{strings.Repeat("x", minEmbedBytes), "short"}
		_, changed := shrinkTexts(in)
		if changed {
			t.Error("expected changed=false when all inputs <= floor")
		}
	})
}

// shrinkSend is a programmable send func for embedShrinking tests. It succeeds
// once every input's byte length is <= fitsAt; until then it returns the
// configured error.
type shrinkSend struct {
	fitsAt   int
	failWith error
	calls    int
}

func (s *shrinkSend) send(texts []string) ([][]float32, error) {
	s.calls++
	for _, t := range texts {
		if len(t) > s.fitsAt {
			return nil, s.failWith
		}
	}
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1}
	}
	return out, nil
}

func TestEmbedShrinking(t *testing.T) {
	tooLong := &PermanentError{Err: errors.New("context length exceeded"), TooLong: true}

	t.Run("success on first try does not shrink", func(t *testing.T) {
		s := &shrinkSend{fitsAt: 10_000, failWith: tooLong}
		out, err := embedShrinking([]string{"short"}, s.send)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(out) != 1 || s.calls != 1 {
			t.Errorf("calls=%d, want 1", s.calls)
		}
	})

	t.Run("shrinks until it fits", func(t *testing.T) {
		s := &shrinkSend{fitsAt: 500, failWith: tooLong}
		out, err := embedShrinking([]string{strings.Repeat("x", 5000)}, s.send)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(out) != 1 {
			t.Fatalf("want 1 vector, got %d", len(out))
		}
		if s.calls < 2 {
			t.Errorf("expected multiple attempts, got %d", s.calls)
		}
	})

	t.Run("gives up at floor with permanent error", func(t *testing.T) {
		s := &shrinkSend{fitsAt: 0, failWith: tooLong} // never fits
		_, err := embedShrinking([]string{strings.Repeat("x", 5000)}, s.send)
		if err == nil {
			t.Fatal("expected error")
		}
		if IsRetryable(err) {
			t.Error("expected a permanent (non-retryable) error at the floor")
		}
	})

	t.Run("non-TooLong permanent error returned without shrinking", func(t *testing.T) {
		authErr := &PermanentError{Err: errors.New("bad key")}
		s := &shrinkSend{fitsAt: 0, failWith: authErr}
		_, err := embedShrinking([]string{strings.Repeat("x", 5000)}, s.send)
		if err == nil {
			t.Fatal("expected error")
		}
		if s.calls != 1 {
			t.Errorf("expected no shrink retries, got %d calls", s.calls)
		}
	})

	t.Run("transient error returned without shrinking", func(t *testing.T) {
		s := &shrinkSend{fitsAt: 0, failWith: errors.New("connection reset")}
		_, err := embedShrinking([]string{strings.Repeat("x", 5000)}, s.send)
		if err == nil {
			t.Fatal("expected error")
		}
		if !IsRetryable(err) {
			t.Error("expected transient error to remain retryable")
		}
		if s.calls != 1 {
			t.Errorf("expected no shrink retries for transient error, got %d calls", s.calls)
		}
	})
}
