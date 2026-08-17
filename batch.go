package embedding

import (
	"context"
	"errors"
	"fmt"
	"log"
)

// defaultBatchSize is the batch size used when callers pass batchSize <= 0.
// Matches the GAIA code-index value, which works well for both Ollama and
// OpenAI-compatible backends without tripping per-request size limits.
const defaultBatchSize = 25

// ErrShortBatch reports that a batch request returned fewer vectors than it
// was given inputs, without reporting an error of its own. It appears as an
// ItemError's Batch cause on inputs that then failed their individual retry.
var ErrShortBatch = errors.New("embedding: batch returned fewer vectors than inputs")

// errNoVector is the cause recorded when a backend accepts an input and
// returns success, but no vector with it. Reported as a failure rather than a
// silent nil so callers can tell it apart from an input they chose to skip.
var errNoVector = errors.New("embedding: backend returned no vector")

// BatchResult is the outcome for one input of BatchEmbedResults. Exactly one
// of Vector and Err is set.
type BatchResult struct {
	// Vector is the embedding, or nil when the input failed.
	Vector []float32
	// Err is the failure, or nil on success. It is always an *ItemError.
	Err error
}

// ItemError is the per-input failure reported by BatchEmbedResults. It
// unwraps to the cause of the individual embed attempt — the message worth
// storing for diagnosis — while Batch separately records whether the
// enclosing batch request failed as a whole.
//
// That split is what lets a caller tell one poisoned input apart from a dead
// backend. A single oversized input fails with Batch set (its batch was
// retried one-by-one because of it) but its neighbours succeed; a backend
// that is down fails every input in the batch with the same Batch cause. A
// caller draining a large queue can stop on the second rather than grinding
// through the remaining work to fail identically.
type ItemError struct {
	// Index is the position of the failed input in the texts slice passed to
	// BatchEmbedResults.
	Index int
	// Err is the cause of the individual attempt for this input.
	Err error
	// Batch is the batch-level cause when the enclosing batch request failed
	// as a whole (an error from the backend, or ErrShortBatch), and nil when
	// only this input failed.
	Batch error
}

func (e *ItemError) Error() string { return fmt.Sprintf("embedding: input %d: %v", e.Index, e.Err) }
func (e *ItemError) Unwrap() error { return e.Err }

// BatchEmbedResults embeds texts in batches of batchSize via e.Embed, falling
// back to one-by-one embedding when a batch returns either an error or a short
// response (fewer vectors than inputs). The optional progress callback is
// invoked once per batch with (done, total).
//
// The returned slice is always len(texts) long and index-aligned with it.
// Failed entries carry an *ItemError explaining why, so a caller can record
// the cause, decide whether the failure is retryable (see IsRetryable), and
// distinguish a bad input from a failing backend.
//
// It returns a non-nil error only when every input failed even after
// one-by-one fallback. Partial failures are reported per entry.
func BatchEmbedResults(
	ctx context.Context,
	e Embedder,
	texts []string,
	batchSize int,
	progress func(done, total int),
) ([]BatchResult, error) {
	if len(texts) == 0 {
		return []BatchResult{}, nil
	}
	if batchSize <= 0 {
		batchSize = defaultBatchSize
	}

	out := make([]BatchResult, len(texts))
	total := len(texts)

	for batchStart := 0; batchStart < total; batchStart += batchSize {
		batchEnd := min(batchStart+batchSize, total)
		batch := texts[batchStart:batchEnd]

		vectors, err := e.Embed(ctx, batch)
		if err == nil && len(vectors) == len(batch) {
			for i, v := range vectors {
				out[batchStart+i].Vector = v
			}
		} else {
			cause := err
			if cause == nil {
				cause = ErrShortBatch
				log.Printf("embedding: batch [%d:%d] returned %d vectors for %d inputs; falling back to one-by-one", batchStart, batchEnd, len(vectors), len(batch))
			} else {
				log.Printf("embedding: batch [%d:%d] failed (%v); falling back to one-by-one", batchStart, batchEnd, err)
			}
			embedOneByOne(ctx, e, batch, batchStart, cause, out)
		}

		if progress != nil {
			progress(batchEnd, total)
		}
	}

	var firstErr error
	successes := 0
	for _, r := range out {
		if r.Err == nil {
			successes++
		} else if firstErr == nil {
			firstErr = r.Err
		}
	}
	if successes == 0 {
		return out, fmt.Errorf("embedding: all %d inputs failed: %w", total, firstErr)
	}
	return out, nil
}

// embedOneByOne retries each input of a failed batch on its own, writing the
// outcome into out at the input's absolute index. batchCause is the batch-level
// failure that triggered the fallback, recorded on every input that also fails
// individually.
func embedOneByOne(ctx context.Context, e Embedder, batch []string, offset int, batchCause error, out []BatchResult) {
	for i, text := range batch {
		v, err := e.Embed(ctx, []string{text})
		switch {
		case err != nil:
			// Keep the individual cause; it is more specific than batchCause.
		case len(v) == 0 || v[0] == nil:
			err = errNoVector
		default:
			out[offset+i].Vector = v[0]
			continue
		}
		log.Printf("embedding: input %d failed: %v", offset+i, err)
		out[offset+i].Err = &ItemError{Index: offset + i, Err: err, Batch: batchCause}
	}
}

// BatchEmbed embeds texts in batches of batchSize via e.Embed, falling back
// to one-by-one embedding when a batch returns either an error or a short
// response (fewer vectors than inputs). The optional progress callback is
// invoked once per batch with (done, total).
//
// The returned slice is always len(texts) long. Failed entries are nil; the
// caller decides whether to drop them, retry later, or treat the whole
// result as invalid.
//
// BatchEmbed only returns a non-nil error when every input failed even
// after one-by-one fallback. Partial failures are signalled by nil entries
// — letting callers preserve index alignment with the original texts slice.
//
// Deprecated: use BatchEmbedResults, which reports why each input failed
// instead of collapsing every failure to a nil vector. A nil entry here
// cannot be told apart from an input the caller meant to skip, and it discards
// the cause a caller needs to decide whether retrying is worthwhile.
func BatchEmbed(
	ctx context.Context,
	e Embedder,
	texts []string,
	batchSize int,
	progress func(done, total int),
) ([][]float32, error) {
	results, err := BatchEmbedResults(ctx, e, texts, batchSize, progress)
	out := make([][]float32, len(results))
	for i, r := range results {
		out[i] = r.Vector
	}
	return out, err
}
