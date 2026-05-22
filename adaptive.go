package embedding

import "errors"

// minEmbedBytes is the floor for adaptive truncation. If a backend still
// rejects an input this small as too long, the cause is not length and
// further shrinking is pointless.
const minEmbedBytes = 256

// Adaptive truncation shrinks an oversize input to 80% of its length per
// retry.
const (
	shrinkFactorNum = 4
	shrinkFactorDen = 5
)

// embedShrinking sends texts via send and, when the backend rejects them as
// too long (a *PermanentError with TooLong set), truncates every input and
// retries down to minEmbedBytes.
//
// This is the backstop beneath the static byte budget in limits.go: dense,
// identifier-heavy text tokenizes past a model's context window at byte counts
// that ordinary prose clears, so a fixed budget cannot prevent every reject.
// Letting the backend be the authority — shrink only when it actually refuses
// — adapts across content types without a tokenizer. If truncation reaches the
// floor and the input still will not fit, the final *PermanentError is
// returned so the caller can quarantine it.
func embedShrinking(texts []string, send func([]string) ([][]float32, error)) ([][]float32, error) {
	out, err := send(texts)
	for err != nil {
		var pe *PermanentError
		if !errors.As(err, &pe) || !pe.TooLong {
			return nil, err
		}
		next, changed := shrinkTexts(texts)
		if !changed {
			return nil, err // at the floor; genuinely permanent
		}
		texts = next
		out, err = send(texts)
	}
	return out, nil
}

// shrinkTexts truncates each text by ~20% toward minEmbedBytes, returning the
// new slice and whether any input actually shrank. Truncation respects UTF-8
// boundaries via truncateToBytes.
func shrinkTexts(texts []string) ([]string, bool) {
	out := make([]string, len(texts))
	changed := false
	for i, t := range texts {
		if len(t) <= minEmbedBytes {
			out[i] = t
			continue
		}
		target := max(len(t)*shrinkFactorNum/shrinkFactorDen, minEmbedBytes)
		trunc := truncateToBytes(t, target)
		if len(trunc) != len(t) {
			changed = true
		}
		out[i] = trunc
	}
	return out, changed
}
