package embedding

import (
	"context"
	"fmt"
	"log"
)

// defaultBatchSize is the batch size used when callers pass batchSize <= 0.
// Matches the GAIA code-index value, which works well for both Ollama and
// OpenAI-compatible backends without tripping per-request size limits.
const defaultBatchSize = 25

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
func BatchEmbed(
	ctx context.Context,
	e Embedder,
	texts []string,
	batchSize int,
	progress func(done, total int),
) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if batchSize <= 0 {
		batchSize = defaultBatchSize
	}

	out := make([][]float32, len(texts))
	total := len(texts)

	for batchStart := 0; batchStart < total; batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > total {
			batchEnd = total
		}
		batch := texts[batchStart:batchEnd]

		vectors, err := e.Embed(ctx, batch)
		if err == nil && len(vectors) == len(batch) {
			copy(out[batchStart:batchEnd], vectors)
		} else {
			if err != nil {
				log.Printf("embedding: BatchEmbed: batch [%d:%d] failed (%v); falling back to one-by-one", batchStart, batchEnd, err)
			} else {
				log.Printf("embedding: BatchEmbed: batch [%d:%d] returned %d vectors for %d inputs; falling back to one-by-one", batchStart, batchEnd, len(vectors), len(batch))
			}
			for i, text := range batch {
				v, err := e.Embed(ctx, []string{text})
				if err != nil || len(v) == 0 {
					log.Printf("embedding: BatchEmbed: input %d failed: %v", batchStart+i, err)
					continue
				}
				out[batchStart+i] = v[0]
			}
		}

		if progress != nil {
			progress(batchEnd, total)
		}
	}

	successes := 0
	for _, v := range out {
		if v != nil {
			successes++
		}
	}
	if successes == 0 {
		return out, fmt.Errorf("embedding: BatchEmbed: all %d inputs failed", total)
	}
	return out, nil
}
