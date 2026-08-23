package embedding

import (
	"context"
	"fmt"
)

// BatchItem is one record's worth of inputs: the rendered texts to embed, in
// order. A record that chunks into several texts contributes them all here, so
// batching happens across records rather than one request per record.
type BatchItem struct {
	Texts []string
}

// BatchItemResult is the outcome for one BatchItem, index-aligned with the
// slice BatchEmbedItems was given.
//
// Empty Vectors with a nil Err means the item had no texts -- a deterministic
// skip rather than a failure. A non-nil Err always comes with no vectors: an
// item embeds completely or not at all.
type BatchItemResult struct {
	Vectors [][]float32
	Err     error
}

// BatchEmbedItems embeds a slice of records in batched requests, returning one
// result per record in the same order.
//
// BatchEmbedResults handles a flat list of texts, but what callers have is a
// list of records that each chunk into several. Every caller that has needed
// this wrote the same flatten-and-regroup around it, so it lives here now.
//
// batchSize counts inputs, not records: chunks are flattened across records
// before batching, since one request per record would pay the per-request
// overhead on every document and the backends serialise per model anyway.
//
// A failed chunk fails its whole record. That is the part worth stating
// explicitly, because the alternative looks harmless and is not: a
// half-embedded record reads as *complete* to a retry query -- its marker is
// set, so a backfill skips it -- while leaving a permanent hole in the middle
// of the document. Failing whole means the record is retried whole.
//
// One record's failure never affects another's. BatchEmbedResults falls back to
// embedding one at a time when a batch errors, so a single unembeddable input
// fails alone rather than taking the rest of its batch with it.
func BatchEmbedItems(ctx context.Context, e Embedder, items []BatchItem, batchSize int) []BatchItemResult {
	out := make([]BatchItemResult, len(items))

	// Flatten, remembering which record each text came from so the vectors can
	// be zipped back afterwards.
	var texts []string
	var owner []int
	for i, it := range items {
		for _, t := range it.Texts {
			texts = append(texts, t)
			owner = append(owner, i)
		}
	}
	if len(texts) == 0 {
		return out
	}

	res, err := BatchEmbedResults(ctx, e, texts, batchSize, nil)
	if len(res) != len(texts) {
		// Every input failed, or the helper broke its index-alignment
		// contract. Either way there is nothing safe to zip -- guessing would
		// put the wrong vectors on the wrong records, which is undetectable
		// downstream -- so fail every record with the cause.
		if err == nil {
			err = fmt.Errorf("embedding: got %d results for %d inputs", len(res), len(texts))
		}
		for _, i := range owner {
			out[i].Err = err
		}
		return out
	}

	for j, r := range res {
		i := owner[j]
		if out[i].Err != nil {
			continue
		}
		if r.Err != nil {
			out[i] = BatchItemResult{Err: fmt.Errorf("embedding: item %d: %w", i, r.Err)}
			continue
		}
		out[i].Vectors = append(out[i].Vectors, r.Vector)
	}
	return out
}
