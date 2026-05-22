package embedding

import (
	"context"
	"strconv"
	"strings"

	lru "github.com/hashicorp/golang-lru/v2"
)

// QueryCache is a bounded, thread-safe LRU mapping a normalized query (scoped
// to the embedding model + vector dimension that produced it) to its embedding
// vector.
//
// It targets the retrieval-query path, where the same text is embedded
// repeatedly (startup recall, re-runs, paging) and each embed is a network
// round-trip to the backend. Embeddings are deterministic per model, so a hit
// is exact, not approximate. Do not use it for document/ingest embedding,
// where every input is distinct: caching there never hits and only evicts
// useful query entries.
//
// A nil *QueryCache is a valid disabled cache: every method falls straight
// through to the underlying Embedder, so callers built with NewQueryCache(0)
// need not special-case the disabled state.
type QueryCache struct {
	lru *lru.Cache[string, []float32]
}

// NewQueryCache returns a cache holding up to capacity entries. A capacity <= 0
// returns a nil *QueryCache (caching disabled), which remains safe to call.
func NewQueryCache(capacity int) *QueryCache {
	if capacity <= 0 {
		return nil
	}
	// lru.New only errors on a non-positive size, already excluded above.
	c, _ := lru.New[string, []float32](capacity)
	return &QueryCache{lru: c}
}

// Single returns the embedding for text, serving it from the cache when
// present and embedding via Single (with retries) on a miss. The returned
// slice is shared with the cache and with other callers; treat it as
// read-only.
func (c *QueryCache) Single(ctx context.Context, e Embedder, text string) ([]float32, error) {
	if c == nil {
		return Single(ctx, e, text)
	}
	if emb, ok := c.lru.Get(cacheKey(e.Fingerprint(), text)); ok {
		return emb, nil
	}
	emb, err := Single(ctx, e, text)
	if err != nil {
		return nil, err
	}
	// Re-read the fingerprint after embedding: Dim is 0 until the first
	// successful Embed, so storing under the post-embed fingerprint keeps the
	// key stable for subsequent lookups.
	c.lru.Add(cacheKey(e.Fingerprint(), text), emb)
	return emb, nil
}

// Embed returns embeddings for texts in order, serving cache hits and
// batch-embedding only the misses through e (via EmbedWithRetry). Returned
// slices are shared with the cache; treat them as read-only.
func (c *QueryCache) Embed(ctx context.Context, e Embedder, texts []string) ([][]float32, error) {
	if c == nil {
		return EmbedWithRetry(ctx, e, texts)
	}

	out := make([][]float32, len(texts))
	fp := e.Fingerprint()
	var missIdx []int
	var missTexts []string
	for i, t := range texts {
		if emb, ok := c.lru.Get(cacheKey(fp, t)); ok {
			out[i] = emb
		} else {
			missIdx = append(missIdx, i)
			missTexts = append(missTexts, t)
		}
	}
	if len(missTexts) == 0 {
		return out, nil
	}

	embs, err := EmbedWithRetry(ctx, e, missTexts)
	if err != nil {
		return nil, err
	}
	fpAfter := e.Fingerprint() // Dim is populated post-embed; see Single.
	for j, idx := range missIdx {
		if j >= len(embs) {
			break
		}
		out[idx] = embs[j]
		c.lru.Add(cacheKey(fpAfter, missTexts[j]), embs[j])
	}
	return out, nil
}

// Len reports the number of entries currently cached. A nil cache reports 0.
func (c *QueryCache) Len() int {
	if c == nil {
		return 0
	}
	return c.lru.Len()
}

// cacheKey scopes a normalized query to a specific model and vector dimension.
// Including the fingerprint guarantees that a model change (a different name,
// or the same name producing a different vector shape) can never serve vectors
// from the old model's space. NUL separators keep the components unambiguous.
func cacheKey(fp Fingerprint, text string) string {
	var b strings.Builder
	b.WriteString(fp.Model)
	b.WriteByte(0)
	b.WriteString(strconv.Itoa(fp.Dim))
	b.WriteByte(0)
	b.WriteString(normalizeQuery(text))
	return b.String()
}

// normalizeQuery folds queries that should share an embedding onto one key:
// surrounding whitespace trimmed, case lowered, internal whitespace runs
// collapsed to a single space. This is an intentional approximation suited to
// retrieval queries, where these surface differences do not change meaning.
// The raw text is still what gets embedded on a miss.
func normalizeQuery(q string) string {
	return strings.Join(strings.Fields(strings.ToLower(q)), " ")
}
