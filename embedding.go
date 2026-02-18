// Package embedding provides vector embedding utilities: an Embedder interface,
// cosine similarity, and binary encoding for float32 vectors. It also includes
// an Ollama-backed Embedder implementation suitable for local inference.
package embedding

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
)

// Embedder produces vector embeddings for text.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Model returns a stable identifier for the embedding model (e.g.
	// "embeddinggemma"). Callers may record this to detect model changes.
	Model() string
}

// embedMaxRetries is the number of retries for transient embedding failures
// (e.g. model loading timeouts). Total attempts = embedMaxRetries + 1.
const embedMaxRetries = 2

// EmbedWithRetry calls e.Embed, retrying up to embedMaxRetries times on
// failure. Returns immediately on context cancellation.
func EmbedWithRetry(ctx context.Context, e Embedder, texts []string) ([][]float32, error) {
	var result [][]float32
	var err error
	for attempt := range embedMaxRetries + 1 {
		result, err = e.Embed(ctx, texts)
		if err == nil {
			return result, nil
		}
		if attempt < embedMaxRetries && ctx.Err() != nil {
			break // caller gave up; don't burn retries
		}
	}
	return nil, fmt.Errorf("embedding failed after %d attempts: %w", embedMaxRetries+1, err)
}

// Single embeds a single text using the given Embedder, with retries.
func Single(ctx context.Context, e Embedder, text string) ([]float32, error) {
	results, err := EmbedWithRetry(ctx, e, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}
	return results[0], nil
}

// CosineSimilarity computes the cosine similarity between two vectors.
// Returns 0 if the vectors differ in length, are empty, or have zero magnitude.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		fa, fb := float64(a[i]), float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// EncodeFloat32s serializes a float32 slice to a little-endian byte slice,
// suitable for storing as a BLOB in SQLite.
func EncodeFloat32s(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DecodeFloat32s deserializes a little-endian byte slice back to a float32 slice.
func DecodeFloat32s(buf []byte) []float32 {
	n := len(buf) / 4
	v := make([]float32, n)
	for i := range n {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return v
}
