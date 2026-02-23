// Package embedding provides a common interface for text embedding models
// and utility functions for vector similarity and serialization.
package embedding

import (
	"context"
	"encoding/binary"
	"math"
)

// Embedder generates vector embeddings for text.
type Embedder interface {
	// Embed returns embedding vectors for each input text, in the same order.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// Model returns the name of the embedding model.
	Model() string
}

// Single embeds a single text and returns its vector.
func Single(ctx context.Context, e Embedder, text string) ([]float32, error) {
	results, err := e.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return results[0], nil
}

// CosineSimilarity returns the cosine similarity between two vectors in [-1, 1].
// Returns 0 if either vector has zero magnitude.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, magA, magB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		magA += float64(a[i]) * float64(a[i])
		magB += float64(b[i]) * float64(b[i])
	}
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

// EncodeFloat32s serializes a float32 slice to bytes (little-endian IEEE 754).
func EncodeFloat32s(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DecodeFloat32s deserializes bytes produced by EncodeFloat32s back to float32 slice.
func DecodeFloat32s(b []byte) []float32 {
	if len(b)%4 != 0 {
		return nil
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}
