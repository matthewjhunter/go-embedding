package embedding

import (
	"math"
	"testing"
)

func TestCosineSimilarity_Identical(t *testing.T) {
	v := []float32{1, 0, 0}
	got := CosineSimilarity(v, v)
	if math.Abs(got-1.0) > 1e-6 {
		t.Errorf("identical vectors: want 1.0, got %f", got)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	got := CosineSimilarity(a, b)
	if math.Abs(got) > 1e-6 {
		t.Errorf("orthogonal vectors: want 0.0, got %f", got)
	}
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{-1, 0, 0}
	got := CosineSimilarity(a, b)
	if math.Abs(got+1.0) > 1e-6 {
		t.Errorf("opposite vectors: want -1.0, got %f", got)
	}
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 0, 0}
	got := CosineSimilarity(a, b)
	if got != 0 {
		t.Errorf("zero vector: want 0.0, got %f", got)
	}
}

func TestCosineSimilarity_LengthMismatch(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{1, 0, 0}
	got := CosineSimilarity(a, b)
	if got != 0 {
		t.Errorf("length mismatch: want 0.0, got %f", got)
	}
}

func TestEncodeDecodeRoundtrip(t *testing.T) {
	original := []float32{1.5, -2.25, 0, 3.14159, math.MaxFloat32}
	encoded := EncodeFloat32s(original)
	decoded := DecodeFloat32s(encoded)

	if len(decoded) != len(original) {
		t.Fatalf("length mismatch: got %d, want %d", len(decoded), len(original))
	}
	for i := range original {
		if decoded[i] != original[i] {
			t.Errorf("index %d: got %f, want %f", i, decoded[i], original[i])
		}
	}
}

func TestEncodeFloat32s_Length(t *testing.T) {
	v := []float32{1, 2, 3}
	b := EncodeFloat32s(v)
	if len(b) != 12 {
		t.Errorf("expected 12 bytes for 3 float32s, got %d", len(b))
	}
}

func TestDecodeFloat32s_InvalidLength(t *testing.T) {
	b := []byte{1, 2, 3} // not a multiple of 4
	result := DecodeFloat32s(b)
	if result != nil {
		t.Errorf("expected nil for invalid input, got %v", result)
	}
}

func TestDecodeFloat32s_Empty(t *testing.T) {
	result := DecodeFloat32s([]byte{})
	if len(result) != 0 {
		t.Errorf("expected empty slice, got %v", result)
	}
}
