package embedding

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float32{1, 2, 3}
	sim := CosineSimilarity(a, a)
	if math.Abs(sim-1.0) > 1e-6 {
		t.Errorf("identical vectors: got %f, want 1.0", sim)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	sim := CosineSimilarity(a, b)
	if math.Abs(sim) > 1e-6 {
		t.Errorf("orthogonal vectors: got %f, want 0.0", sim)
	}
}

func TestCosineSimilarity_DifferentLengths(t *testing.T) {
	a := []float32{1, 2}
	b := []float32{1, 2, 3}
	sim := CosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("different length vectors: got %f, want 0.0", sim)
	}
}

func TestCosineSimilarity_Empty(t *testing.T) {
	sim := CosineSimilarity(nil, nil)
	if sim != 0 {
		t.Errorf("empty vectors: got %f, want 0.0", sim)
	}
}

func TestCosineSimilarity_ZeroMagnitude(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	sim := CosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("zero magnitude vector: got %f, want 0.0", sim)
	}
}

func TestEncodeDecodeFloat32s_Roundtrip(t *testing.T) {
	original := []float32{1.5, -2.3, 0, math.MaxFloat32, math.SmallestNonzeroFloat32}
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

func TestEncodeFloat32s_Empty(t *testing.T) {
	encoded := EncodeFloat32s(nil)
	if len(encoded) != 0 {
		t.Errorf("empty input should produce empty output, got %d bytes", len(encoded))
	}
}

func TestDecodeFloat32s_Empty(t *testing.T) {
	decoded := DecodeFloat32s(nil)
	if len(decoded) != 0 {
		t.Errorf("empty input should produce empty output, got %d elements", len(decoded))
	}
}

func TestOllamaEmbedder_Embed(t *testing.T) {
	// Mock Ollama server
	expectedModel := "embeddinggemma"
	mockEmbeddings := [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Model != expectedModel {
			t.Errorf("unexpected model: got %s, want %s", req.Model, expectedModel)
		}
		if len(req.Input) != 2 {
			t.Errorf("unexpected input count: got %d, want 2", len(req.Input))
		}

		resp := ollamaEmbedResponse{Embeddings: mockEmbeddings}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewOllamaEmbedder(server.URL, expectedModel)
	results, err := embedder.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(results))
	}
	for i := range mockEmbeddings {
		for j := range mockEmbeddings[i] {
			if results[i][j] != mockEmbeddings[i][j] {
				t.Errorf("embedding[%d][%d]: got %f, want %f", i, j, results[i][j], mockEmbeddings[i][j])
			}
		}
	}
}

func TestOllamaEmbedder_Model(t *testing.T) {
	embedder := NewOllamaEmbedder("http://localhost:11434", "embeddinggemma")
	if embedder.Model() != "embeddinggemma" {
		t.Errorf("Model(): got %s, want embeddinggemma", embedder.Model())
	}
}

func TestOllamaEmbedder_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer server.Close()

	embedder := NewOllamaEmbedder(server.URL, "nonexistent")
	_, err := embedder.Embed(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error for HTTP 404")
	}
}

func TestSingle(t *testing.T) {
	mockVec := []float32{0.1, 0.2, 0.3}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ollamaEmbedResponse{Embeddings: [][]float32{mockVec}}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	embedder := NewOllamaEmbedder(server.URL, "embeddinggemma")
	vec, err := Single(context.Background(), embedder, "hello")
	if err != nil {
		t.Fatalf("Single failed: %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("expected 3-element vector, got %d", len(vec))
	}
	for i := range mockVec {
		if vec[i] != mockVec[i] {
			t.Errorf("index %d: got %f, want %f", i, vec[i], mockVec[i])
		}
	}
}
