package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
)

// OllamaEmbedder implements Embedder using the Ollama HTTP API (POST /api/embed).
type OllamaEmbedder struct {
	baseURL string
	model   string
	client  *http.Client
	dim     atomic.Int32
	strict  bool
}

// NewOllamaEmbedder creates an embedder that calls the Ollama /api/embed endpoint.
//
// Deprecated: prefer New(Config{Backend: BackendOllama, BaseURL: baseURL, Model: model}).
// This constructor will be removed in v1.0.
func NewOllamaEmbedder(baseURL, model string) *OllamaEmbedder {
	return &OllamaEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		client:  &http.Client{},
	}
}

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// Embed generates vector embeddings for the given texts via the Ollama API.
// Oversize input is first truncated to the model's registered byte budget
// (see limits.go), then, if the backend still rejects it as too long,
// adaptively shrunk and retried (see embedShrinking).
func (e *OllamaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	texts, err := applyLimits(texts, e.model, e.strict)
	if err != nil {
		return nil, err
	}
	return embedShrinking(texts, func(ts []string) ([][]float32, error) {
		return e.send(ctx, ts)
	})
}

// send performs one Ollama /api/embed request. A non-2xx response is
// classified via classifyHTTPError so callers can distinguish permanent
// (4xx, too-long) from transient (5xx) failures.
func (e *OllamaEmbedder) send(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := ollamaEmbedRequest{
		Model: e.model,
		Input: texts,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/api/embed", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ollama embed: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, classifyHTTPError(
			fmt.Errorf("ollama embed: HTTP %d: %s", resp.StatusCode, body),
			resp.StatusCode, body)
	}

	var embedResp ollamaEmbedResponse
	if err := json.Unmarshal(body, &embedResp); err != nil {
		return nil, fmt.Errorf("ollama embed: unmarshal: %w", err)
	}

	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama embed: empty response")
	}

	if d := len(embedResp.Embeddings[0]); d > 0 {
		e.dim.Store(int32(d))
	}

	return embedResp.Embeddings, nil
}

// Model returns the configured embedding model name.
func (e *OllamaEmbedder) Model() string { return e.model }

// Fingerprint returns the model name and vector dimension. Dim is 0 until
// the first successful Embed call.
func (e *OllamaEmbedder) Fingerprint() Fingerprint {
	return Fingerprint{Model: e.model, Dim: int(e.dim.Load())}
}
