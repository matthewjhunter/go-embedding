package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync/atomic"
)

// OpenAIEmbedder implements Embedder using the OpenAI-compatible embeddings API
// (POST /v1/embeddings). Works with OpenAI, LiteLLM, vLLM, Ollama (>=0.1.24),
// and any other service that speaks the OpenAI embeddings protocol.
type OpenAIEmbedder struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
	dim     atomic.Int32
	strict  bool
}

// NewOpenAIEmbedder creates an embedder that calls POST /v1/embeddings.
// apiKey may be empty for unauthenticated local endpoints.
//
// Deprecated: prefer New(Config{Backend: BackendOpenAI, BaseURL: baseURL, APIKey: apiKey, Model: model}).
// This constructor will be removed in v1.0.
func NewOpenAIEmbedder(baseURL, apiKey, model string) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		client:  &http.Client{},
	}
}

type openAIEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

// Embed generates vector embeddings for the given texts via the OpenAI API.
// Oversize input is first truncated to the model's registered byte budget
// (see limits.go), then, if the backend still rejects it as too long,
// adaptively shrunk and retried (see embedShrinking).
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	texts, err := applyLimits(texts, e.model, e.strict)
	if err != nil {
		return nil, err
	}
	return embedShrinking(texts, func(ts []string) ([][]float32, error) {
		return e.send(ctx, ts)
	})
}

// send performs one OpenAI /v1/embeddings request. A non-2xx response is
// classified via classifyHTTPError so callers can distinguish permanent
// (4xx, too-long) from transient (5xx) failures.
func (e *OpenAIEmbedder) send(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := openAIEmbedRequest{
		Model: e.model,
		Input: texts,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("openai embed: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/v1/embeddings", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("openai embed: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai embed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai embed: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, classifyHTTPError(
			fmt.Errorf("openai embed: HTTP %d: %s", resp.StatusCode, body),
			resp.StatusCode, body)
	}

	var embedResp openAIEmbedResponse
	if err := json.Unmarshal(body, &embedResp); err != nil {
		return nil, fmt.Errorf("openai embed: unmarshal: %w", err)
	}

	if len(embedResp.Data) == 0 {
		return nil, fmt.Errorf("openai embed: empty response")
	}

	// Sort by index to preserve input order (OpenAI spec allows any order).
	sort.Slice(embedResp.Data, func(i, j int) bool {
		return embedResp.Data[i].Index < embedResp.Data[j].Index
	})

	results := make([][]float32, len(embedResp.Data))
	for i, d := range embedResp.Data {
		results[i] = d.Embedding
	}

	if d := len(results[0]); d > 0 {
		e.dim.Store(int32(d))
	}

	return results, nil
}

// Model returns the configured embedding model name.
func (e *OpenAIEmbedder) Model() string { return e.model }

// Fingerprint returns the model name and vector dimension. Dim is 0 until
// the first successful Embed call.
func (e *OpenAIEmbedder) Fingerprint() Fingerprint {
	return Fingerprint{Model: e.model, Dim: int(e.dim.Load())}
}
