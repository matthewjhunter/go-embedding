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
)

// maxRerankBatch caps how many documents the Jina backend sends in one
// request; larger candidate sets are split across requests and merged. The
// value is conservative — most rerank servers (llama.cpp, TEI, vLLM, Cohere)
// accept far more, and the real consumer (a search shortlist of ~30-50) never
// splits. It is a constant rather than config until a backend's true per-request
// limit forces tuning.
const maxRerankBatch = 256

// JinaReranker implements Reranker against a Cohere/Jina-style rerank endpoint
// (POST {BaseURL}/v1/rerank). This is the de-facto standard rerank protocol:
// llama.cpp (--reranking), vLLM, Infinity, and the Cohere and Jina clouds all
// speak it. APIKey, when set, is sent as a Bearer token; it is omitted for
// unauthenticated local sidecars.
//
// Documents are truncated to the model's registered byte budget (limits.go)
// before sending, so an over-length (query+document) pair can't trip a serving
// stack that rejects oversize inputs (e.g. llama.cpp's HTTP 500). Strict mode
// rejects instead of truncating; an unregistered model is never clipped.
type JinaReranker struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
	// strict mirrors RerankConfig.Strict: when true, a document that exceeds the
	// model's registered byte budget is rejected (PermanentError) rather than
	// truncated. Set by NewReranker; NewJinaReranker leaves it false (truncate).
	strict bool
}

// NewJinaReranker creates a reranker that calls POST {baseURL}/v1/rerank.
// apiKey may be empty for unauthenticated local endpoints.
func NewJinaReranker(baseURL, apiKey, model string) *JinaReranker {
	return &JinaReranker{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		client:  &http.Client{},
	}
}

type jinaRerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

type jinaRerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// Rerank scores req.Documents against req.Query and returns results sorted by
// descending Score, each carrying its index in req.Documents. See Reranker.
//
// top_n is never sent to the backend: the cutoff is applied client-side after
// merging, which is the only correct behaviour when a large candidate set is
// fanned out across multiple requests (a per-request top_n would drop
// candidates). req.Instruction, when set, is folded into the query because the
// Cohere/Jina wire shape has no instruction field.
func (r *JinaReranker) Rerank(ctx context.Context, req RerankRequest) ([]RerankResult, error) {
	if len(req.Documents) == 0 {
		return nil, nil
	}

	query := req.Query
	if req.Instruction != "" {
		// Plain cross-encoders treat the whole string as the query;
		// instruction-tuned models read the leading task line.
		query = req.Instruction + "\n" + req.Query
	}

	// Truncate each document to the model's registered byte budget before
	// sending. A rerank server processes a (query+document) pair as one
	// non-causal sequence that must fit its context/micro-batch in a single
	// forward pass; an oversize pair is rejected (e.g. llama.cpp returns HTTP
	// 500), which the consumer would misread as a transient outage and silently
	// degrade. Truncation preserves document count and order, so the indices
	// returned below still map back to the caller's slice. Unregistered models
	// have no budget and pass through unchanged.
	docs, err := applyLimits(req.Documents, r.model, r.strict)
	if err != nil {
		return nil, err
	}

	results := make([]RerankResult, 0, len(docs))
	for start := 0; start < len(docs); start += maxRerankBatch {
		end := min(start+maxRerankBatch, len(docs))
		batch, err := r.send(ctx, query, docs[start:end])
		if err != nil {
			return nil, err
		}
		for _, b := range batch {
			// Map the per-batch index back to the caller's original slice.
			results = append(results, RerankResult{Index: start + b.Index, Score: b.Score})
		}
	}

	// Each batch comes back sorted, but a fanned-out result must be re-sorted
	// across batches. Stable so equal scores keep input (first-stage) order.
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if req.TopN > 0 && req.TopN < len(results) {
		results = results[:req.TopN]
	}
	return results, nil
}

// send performs one /v1/rerank request over the given document batch. Indices
// in the returned results are relative to documents. A transport failure wraps
// ErrRerankUnavailable; a non-2xx response is classified via
// classifyRerankHTTPError.
func (r *JinaReranker) send(ctx context.Context, query string, documents []string) ([]RerankResult, error) {
	data, err := json.Marshal(jinaRerankRequest{
		Model:     r.model,
		Query:     query,
		Documents: documents,
	})
	if err != nil {
		return nil, fmt.Errorf("jina rerank: marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, r.baseURL+"/v1/rerank", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("jina rerank: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if r.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+r.apiKey)
	}

	resp, err := r.client.Do(httpReq)
	if err != nil {
		// A transport failure (connection refused/reset, DNS, TLS, timeout)
		// means the backend is unreachable: wrap so the consumer degrades to
		// first-stage order rather than failing the query.
		return nil, fmt.Errorf("%w: jina rerank: %w", ErrRerankUnavailable, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("jina rerank: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, classifyRerankHTTPError(
			fmt.Errorf("jina rerank: HTTP %d: %s", resp.StatusCode, body),
			resp.StatusCode, body)
	}

	var rerankResp jinaRerankResponse
	if err := json.Unmarshal(body, &rerankResp); err != nil {
		return nil, fmt.Errorf("jina rerank: unmarshal: %w", err)
	}

	results := make([]RerankResult, len(rerankResp.Results))
	for i, res := range rerankResp.Results {
		// Defensive: never index the caller's slice with a server-supplied
		// value without bounds-checking it.
		if res.Index < 0 || res.Index >= len(documents) {
			return nil, fmt.Errorf("jina rerank: result index %d out of range for %d documents", res.Index, len(documents))
		}
		results[i] = RerankResult{Index: res.Index, Score: res.RelevanceScore}
	}
	return results, nil
}

// Model returns the configured rerank model name.
func (r *JinaReranker) Model() string { return r.model }
