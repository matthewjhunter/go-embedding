package embedding

import "fmt"

// Backend identifies which embedding backend an Embedder uses.
type Backend string

const (
	// BackendOllama dispatches to the native Ollama /api/embed endpoint.
	BackendOllama Backend = "ollama"
	// BackendOpenAI dispatches to the OpenAI-compatible /v1/embeddings
	// endpoint. Works with OpenAI, LiteLLM, vLLM, Ollama (>=0.1.24),
	// Lemonade, and any other OpenAI-protocol service.
	BackendOpenAI Backend = "openai"
)

// Config configures a new Embedder.
//
// Backend, BaseURL, and Model are required. APIKey is optional and only
// meaningful for backends that authenticate (it is silently ignored by
// BackendOllama).
//
// Strict controls how the embedder reacts to text exceeding the model's
// registered Limits. When false (default), oversize text is truncated to
// the limit and a log line is emitted. When true, Embed returns an error
// instead of truncating.
//
// Model is treated as an opaque storage key by callers that persist
// embeddings. Do not canonicalise it (do not strip ":latest" or other
// tags, do not lowercase). Two model names that differ by even a tag
// suffix can produce incompatible vectors — a `:q4_0` quantization, a
// `:v2` version bump, or a `:latest` that pinned to different artifacts
// at different times all yield vector spaces that should not be merged.
// Limit lookups (LookupLimits) DO fall back to the bare name because
// limits are an architectural property; storage equivalence is not.
type Config struct {
	Backend Backend
	BaseURL string
	APIKey  string
	Model   string
	Strict  bool
}

// New constructs an Embedder from cfg. Returns an error if any required
// field is missing or if Backend is not recognised.
func New(cfg Config) (Embedder, error) {
	switch {
	case cfg.Backend == "":
		return nil, fmt.Errorf("embedding: Backend is required")
	case cfg.BaseURL == "":
		return nil, fmt.Errorf("embedding: BaseURL is required")
	case cfg.Model == "":
		return nil, fmt.Errorf("embedding: Model is required")
	}

	switch cfg.Backend {
	case BackendOllama:
		e := NewOllamaEmbedder(cfg.BaseURL, cfg.Model)
		e.strict = cfg.Strict
		return e, nil
	case BackendOpenAI:
		e := NewOpenAIEmbedder(cfg.BaseURL, cfg.APIKey, cfg.Model)
		e.strict = cfg.Strict
		return e, nil
	default:
		return nil, fmt.Errorf("embedding: unknown backend %q", cfg.Backend)
	}
}
