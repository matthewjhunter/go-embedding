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
type Config struct {
	Backend Backend
	BaseURL string
	APIKey  string
	Model   string
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
		return NewOllamaEmbedder(cfg.BaseURL, cfg.Model), nil
	case BackendOpenAI:
		return NewOpenAIEmbedder(cfg.BaseURL, cfg.APIKey, cfg.Model), nil
	default:
		return nil, fmt.Errorf("embedding: unknown backend %q", cfg.Backend)
	}
}
