package embedding

import (
	"fmt"
	"time"
)

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

	// Timeout is the base deadline applied to each HTTP request, conveyed as
	// a context deadline so failures arrive as context.DeadlineExceeded and
	// compose with the caller's own cancellation. Zero uses DefaultTimeout;
	// NoTimeout leaves requests unbounded.
	Timeout time.Duration

	// PerInputTimeout is added to Timeout for each input in a request, so a
	// batch gets a budget proportional to its size. Zero uses
	// DefaultPerInputTimeout; NoTimeout disables the scaling and gives every
	// request the flat base budget.
	PerInputTimeout time.Duration

	// MaxBytes and MaxTokens override the registered per-model budget (see
	// limits.go). Zero means "use whatever is registered". Setting MaxTokens
	// for an unregistered model derives a byte budget from it, so a new model
	// is enforceable from config alone without a library edit.
	MaxBytes  int
	MaxTokens int

	// Tokenizer, when set, makes token budgets exact: input is truncated to
	// MaxTokens rather than to a byte budget standing in for it. Without one,
	// every token budget is converted through a ratio that is wrong by
	// 50-70% on real text. See TokenCounter.
	Tokenizer TokenCounter

	// BytesPerToken is the bytes-per-token ratio to assume when converting a
	// token budget to a byte budget, for callers with no tokenizer. Zero uses
	// a conservative built-in guess. Derive it from CalibrationFor over your
	// own corpus; the right value differs per corpus as well as per model,
	// which is why the library does not learn it for you.
	//
	// Ignored when Tokenizer is set, since nothing then needs estimating.
	BytesPerToken float64

	// OnUsage, when set, receives a Usage after each request the backend
	// reported a token count for. It is how a caller measures its real clip
	// rate; without it, a backend that silently truncates over-length input
	// stays invisible. Called synchronously on the embedding path, so keep it
	// cheap and non-blocking. Wrap a plain function with UsageFunc.
	OnUsage UsageReporter
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
		e.timeouts = resolveTimeouts(cfg.Timeout, cfg.PerInputTimeout)
		e.limits = cfg.Limits()
		e.onUsage = cfg.OnUsage
		e.tokenizer = cfg.Tokenizer
		return e, nil
	case BackendOpenAI:
		e := NewOpenAIEmbedder(cfg.BaseURL, cfg.APIKey, cfg.Model)
		e.strict = cfg.Strict
		e.timeouts = resolveTimeouts(cfg.Timeout, cfg.PerInputTimeout)
		e.limits = cfg.Limits()
		e.onUsage = cfg.OnUsage
		e.tokenizer = cfg.Tokenizer
		return e, nil
	default:
		return nil, fmt.Errorf("embedding: unknown backend %q", cfg.Backend)
	}
}
