package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// sleepyEmbedServer serves both the Ollama and OpenAI embed shapes after a
// delay, so a test can drive a request past its deadline.
func sleepyEmbedServer(t *testing.T, delay time.Duration) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
		if r.URL.Path == "/api/embed" {
			_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{Embeddings: [][]float32{{1}}})
			return
		}
		var resp openAIEmbedResponse
		resp.Data = append(resp.Data, struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{Embedding: []float32{1}, Index: 0})
		_ = json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestResolveTimeouts(t *testing.T) {
	tests := []struct {
		name     string
		base     time.Duration
		perInput time.Duration
		inputs   int
		want     time.Duration
	}{
		{"zero uses defaults", 0, 0, 1, DefaultTimeout + DefaultPerInputTimeout},
		{"defaults scale with batch", 0, 0, 25, DefaultTimeout + 25*DefaultPerInputTimeout},
		{"explicit values", 10 * time.Second, time.Second, 5, 15 * time.Second},
		{"negative per-input is flat", 10 * time.Second, NoTimeout, 25, 10 * time.Second},
		{"empty batch gets the base", 10 * time.Second, time.Second, 0, 10 * time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveTimeouts(tt.base, tt.perInput).budget(tt.inputs)
			if got != tt.want {
				t.Errorf("budget(%d) = %v, want %v", tt.inputs, got, tt.want)
			}
		})
	}
}

func TestResolveTimeouts_NoTimeoutDisables(t *testing.T) {
	ts := resolveTimeouts(NoTimeout, time.Second)
	if !ts.disabled() {
		t.Fatal("NoTimeout base should disable the deadline")
	}
	ctx, cancel := ts.apply(context.Background(), 10)
	defer cancel()
	if _, ok := ctx.Deadline(); ok {
		t.Error("apply set a deadline despite NoTimeout")
	}
}

// The deadline must arrive as a context cancellation, so callers can classify
// it with errors.Is rather than string-matching a *url.Error.
func TestEmbed_TimeoutIsDeadlineExceeded(t *testing.T) {
	t.Parallel()
	for _, backend := range []Backend{BackendOllama, BackendOpenAI} {
		t.Run(string(backend), func(t *testing.T) {
			srv := sleepyEmbedServer(t, 500*time.Millisecond)
			e, err := New(Config{
				Backend: backend, BaseURL: srv.URL, Model: "test",
				Timeout: 30 * time.Millisecond, PerInputTimeout: NoTimeout,
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			_, err = e.Embed(context.Background(), []string{"hello"})
			if err == nil {
				t.Fatal("want a timeout error, got nil")
			}
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("error %v does not match context.DeadlineExceeded", err)
			}
			// A timeout is transient; a caller must not quarantine the input.
			if !IsRetryable(err) {
				t.Errorf("IsRetryable(%v) = false, want true", err)
			}
		})
	}
}

func TestRerank_TimeoutIsDeadlineExceeded(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(500 * time.Millisecond):
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()

	rr, err := NewReranker(RerankConfig{
		Backend: RerankBackendJina, BaseURL: srv.URL, Model: "test",
		Timeout: 30 * time.Millisecond, PerInputTimeout: NoTimeout,
	})
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	_, err = rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{"a"}})
	if err == nil {
		t.Fatal("want a timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error %v does not match context.DeadlineExceeded", err)
	}
}

// The per-input budget must actually reach the request, not just the
// resolver: a batch that would blow a flat deadline succeeds when the budget
// scales with its size.
func TestEmbed_PerInputBudgetScalesWithBatch(t *testing.T) {
	t.Parallel()
	srv := sleepyEmbedServer(t, 150*time.Millisecond)
	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "test",
		Timeout: 20 * time.Millisecond, PerInputTimeout: 200 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// 3 inputs -> 20ms + 600ms of budget against a 150ms server.
	if _, err := e.Embed(context.Background(), []string{"a", "b", "c"}); err != nil {
		t.Fatalf("Embed with a scaled budget: %v", err)
	}
}

// A caller's own deadline is shorter than the embedder's, and must still win.
func TestEmbed_CallerDeadlineWins(t *testing.T) {
	t.Parallel()
	srv := sleepyEmbedServer(t, 500*time.Millisecond)
	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "test",
		Timeout: time.Minute,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	start := time.Now()
	if _, err := e.Embed(ctx, []string{"a"}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error %v does not match context.DeadlineExceeded", err)
	}
	if elapsed := time.Since(start); elapsed > 300*time.Millisecond {
		t.Errorf("took %v; the embedder's minute-long budget overrode the caller's 30ms", elapsed)
	}
}

// Cancellation must still abort promptly rather than waiting out the budget.
func TestEmbed_CallerCancellationAbortsPromptly(t *testing.T) {
	t.Parallel()
	srv := sleepyEmbedServer(t, 500*time.Millisecond)
	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "test",
		Timeout: time.Minute,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	if _, err := e.Embed(ctx, []string{"a"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("error %v does not match context.Canceled", err)
	}
	if elapsed := time.Since(start); elapsed > 300*time.Millisecond {
		t.Errorf("took %v; cancellation did not abort the request", elapsed)
	}
}

// NoTimeout restores the old unbounded behaviour for callers that want it.
func TestEmbed_NoTimeoutDoesNotExpire(t *testing.T) {
	t.Parallel()
	srv := sleepyEmbedServer(t, 100*time.Millisecond)
	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "test",
		Timeout: NoTimeout,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := e.Embed(context.Background(), []string{"a"}); err != nil {
		t.Fatalf("Embed with NoTimeout: %v", err)
	}
}

// The default is on: a caller that sets nothing still gets a deadline, which
// is the whole point of the change.
func TestNew_DefaultsToABoundedRequest(t *testing.T) {
	srv := sleepyEmbedServer(t, time.Millisecond)
	e, err := New(Config{Backend: BackendOllama, BaseURL: srv.URL, Model: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	oe, ok := e.(*OllamaEmbedder)
	if !ok {
		t.Fatalf("New returned %T, want *OllamaEmbedder", e)
	}
	if oe.timeouts.disabled() {
		t.Error("a default Config produced an unbounded embedder")
	}
	if got, want := oe.timeouts.budget(1), DefaultTimeout+DefaultPerInputTimeout; got != want {
		t.Errorf("default budget for 1 input = %v, want %v", got, want)
	}
}

func TestConfigFromEnvPrefix_Timeouts(t *testing.T) {
	tests := []struct {
		name         string
		timeout      string
		perInput     string
		wantTimeout  time.Duration
		wantPerInput time.Duration
		wantErr      bool
	}{
		{name: "unset leaves zero values"},
		{name: "durations parse", timeout: "45s", perInput: "500ms", wantTimeout: 45 * time.Second, wantPerInput: 500 * time.Millisecond},
		{name: "none disables", timeout: "none", wantTimeout: NoTimeout},
		{name: "none disables per-input scaling", perInput: "none", wantPerInput: NoTimeout},
		{name: "garbage errors", timeout: "soon", wantErr: true},
		{name: "bare number errors", perInput: "30", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.timeout != "" {
				t.Setenv("TESTEMBED_TIMEOUT", tt.timeout)
			}
			if tt.perInput != "" {
				t.Setenv("TESTEMBED_PER_INPUT_TIMEOUT", tt.perInput)
			}

			cfg, err := ConfigFromEnvPrefix("TESTEMBED")
			if tt.wantErr {
				if err == nil {
					t.Fatal("want a parse error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("ConfigFromEnvPrefix: %v", err)
			}
			if cfg.Timeout != tt.wantTimeout {
				t.Errorf("Timeout = %v, want %v", cfg.Timeout, tt.wantTimeout)
			}
			if cfg.PerInputTimeout != tt.wantPerInput {
				t.Errorf("PerInputTimeout = %v, want %v", cfg.PerInputTimeout, tt.wantPerInput)
			}
		})
	}
}

func TestRerankConfigFromEnvPrefix_Timeouts(t *testing.T) {
	t.Setenv("TESTRERANK_TIMEOUT", "45s")
	t.Setenv("TESTRERANK_PER_INPUT_TIMEOUT", "none")

	cfg, err := RerankConfigFromEnvPrefix("TESTRERANK")
	if err != nil {
		t.Fatalf("RerankConfigFromEnvPrefix: %v", err)
	}
	if cfg.Timeout != 45*time.Second {
		t.Errorf("Timeout = %v, want 45s", cfg.Timeout)
	}
	if cfg.PerInputTimeout != NoTimeout {
		t.Errorf("PerInputTimeout = %v, want NoTimeout", cfg.PerInputTimeout)
	}
}
