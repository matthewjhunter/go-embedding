package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

// tokenReportingServer serves both backend shapes, reporting tokens as the
// backend's own count of the prompt it processed.
func tokenReportingServer(t *testing.T, tokens int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/embed" {
			_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{
				Embeddings:      [][]float32{{1}},
				PromptEvalCount: tokens,
			})
			return
		}
		var resp openAIEmbedResponse
		resp.Data = append(resp.Data, struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{Embedding: []float32{1}, Index: 0})
		resp.Usage.PromptTokens = tokens
		_ = json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestCheckUsage_ClipDetection(t *testing.T) {
	tests := []struct {
		name        string
		limits      Limits
		texts       []string
		tokens      int
		wantUsage   bool
		wantClipped bool
	}{
		{
			name:   "no token budget means no verdict",
			limits: Limits{MaxBytes: 6000},
			texts:  []string{"a"}, tokens: 9999,
			wantUsage: true, wantClipped: false,
		},
		{
			name:   "backend reported nothing",
			limits: Limits{MaxTokens: 2000},
			texts:  []string{"a"}, tokens: 0,
			wantUsage: false,
		},
		{
			name:   "comfortably under the budget",
			limits: Limits{MaxTokens: 2000},
			texts:  []string{"a"}, tokens: 500,
			wantUsage: true, wantClipped: false,
		},
		{
			name:   "at the budget is the fingerprint of a clip",
			limits: Limits{MaxTokens: 2000},
			texts:  []string{"a"}, tokens: 2000,
			wantUsage: true, wantClipped: true,
		},
		{
			name:   "over the budget",
			limits: Limits{MaxTokens: 2000},
			texts:  []string{"a"}, tokens: 2048,
			wantUsage: true, wantClipped: true,
		},
		{
			// A batch reports one total for every input, so an over-budget
			// total is the expected shape, not evidence any single input was
			// clipped. Claiming otherwise would fire on every healthy batch.
			name:   "batch totals are not attributable",
			limits: Limits{MaxTokens: 2000},
			texts:  []string{"a", "b"}, tokens: 4000,
			wantUsage: true, wantClipped: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got *Usage
			err := checkUsage("m", tt.limits, tt.texts, tt.tokens, false, UsageFunc(func(u Usage) { got = &u }))
			if err != nil {
				t.Fatalf("checkUsage: %v", err)
			}
			if !tt.wantUsage {
				if got != nil {
					t.Fatalf("reported usage %+v for an unreported token count", *got)
				}
				return
			}
			if got == nil {
				t.Fatal("no usage reported")
			}
			if got.Clipped != tt.wantClipped {
				t.Errorf("Clipped = %v, want %v", got.Clipped, tt.wantClipped)
			}
			if got.Tokens != tt.tokens {
				t.Errorf("Tokens = %d, want %d", got.Tokens, tt.tokens)
			}
			if got.Inputs != len(tt.texts) {
				t.Errorf("Inputs = %d, want %d", got.Inputs, len(tt.texts))
			}
		})
	}
}

// Strict mode already refuses to truncate before sending; it must equally
// refuse a vector the backend computed from clipped text.
func TestCheckUsage_StrictRejectsAClippedResult(t *testing.T) {
	err := checkUsage("m", Limits{MaxTokens: 2000}, []string{"a"}, 2048, true, nil)
	if err == nil {
		t.Fatal("want an error in strict mode, got nil")
	}
	var pe *PermanentError
	if !errors.As(err, &pe) {
		t.Fatalf("error %v is not a *PermanentError", err)
	}
	// Not TooLong: strict opts out of shrink-and-retry, mirroring applyLimits.
	if pe.TooLong {
		t.Error("strict clip error is TooLong, which would trigger adaptive shrinking")
	}
	if IsRetryable(err) {
		t.Error("a clipped input is not retryable as-is")
	}
}

// The token count has to survive the round trip from each backend's own
// response shape, or nothing above it can work.
func TestEmbed_ReportsBackendTokenCount(t *testing.T) {
	for _, backend := range []Backend{BackendOllama, BackendOpenAI} {
		t.Run(string(backend), func(t *testing.T) {
			srv := tokenReportingServer(t, 2048)

			var got Usage
			e, err := New(Config{
				Backend: backend, BaseURL: srv.URL, Model: "nomic-embed-text",
				OnUsage: UsageFunc(func(u Usage) { got = u }),
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if _, err := e.Embed(context.Background(), []string{"hello"}); err != nil {
				t.Fatalf("Embed: %v", err)
			}

			if got.Tokens != 2048 {
				t.Errorf("Tokens = %d, want 2048", got.Tokens)
			}
			if !got.Clipped {
				t.Error("Clipped = false; 2048 tokens against a 2000-token budget is a clip")
			}
			if got.Bytes != len("hello") {
				t.Errorf("Bytes = %d, want %d", got.Bytes, len("hello"))
			}
			if got.Model != "nomic-embed-text" {
				t.Errorf("Model = %q, want nomic-embed-text", got.Model)
			}
		})
	}
}

func TestEmbed_StrictFailsOnAClippedResult(t *testing.T) {
	srv := tokenReportingServer(t, 2048)
	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text",
		Strict: true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := e.Embed(context.Background(), []string{"hello"}); err == nil {
		t.Fatal("want an error for a clipped result in strict mode, got nil")
	}
}

func TestEffectiveLimits(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		override Limits
		want     Limits
	}{
		{
			name: "registry supplies both", model: "nomic-embed-text",
			want: Limits{MaxBytes: 6000, MaxTokens: 2000},
		},
		{
			name: "config overrides the registry", model: "nomic-embed-text",
			override: Limits{MaxBytes: 4000, MaxTokens: 1500},
			want:     Limits{MaxBytes: 4000, MaxTokens: 1500},
		},
		{
			name: "partial override keeps the other field", model: "nomic-embed-text",
			override: Limits{MaxTokens: 1500},
			want:     Limits{MaxBytes: 6000, MaxTokens: 1500},
		},
		{
			// The point of the override: a model the registry has never heard
			// of becomes enforceable from config alone, with no library edit.
			name: "unknown model derives bytes from tokens", model: "brand-new-model",
			override: Limits{MaxTokens: 1000},
			want:     Limits{MaxBytes: 1000 * conservativeBytesPerToken, MaxTokens: 1000},
		},
		{
			name: "unknown model with no override is unenforced", model: "brand-new-model",
			want: Limits{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := effectiveLimits(tt.model, tt.override); got != tt.want {
				t.Errorf("effectiveLimits = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// A configured budget has to reach the pre-flight byte check, not just the
// resolver.
func TestEmbed_ConfiguredMaxBytesTruncates(t *testing.T) {
	var sent string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		sent = req.Input[0]
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{Embeddings: [][]float32{{1}}})
	}))
	defer srv.Close()

	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text",
		MaxBytes: 10,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := e.Embed(context.Background(), []string{"aaaaaaaaaaaaaaaaaaaaaaaaaaa"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(sent) != 10 {
		t.Errorf("sent %d bytes, want 10 -- Config.MaxBytes did not reach applyLimits", len(sent))
	}
}

func TestConfigFromEnvPrefix_Limits(t *testing.T) {
	t.Setenv("TESTLIM_MAX_TOKENS", "1500")
	t.Setenv("TESTLIM_MAX_BYTES", "4000")

	cfg, err := ConfigFromEnvPrefix("TESTLIM")
	if err != nil {
		t.Fatalf("ConfigFromEnvPrefix: %v", err)
	}
	if cfg.MaxTokens != 1500 {
		t.Errorf("MaxTokens = %d, want 1500", cfg.MaxTokens)
	}
	if cfg.MaxBytes != 4000 {
		t.Errorf("MaxBytes = %d, want 4000", cfg.MaxBytes)
	}
}

func TestConfigFromEnvPrefix_LimitsRejectGarbage(t *testing.T) {
	for _, key := range []string{"TESTLIM2_MAX_TOKENS", "TESTLIM2_MAX_BYTES"} {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, "lots")
			if _, err := ConfigFromEnvPrefix("TESTLIM2"); err == nil {
				t.Fatal("want a parse error, got nil")
			}
		})
	}
}

func TestConfigFromEnvPrefix_LimitsRejectNegative(t *testing.T) {
	t.Setenv("TESTLIM3_MAX_TOKENS", "-1")
	if _, err := ConfigFromEnvPrefix("TESTLIM3"); err == nil {
		t.Fatal("want an error for a negative budget, got nil")
	}
}
