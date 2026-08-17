package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// A budget of zero is a misconfiguration, not a request for the registry
// default. Accepting it silently is the same class of problem clip detection
// exists to remove: the operator believes a budget is in force, no budget is
// in force, and nothing says so. Negative values are already rejected; zero
// reads identically to "unset" once it reaches Config.
func TestConfigFromEnvPrefix_LimitsRejectZero(t *testing.T) {
	for _, key := range []string{"TESTLIMZ_MAX_TOKENS", "TESTLIMZ_MAX_BYTES"} {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, "0")
			if _, err := ConfigFromEnvPrefix("TESTLIMZ"); err == nil {
				t.Fatalf("%s=0: want an error, got nil", key)
			}
		})
	}
}

// An explicitly empty value still means "unset", and must not be confused
// with an explicit zero.
func TestConfigFromEnvPrefix_LimitsAllowEmpty(t *testing.T) {
	t.Setenv("TESTLIME_MAX_TOKENS", "")
	t.Setenv("TESTLIME_MAX_BYTES", "")
	cfg, err := ConfigFromEnvPrefix("TESTLIME")
	if err != nil {
		t.Fatalf("ConfigFromEnvPrefix: %v", err)
	}
	if cfg.MaxTokens != 0 || cfg.MaxBytes != 0 {
		t.Fatalf("got {MaxTokens:%d MaxBytes:%d}, want both 0 so the registry applies",
			cfg.MaxTokens, cfg.MaxBytes)
	}
}

// proportionalTokenServer reports a token count derived from the input size
// and clamps it at window, the way Ollama does.
//
// The distinction from a fixed-count server matters: with a constant count,
// shrinking an input never changes the verdict, so a strict-mode clip error
// that wrongly routes into the adaptive shrink path still ends up failing
// (at the shrink floor) and the test passes anyway. Only a size-sensitive
// backend exposes the difference.
func proportionalTokenServer(t *testing.T, window int, bytesPerToken int, sawBytes *int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		total := 0
		for _, in := range req.Input {
			total += len(in)
		}
		if sawBytes != nil {
			*sawBytes = total
		}
		tokens := total / bytesPerToken
		if tokens > window {
			tokens = window // clamp, like Ollama
		}
		vecs := make([][]float32, len(req.Input))
		for i := range vecs {
			vecs[i] = []float32{0.1, 0.2}
		}
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings:      vecs,
			PromptEvalCount: tokens,
		})
	}))
}

// Regression guard: a clip verdict must not be reported as TooLong.
//
// TooLong drives embedShrinking, which truncates the input ~20% and retries
// until the backend stops complaining. Applied to a clip verdict in strict
// mode that inverts the contract: the input gets shrunk until the clamped
// token count falls under the budget, and Embed returns a vector computed
// from deliberately truncated text -- the exact outcome strict mode exists to
// prevent. Strict callers opted out of being shrunk.
//
// The input here is deliberately under the 6000-byte budget so applyLimits
// passes it through, but dense enough (~2 bytes/token) to overrun the
// 2000-token budget. That band -- fits the byte budget, overruns the context
// -- is where real documents land, and it is the only band where this bug is
// reachable.
func TestEmbed_StrictClipIsNotAdaptivelyShrunk(t *testing.T) {
	var sawBytes int
	srv := proportionalTokenServer(t, 2000, 2, &sawBytes)

	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text",
		Strict: true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	long := make([]byte, 5000)
	for i := range long {
		long[i] = 'a'
	}

	if _, err := e.Embed(context.Background(), []string{string(long)}); err == nil {
		t.Fatalf("strict mode returned a vector for a clipped input; "+
			"the backend last saw %d bytes of the original 5000, meaning the "+
			"input was shrunk and retried until it fit", sawBytes)
	}
	if sawBytes != 5000 {
		t.Errorf("backend last saw %d bytes, want 5000: a clip verdict must fail "+
			"outright, not be shrunk and retried", sawBytes)
	}
}

// The same input under the default (non-strict) mode should succeed, report
// the clip, and still not be shrunk -- the caller gets a vector plus the fact
// that it came from clipped text, which is the whole point of the Usage hook.
func TestEmbed_NonStrictClipIsReportedNotShrunk(t *testing.T) {
	var sawBytes int
	srv := proportionalTokenServer(t, 2000, 2, &sawBytes)

	var got Usage
	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text",
		OnUsage: UsageFunc(func(u Usage) { got = u }),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	long := make([]byte, 5000)
	for i := range long {
		long[i] = 'a'
	}

	if _, err := e.Embed(context.Background(), []string{string(long)}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !got.Clipped {
		t.Error("Clipped = false; the backend clamped at the budget")
	}
	if sawBytes != 5000 {
		t.Errorf("backend last saw %d bytes, want 5000: a reported clip must not "+
			"trigger a shrink-and-retry", sawBytes)
	}
}
