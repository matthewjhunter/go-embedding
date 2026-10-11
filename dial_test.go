package embedding

import (
	"context"
	"errors"
	"net"
	"net/http"
	"testing"
	"time"
)

// blackholeURL is a TEST-NET-1 address (RFC 5737): nothing answers, so a
// connect attempt hangs until something gives up on it. It stands in for a
// backend behind a firewall that drops rather than rejects.
const blackholeURL = "http://192.0.2.1:9"

func TestResolveDialTimeout(t *testing.T) {
	tests := []struct {
		name string
		in   time.Duration
		want time.Duration
	}{
		{"zero uses the default", 0, DefaultDialTimeout},
		{"explicit value", 3 * time.Second, 3 * time.Second},
		{"NoTimeout disables", NoTimeout, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveDialTimeout(tt.in); got != tt.want {
				t.Errorf("resolveDialTimeout(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

// The dial timeout must bound the connect on its own, with no help from a
// request deadline: a blackholed backend fails in the dial budget, not the
// inference budget.
func TestEmbed_UnreachableBackendFailsWithinDialTimeout(t *testing.T) {
	t.Parallel()
	for _, backend := range []Backend{BackendOllama, BackendOpenAI} {
		t.Run(string(backend), func(t *testing.T) {
			t.Parallel()
			e, err := New(Config{
				Backend: backend, BaseURL: blackholeURL, Model: "test",
				DialTimeout: 50 * time.Millisecond,
				Timeout:     time.Minute, PerInputTimeout: NoTimeout,
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			start := time.Now()
			_, err = e.Embed(context.Background(), []string{"hello"})
			elapsed := time.Since(start)
			if err == nil {
				t.Fatal("want a dial error, got nil")
			}
			if elapsed > 2*time.Second {
				t.Errorf("took %v; the dial timeout did not bound the connect", elapsed)
			}
			if !isDialError(err) {
				t.Errorf("error %v is not a dial error", err)
			}
			// Unreachable is transient: the backend may come back.
			if !IsRetryable(err) {
				t.Errorf("IsRetryable(%v) = false, want true", err)
			}
		})
	}
}

func TestRerank_UnreachableBackendFailsWithinDialTimeout(t *testing.T) {
	t.Parallel()
	rr, err := NewReranker(RerankConfig{
		Backend: RerankBackendJina, BaseURL: blackholeURL, Model: "test",
		DialTimeout: 50 * time.Millisecond,
		Timeout:     time.Minute, PerInputTimeout: NoTimeout,
	})
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	start := time.Now()
	_, err = rr.Rerank(context.Background(), RerankRequest{Query: "q", Documents: []string{"a"}})
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("want a dial error, got nil")
	}
	if elapsed > 2*time.Second {
		t.Errorf("took %v; the dial timeout did not bound the connect", elapsed)
	}
	if IsRerankAvailable(err) {
		t.Errorf("IsRerankAvailable(%v) = true; an unreachable backend must degrade", err)
	}
}

// A caller-supplied client is used verbatim, so proxies, TLS settings and
// instrumentation stay the caller's business.
func TestNew_HTTPClientOverride(t *testing.T) {
	custom := &http.Client{}
	for _, backend := range []Backend{BackendOllama, BackendOpenAI} {
		t.Run(string(backend), func(t *testing.T) {
			e, err := New(Config{
				Backend: backend, BaseURL: "http://localhost:1", Model: "test",
				HTTPClient: custom,
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			var got *http.Client
			switch e := e.(type) {
			case *OllamaEmbedder:
				got = e.client
			case *OpenAIEmbedder:
				got = e.client
			default:
				t.Fatalf("New returned %T", e)
			}
			if got != custom {
				t.Error("New did not use Config.HTTPClient")
			}
		})
	}

	rr, err := NewReranker(RerankConfig{
		Backend: RerankBackendJina, BaseURL: "http://localhost:1", Model: "test",
		HTTPClient: custom,
	})
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if rr.(*JinaReranker).client != custom {
		t.Error("NewReranker did not use RerankConfig.HTTPClient")
	}
}

// The default client is the fix, so it must be the one a zero Config gets:
// a bare http.Client{} would silently reintroduce the 30s connect hang.
func TestNew_DefaultClientHasDialTimeout(t *testing.T) {
	e, err := New(Config{Backend: BackendOllama, BaseURL: "http://localhost:1", Model: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c := e.(*OllamaEmbedder).client
	if c.Transport == nil {
		t.Fatal("default client has no Transport; the dial timeout lives there")
	}
	if _, ok := c.Transport.(*http.Transport); !ok {
		t.Fatalf("default Transport is %T, want *http.Transport", c.Transport)
	}
}

func TestIsDialError(t *testing.T) {
	dial := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("i/o timeout")}
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"dial", dial, true},
		{"wrapped dial", errors.Join(errors.New("ollama embed"), dial), true},
		{"read on an open connection", &net.OpError{Op: "read", Net: "tcp", Err: errors.New("reset")}, false},
		{"deadline", context.DeadlineExceeded, false},
		{"permanent", &PermanentError{Err: errors.New("400")}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isDialError(tt.err); got != tt.want {
				t.Errorf("isDialError(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestConfigFromEnvPrefix_DialTimeout(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		want    time.Duration
		wantErr bool
	}{
		{name: "unset leaves zero"},
		{name: "duration parses", value: "2s", want: 2 * time.Second},
		{name: "none disables", value: "none", want: NoTimeout},
		{name: "bare number errors", value: "5", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value != "" {
				t.Setenv("TESTEMBED_DIAL_TIMEOUT", tt.value)
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
			if cfg.DialTimeout != tt.want {
				t.Errorf("DialTimeout = %v, want %v", cfg.DialTimeout, tt.want)
			}
		})
	}
}

func TestRerankConfigFromEnvPrefix_DialTimeout(t *testing.T) {
	t.Setenv("TESTRERANK_DIAL_TIMEOUT", "2s")
	cfg, err := RerankConfigFromEnvPrefix("TESTRERANK")
	if err != nil {
		t.Fatalf("RerankConfigFromEnvPrefix: %v", err)
	}
	if cfg.DialTimeout != 2*time.Second {
		t.Errorf("DialTimeout = %v, want 2s", cfg.DialTimeout)
	}
}
