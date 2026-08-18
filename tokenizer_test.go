package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// wordTokenizer counts whitespace-separated words. It is monotonic over
// prefixes, which is the only property the truncation search relies on.
type wordTokenizer struct{ calls int }

func (w *wordTokenizer) CountTokens(text string) int {
	w.calls++
	return len(strings.Fields(text))
}

func TestTokenCountFunc(t *testing.T) {
	var tc TokenCounter = TokenCountFunc(func(s string) int { return len(s) })
	if got := tc.CountTokens("abcd"); got != 4 {
		t.Errorf("CountTokens = %d, want 4", got)
	}
}

// With a tokenizer, the token budget is enforced exactly before the request
// goes out, instead of a byte budget standing in for it.
func TestApplyLimits_ExactTokenTruncation(t *testing.T) {
	limits := Limits{MaxBytes: 100_000, MaxTokens: 10}
	text := strings.Repeat("word ", 50) // 50 tokens, well under the byte budget

	got, err := applyLimits([]string{text}, limits, "m", false, &wordTokenizer{})
	if err != nil {
		t.Fatalf("applyLimits: %v", err)
	}
	if n := len(strings.Fields(got[0])); n != 10 {
		t.Errorf("truncated to %d tokens, want 10", n)
	}
}

func TestApplyLimits_TokenizerLeavesShortInputAlone(t *testing.T) {
	limits := Limits{MaxBytes: 100_000, MaxTokens: 100}
	text := "just a few words here"

	tok := &wordTokenizer{}
	got, err := applyLimits([]string{text}, limits, "m", false, tok)
	if err != nil {
		t.Fatalf("applyLimits: %v", err)
	}
	if got[0] != text {
		t.Errorf("text was modified: %q", got[0])
	}
	// One count to establish it fits; no search.
	if tok.calls != 1 {
		t.Errorf("made %d CountTokens calls for an input that fits, want 1", tok.calls)
	}
}

func TestApplyLimits_StrictRejectsOverTokenBudget(t *testing.T) {
	limits := Limits{MaxBytes: 100_000, MaxTokens: 10}
	text := strings.Repeat("word ", 50)

	_, err := applyLimits([]string{text}, limits, "m", true, &wordTokenizer{})
	if err == nil {
		t.Fatal("want an error in strict mode, got nil")
	}
	var pe *PermanentError
	if !errors.As(err, &pe) {
		t.Errorf("error %v is not a *PermanentError", err)
	}
}

// Without a tokenizer nothing changes: the byte budget is still the gate.
func TestApplyLimits_WithoutTokenizerUsesBytes(t *testing.T) {
	limits := Limits{MaxBytes: 20, MaxTokens: 1000}
	text := strings.Repeat("a", 100)

	got, err := applyLimits([]string{text}, limits, "m", false, nil)
	if err != nil {
		t.Fatalf("applyLimits: %v", err)
	}
	if len(got[0]) != 20 {
		t.Errorf("got %d bytes, want 20", len(got[0]))
	}
}

// A token budget with no tokenizer to enforce it must not silently pass
// oversize text through -- the byte budget still applies.
func TestApplyLimits_TokenBudgetWithoutTokenizerStillClipsBytes(t *testing.T) {
	limits := Limits{MaxBytes: 30, MaxTokens: 5}
	text := strings.Repeat("word ", 50)

	got, err := applyLimits([]string{text}, limits, "m", false, nil)
	if err != nil {
		t.Fatalf("applyLimits: %v", err)
	}
	if len(got[0]) > 30 {
		t.Errorf("got %d bytes, want at most 30", len(got[0]))
	}
}

func TestEmbed_TokenizerEnforcesBudgetEndToEnd(t *testing.T) {
	var sent string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		sent = req.Input[0]
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{Embeddings: [][]float32{{1}}})
	}))
	defer srv.Close()

	e, err := New(Config{
		Backend: BackendOllama, BaseURL: srv.URL, Model: "m",
		MaxTokens: 8, Tokenizer: &wordTokenizer{},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := e.Embed(context.Background(), []string{strings.Repeat("word ", 100)}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if n := len(strings.Fields(sent)); n != 8 {
		t.Errorf("backend received %d tokens, want 8", n)
	}
}

func TestEffectiveLimits_CallerRatio(t *testing.T) {
	// An unregistered model with only a token budget derives bytes from the
	// caller's measured ratio rather than the built-in guess.
	got := effectiveLimits("brand-new-model", Limits{MaxTokens: 1000}, 1.75)
	if got.MaxBytes != 1750 {
		t.Errorf("MaxBytes = %d, want 1750", got.MaxBytes)
	}

	fallback := effectiveLimits("brand-new-model", Limits{MaxTokens: 1000}, 0)
	if fallback.MaxBytes != 1000*conservativeBytesPerToken {
		t.Errorf("MaxBytes = %d, want the static ratio", fallback.MaxBytes)
	}
}

func TestConfig_LimitsUsesBytesPerToken(t *testing.T) {
	cfg := Config{Model: "brand-new-model", MaxTokens: 500, BytesPerToken: 3.5}
	if got := cfg.Limits().MaxBytes; got != 1750 {
		t.Errorf("MaxBytes = %d, want 1750", got)
	}
}

// Split can size chunks in tokens once it has something that counts them,
// which is the number the caller actually cares about.
func TestSplit_MaxTokensWithTokenizer(t *testing.T) {
	text := strings.Repeat("word ", 500)
	got := Split("m", text, SplitOptions{MaxTokens: 20, Tokenizer: &wordTokenizer{}})

	if len(got) < 20 {
		t.Fatalf("got %d chunks for 500 tokens at 20 per chunk", len(got))
	}
	for _, c := range got {
		if n := len(strings.Fields(c.Text)); n > 20 {
			t.Errorf("chunk %d holds %d tokens, over the 20 budget", c.Ordinal, n)
		}
	}
	if want, have := stripSpace(text), stripSpace(concat(got)); want != have {
		t.Error("token-sized splitting changed the content")
	}
}

// Without a tokenizer, a token target is converted through the ratio rather
// than ignored.
func TestSplit_MaxTokensWithoutTokenizerUsesRatio(t *testing.T) {
	text := strings.Repeat("word ", 500)

	got := Split("m", text, SplitOptions{MaxTokens: 20, BytesPerToken: 5})
	checkInvariants(t, text, got, 100) // 20 tokens * 5 bytes
	if len(got) < 2 {
		t.Fatal("a token target with no tokenizer did not split at all")
	}
}

func TestSplit_TokenBudgetRespectsExplicitByteCeiling(t *testing.T) {
	text := strings.Repeat("word ", 500)
	got := Split("m", text, SplitOptions{
		MaxTokens: 1000, MaxBytes: 60, Tokenizer: &wordTokenizer{},
	})
	checkInvariants(t, text, got, 60)
	if len(got) < 2 {
		t.Fatal("the byte ceiling was ignored")
	}
}

// Chunks must still land on real boundaries when sized by tokens.
func TestSplit_MaxTokensStillPrefersBoundaries(t *testing.T) {
	text := strings.Repeat("This is a sentence of moderate length. ", 30)
	got := Split("m", text, SplitOptions{MaxTokens: 16, Tokenizer: &wordTokenizer{}})

	ends := 0
	for _, c := range got {
		if strings.HasSuffix(c.Text, ".") {
			ends++
		}
	}
	if ends < len(got)/2 {
		t.Errorf("only %d of %d chunks end at a sentence boundary", ends, len(got))
	}
}

// FuzzSplitTokens drives the token-governed path. The binary search and the
// token window are new places for the splitter to stall or overshoot, and the
// byte-path fuzzer does not reach them.
func FuzzSplitTokens(f *testing.F) {
	f.Add("one two three four five six", 2, 0, 0)
	f.Add(strings.Repeat("para\n\ngraph words ", 20), 5, 2, 1)
	f.Add("日本語 のテキスト です", 1, 0, 0)
	f.Add("", 3, 0, 0)
	f.Add("single-unbroken-token-with-no-spaces-at-all", 1, 0, 0)

	f.Fuzz(func(t *testing.T, text string, maxTokens, overlap, minBytes int) {
		if maxTokens < 0 || maxTokens > 1<<12 || overlap < 0 || overlap > 1<<16 || minBytes < 0 || minBytes > 1<<16 {
			t.Skip()
		}
		tok := &wordTokenizer{}
		got := Split("m", text, SplitOptions{
			MaxTokens: maxTokens, Overlap: overlap, MinBytes: minBytes, Tokenizer: tok,
		})
		// 0 skips the byte-size check: this path is governed by tokens.
		checkInvariants(t, text, got, 0)

		if maxTokens == 0 {
			return
		}
		for _, c := range got {
			// A single token longer than the budget cannot be split further,
			// so one-token chunks are allowed to exceed it.
			if n := tok.CountTokens(c.Text); n > maxTokens && n > 1 {
				t.Errorf("chunk %d holds %d tokens, over the %d budget: %q",
					c.Ordinal, n, maxTokens, c.Text)
			}
		}
	})
}
