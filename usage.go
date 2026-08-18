package embedding

import (
	"fmt"
	"log"
)

// Usage reports what one embed request actually cost at the backend. Both
// supported protocols return the token count of the prompt they processed
// (Ollama's prompt_eval_count, the OpenAI shape's usage.prompt_tokens), which
// is the only ground truth available without shipping a tokenizer.
//
// It is delivered to Config.OnUsage after each successful request, and only
// when the backend actually reported a count.
type Usage struct {
	// Model is the embedding model the request was sent to.
	Model string
	// Inputs is how many texts the request carried.
	Inputs int
	// Bytes is the total size of those texts as sent, after any pre-flight
	// truncation to the byte budget.
	Bytes int
	// Tokens is the token count the backend reported for the whole request.
	Tokens int
	// MaxTokens is the token budget in force for the model, or 0 when none
	// is known.
	MaxTokens int
	// Clipped reports that the input almost certainly overran the model's
	// context and was silently truncated by the backend. See checkUsage for
	// when this can and cannot be determined.
	Clipped bool
}

// UsageReporter receives a Usage after each request the backend reported a
// token count for.
//
// It is an interface rather than a func field so Config stays comparable --
// a func-typed field would make every Config uncomparable, breaking callers
// that compare them. UsageFunc adapts an ordinary function, mirroring
// http.Handler / http.HandlerFunc.
type UsageReporter interface {
	ReportUsage(Usage)
}

// UsageFunc adapts a function to the UsageReporter interface.
type UsageFunc func(Usage)

// ReportUsage calls f(u).
func (f UsageFunc) ReportUsage(u Usage) { f(u) }

// checkUsage turns a backend-reported token count into a Usage observation,
// delivers it to onUsage, and enforces strict mode.
//
// Clipping is inferred rather than reported: Ollama silently clamps an
// over-length input at the context window and returns a vector computed from
// the clipped text, so a count that has reached the budget is the fingerprint
// of an input that was cut. The inference has two limits worth knowing.
//
// It cannot attribute a batch. Both protocols report one total for the whole
// request, so an over-budget total across several inputs is the normal shape
// rather than evidence about any single one; Clipped stays false there.
// Single-input requests are the case this catches, which includes every
// one-by-one fallback BatchEmbedResults performs.
//
// It also cannot distinguish a clipped input from one that legitimately
// lands within a few tokens of the budget. That false positive is the right
// way to be wrong: a budget is set below the true context window precisely so
// nothing lands there.
func checkUsage(model string, limits Limits, texts []string, tokens int, strict bool, onUsage UsageReporter) error {
	if tokens <= 0 {
		// The backend reported nothing. Inventing an observation here would
		// poison anything that later averages these.
		return nil
	}

	bytes := 0
	for _, t := range texts {
		bytes += len(t)
	}

	u := Usage{
		Model:     model,
		Inputs:    len(texts),
		Bytes:     bytes,
		Tokens:    tokens,
		MaxTokens: limits.MaxTokens,
		Clipped:   limits.MaxTokens > 0 && len(texts) == 1 && tokens >= limits.MaxTokens,
	}
	recordUsage(u)
	if onUsage != nil {
		onUsage.ReportUsage(u)
	}
	if !u.Clipped {
		return nil
	}
	if strict {
		// Strict mode refuses to truncate before sending; it refuses a vector
		// computed from clipped text for the same reason. Not TooLong, which
		// would send this to the adaptive shrink path -- strict callers have
		// opted out of being silently shrunk (see applyLimits).
		return &PermanentError{Err: fmt.Errorf(
			"embedding: input for model %q was clipped by the backend (%d tokens against a %d budget)",
			model, tokens, limits.MaxTokens,
		)}
	}
	log.Printf(
		"embedding: input for model %q was clipped by the backend (%d tokens reached the %d-token budget); the tail of this document is not in the vector",
		model, tokens, limits.MaxTokens,
	)
	return nil
}
