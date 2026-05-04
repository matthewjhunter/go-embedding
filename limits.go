package embedding

import (
	"fmt"
	"log"
)

// Limits describes the maximum input size a model accepts in a single
// embed call. Zero values mean "no enforcement."
//
// MaxBytes is enforced before the request is sent. MaxTokens is reserved
// for a future tokenizer-aware enforcement path; it is informational in
// this version.
type Limits struct {
	MaxBytes  int
	MaxTokens int
}

// modelLimits is the package-level registry. Callers may inspect it via
// LookupLimits and extend it via RegisterLimits.
var modelLimits = map[string]Limits{
	// nomic-embed-text family — 8000-byte cap is conservative; the model
	// supports more but quality degrades on longer inputs and the prefix
	// usually carries the signal.
	"nomic-embed-text":    {MaxBytes: 8000, MaxTokens: 2000},
	"nomic-embed-text-v2": {MaxBytes: 8000, MaxTokens: 2000},
	"embeddinggemma":      {MaxBytes: 8000, MaxTokens: 2000},
}

// LookupLimits returns the registered limits for model, or a zero Limits if
// the model is unknown. A zero Limits means no enforcement is applied.
func LookupLimits(model string) Limits {
	return modelLimits[model]
}

// RegisterLimits adds or overrides the limits for a model. Useful for
// custom or unrecognised models. Not safe for concurrent use during runtime;
// call at process startup.
func RegisterLimits(model string, l Limits) {
	modelLimits[model] = l
}

// applyLimits enforces limits against texts. In Strict mode, oversize input
// returns an error. Otherwise oversize input is truncated to MaxBytes and a
// log line is emitted so the truncation is not silent.
//
// Returns the (possibly truncated) texts, or an error in Strict mode.
func applyLimits(texts []string, model string, strict bool) ([]string, error) {
	limits := LookupLimits(model)
	if limits.MaxBytes == 0 {
		return texts, nil
	}

	out := make([]string, len(texts))
	for i, t := range texts {
		if len(t) <= limits.MaxBytes {
			out[i] = t
			continue
		}
		if strict {
			return nil, fmt.Errorf(
				"embedding: input %d exceeds %s MaxBytes (%d > %d)",
				i, model, len(t), limits.MaxBytes,
			)
		}
		truncated := truncateToBytes(t, limits.MaxBytes)
		log.Printf(
			"embedding: truncated input %d for model %q from %d to %d bytes",
			i, model, len(t), len(truncated),
		)
		out[i] = truncated
	}
	return out, nil
}

// truncateToBytes returns s clipped to at most max bytes, backing off any
// trailing UTF-8 continuation bytes so the result is a valid UTF-8 string.
func truncateToBytes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	for max > 0 && (s[max]&0xC0) == 0x80 {
		max--
	}
	return s[:max]
}
