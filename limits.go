package embedding

import (
	"fmt"
	"log"
	"sync"
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

// modelLimits is the package-level registry. Reads go through LookupLimits,
// writes through RegisterLimits — both take modelLimitsMu so the registry
// is safe to mutate at any time, not only at startup.
var (
	modelLimitsMu sync.RWMutex
	modelLimits   = map[string]Limits{
		"nomic-embed-text":    {MaxBytes: 8000, MaxTokens: 2000},
		"nomic-embed-text-v2": {MaxBytes: 8000, MaxTokens: 2000},
		"embeddinggemma":      {MaxBytes: 8000, MaxTokens: 2000},
	}
)

// LookupLimits returns the registered limits for model, or a zero Limits if
// the model is unknown. A zero Limits means no enforcement is applied.
func LookupLimits(model string) Limits {
	modelLimitsMu.RLock()
	defer modelLimitsMu.RUnlock()
	return modelLimits[model]
}

// RegisterLimits adds or overrides the limits for a model. Safe to call at
// any time — concurrent embedders will pick up the new limits on their next
// LookupLimits call.
func RegisterLimits(model string, l Limits) {
	modelLimitsMu.Lock()
	defer modelLimitsMu.Unlock()
	modelLimits[model] = l
}

// unregisterLimits removes a model from the registry. Used by tests; not
// part of the public API because production code should use RegisterLimits
// to set explicit limits rather than fall back to "unknown model" behavior.
func unregisterLimits(model string) {
	modelLimitsMu.Lock()
	defer modelLimitsMu.Unlock()
	delete(modelLimits, model)
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
