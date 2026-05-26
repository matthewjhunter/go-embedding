package embedding

import (
	"fmt"
	"log"
	"strings"
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

// Registered byte budgets are deliberately conservative. A model with a
// 2048-token context window typically accepts ~3 bytes/token of dense
// English news (markup, URLs, and special tokens eat headroom faster than
// the naive ~4 bytes/token estimate suggests). 6000 bytes ≈ 2000 tokens
// with a small safety margin, which empirically avoids backend rejects
// where 8000 bytes did not.
var (
	modelLimitsMu sync.RWMutex
	modelLimits   = map[string]Limits{
		"nomic-embed-text":    {MaxBytes: 6000, MaxTokens: 2000},
		"nomic-embed-text-v2": {MaxBytes: 6000, MaxTokens: 2000},
		"embeddinggemma":      {MaxBytes: 6000, MaxTokens: 2000},
		// Reranker. bge-reranker-v2-m3 has an 8192-token context shared by the
		// (query+document) pair. Budget the document well under that — ~6000
		// tokens at the same conservative ~3 bytes/token used above — leaving
		// generous headroom for the query and the model's special tokens. This
		// only clips pathologically long facts; typical facts sit far below it.
		"bge-reranker-v2-m3": {MaxBytes: 18000, MaxTokens: 6000},
	}
)

// LookupLimits returns the registered limits for model, or a zero Limits
// if the model is unknown. A zero Limits means no enforcement is applied.
//
// If an exact match is not found and the model name carries a tag suffix
// (e.g. "nomic-embed-text:latest", "nomic-embed-text:q4_0"), LookupLimits
// retries with the bare model name. Limits describe an architectural
// property — the model's context window — which is shared across tagged
// variants of the same base model. Storage keys are NOT canonicalised
// this way; vectors from different tags are not interchangeable.
func LookupLimits(model string) Limits {
	modelLimitsMu.RLock()
	defer modelLimitsMu.RUnlock()
	if l, ok := modelLimits[model]; ok {
		return l
	}
	if i := strings.IndexByte(model, ':'); i > 0 {
		return modelLimits[model[:i]]
	}
	return Limits{}
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
			// Strict mode opts out of truncation, so this is a permanent
			// failure for this input — but not TooLong-adaptive, since the
			// caller explicitly does not want it shrunk.
			return nil, &PermanentError{Err: fmt.Errorf(
				"embedding: input %d exceeds %s MaxBytes (%d > %d)",
				i, model, len(t), limits.MaxBytes,
			)}
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

// limitDocuments truncates each document to maxBytes (maxBytes <= 0 means no
// limit). In Strict mode an oversize document is a *PermanentError instead of
// being truncated. Unlike applyLimits it does not log per document: rerank
// callers truncate on every request (a per-prompt recall path truncates its
// whole pool each time), so per-document logging would be noise — truncation
// here is configured behaviour, not an anomaly. model is used only for error
// context.
func limitDocuments(docs []string, maxBytes int, model string, strict bool) ([]string, error) {
	if maxBytes <= 0 {
		return docs, nil
	}
	out := make([]string, len(docs))
	for i, d := range docs {
		if len(d) <= maxBytes {
			out[i] = d
			continue
		}
		if strict {
			return nil, &PermanentError{Err: fmt.Errorf(
				"embedding: rerank document %d exceeds %s budget (%d > %d bytes)",
				i, model, len(d), maxBytes,
			)}
		}
		out[i] = truncateToBytes(d, maxBytes)
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
