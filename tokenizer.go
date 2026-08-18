package embedding

import "unicode/utf8"

// TokenCounter reports how many tokens a model's tokenizer makes of a string.
//
// Supplying one makes token budgets exact. Without it, every token budget in
// this library is converted to a byte budget through a ratio -- a guess that
// is wrong by 50-70% on real text, differs per corpus, and changes with every
// model. With it, there is no ratio: input is truncated to the token budget it
// actually has, and Split sizes chunks in the unit a caller reasons in.
//
// The library deliberately does not ship a tokenizer. Vocabularies belong to
// models, Go bindings vary in weight and licence, and a tokenizer would have
// to be swapped in lockstep with every embedder change. Pure-Go
// implementations exist for the common families (SentencePiece for the Gemma
// line, WordPiece for BERT-derived models); wire one in through this
// interface.
//
// It is an interface rather than a func field so Config stays comparable.
// TokenCountFunc adapts an ordinary function.
//
// Implementations must be safe for concurrent use, and must be monotonic over
// prefixes: a longer prefix of the same string never counts fewer tokens. The
// truncation search relies on that and on nothing else.
type TokenCounter interface {
	CountTokens(text string) int
}

// TokenCountFunc adapts a function to the TokenCounter interface.
type TokenCountFunc func(text string) int

// CountTokens calls f(text).
func (f TokenCountFunc) CountTokens(text string) int { return f(text) }

// truncateToTokens returns the longest prefix of s that fits maxTokens,
// or s unchanged when it already does.
//
// It binary-searches byte offsets rather than walking tokens, so it needs
// nothing from the tokenizer but a count, and costs O(log n) counts. The
// common case -- input that already fits -- costs exactly one.
func truncateToTokens(tc TokenCounter, s string, maxTokens int) string {
	if maxTokens <= 0 || s == "" {
		return s
	}
	if tc.CountTokens(s) <= maxTokens {
		return s
	}

	// Invariant: lo always fits, hi never does.
	lo, hi := 0, len(s)
	for lo < hi {
		mid := backUpToRune(s, lo+(hi-lo+1)/2)
		if mid <= lo {
			break
		}
		if tc.CountTokens(s[:mid]) <= maxTokens {
			lo = mid
		} else {
			hi = mid - 1
			hi = backUpToRune(s, hi)
			if hi < lo {
				hi = lo
			}
		}
	}
	return s[:lo]
}

// tokenExtent returns the largest end offset at or before limit such that
// text[start:end] fits maxTokens. It is truncateToTokens expressed as an
// offset, for the splitter's window.
func tokenExtent(tc TokenCounter, text string, start, limit, maxTokens int) int {
	if start >= limit {
		return limit
	}
	fitted := truncateToTokens(tc, text[start:limit], maxTokens)
	end := start + len(fitted)
	if end <= start {
		// Not even one rune's worth fits the budget -- a budget smaller than
		// a single token. Advance one rune so the caller cannot stall.
		_, size := utf8.DecodeRuneInString(text[start:])
		return min(start+max(size, 1), limit)
	}
	return end
}
