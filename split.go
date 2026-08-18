package embedding

import (
	"strings"
	"unicode/utf8"
)

// Chunk is one segment of a split document. Text is exactly
// source[Start:End], so a caller can record provenance -- which part of which
// document a vector came from -- without re-deriving offsets that overlap
// makes impossible to recompute.
type Chunk struct {
	// Text is the chunk's content, trimmed of surrounding whitespace.
	Text string
	// Start and End are byte offsets into the text passed to Split.
	Start int
	End   int
	// Ordinal is the chunk's position in the returned slice, from 0.
	Ordinal int
	// Headings is the heading path in force at this chunk, outermost first
	// (e.g. {"Deployment", "Rollback", "Manual steps"}). Populated only under
	// StructureMarkdown, and nil for a chunk preceding the first heading.
	//
	// It is reported, not spliced into Text: prepending it before embedding
	// is a caller decision, and injecting it would break the guarantee that
	// Text is exactly source[Start:End].
	Headings []string
}

// SplitOptions tunes Split. The zero value splits at the model's registered
// byte budget with no overlap, which is the safe default rather than the good
// one -- see Split.
type SplitOptions struct {
	// MaxBytes is the maximum size of a chunk. Zero uses the model's
	// registered byte budget. Values below utf8.UTFMax are raised to it: a
	// budget narrower than a single rune cannot be honoured without cutting
	// one in half.
	MaxBytes int
	// Overlap is how many bytes of the previous chunk are repeated at the
	// start of the next, so a boundary landing mid-argument does not strand
	// the two halves. Zero means no overlap. Values at or above half of
	// MaxBytes are clamped, since a rewind larger than the step forward would
	// never terminate.
	Overlap int
	// MinBytes absorbs a trailing chunk shorter than this into its
	// predecessor, when the two fit within MaxBytes together. A sliver of a
	// few words embeds to a vector that matches almost nothing useful.
	MinBytes int
	// Structure is how much of the input's shape to understand. The zero
	// value treats it as flat text; StructureMarkdown reads headings, records
	// the path on each chunk, and prefers to break where sections do.
	Structure Structure
	// MaxTokens sizes chunks in tokens rather than bytes, which is the unit
	// chunk size is actually reasoned about in. With Tokenizer set it is
	// exact; without one it is converted through BytesPerToken. Zero leaves
	// sizing to MaxBytes.
	//
	// When both MaxTokens and MaxBytes are set, whichever binds first wins,
	// so MaxBytes acts as a hard ceiling on a token-sized chunk.
	MaxTokens int
	// Tokenizer makes MaxTokens exact. See TokenCounter.
	Tokenizer TokenCounter
	// BytesPerToken converts MaxTokens to a byte budget when Tokenizer is
	// nil. Zero uses a conservative built-in guess.
	BytesPerToken float64
}

// minFillPercent is how full a chunk must be before a boundary is worth
// taking. Without it, a paragraph break near the start of the window would
// emit a tiny chunk and waste most of the budget; with it, the splitter falls
// through to a finer boundary that lands closer to the target size.
const minFillPercent = 50

// Split divides text into chunks that each fit a byte budget, preferring to
// break at a paragraph, then a line, then a sentence, then a word, and only
// cutting mid-word when a single run leaves no choice. Cuts always land on a
// rune boundary.
//
// The budget is opts.MaxBytes, or the model's registered byte budget when
// that is zero. A model with no registered budget and no override is returned
// as a single chunk: there is nothing to split against, and inventing a size
// would be worse than passing the document through.
//
// Note that the model's full budget is the wrong target for retrieval. A
// vector has fixed capacity, so filling the whole context window averages
// away the specificity that retrieval depends on; 256-512 tokens per chunk is
// the usual working range, which is well under a 2048-token window. Split
// defaults to the budget because that is the only figure the library knows,
// not because it is the figure a caller wants. Pass MaxBytes.
//
// Whitespace at a chunk's edges is trimmed, so with no overlap the chunks
// reconstruct the source modulo whitespace, and no content is dropped.
func Split(model, text string, opts SplitOptions) []Chunk {
	if strings.TrimSpace(text) == "" {
		return nil
	}

	// A tokenizer makes MaxTokens exact; without one it converts to bytes
	// through the ratio, the same fallback the pre-flight budget uses.
	tc := opts.Tokenizer
	maxTokens := opts.MaxTokens
	maxBytes := opts.MaxBytes
	switch {
	case maxTokens > 0 && tc != nil:
		// Token-governed. MaxBytes, if given, stays a hard ceiling; if not,
		// the document itself is the only bound on the search window.
		if maxBytes <= 0 {
			maxBytes = len(text)
		}
	case maxTokens > 0:
		// A token target with nothing to count tokens: convert it, and keep
		// any explicit byte ceiling that is tighter.
		maxTokens = 0
		converted := int(float64(opts.MaxTokens) * bytesPerToken(opts.BytesPerToken))
		if converted > 0 {
			if maxBytes > 0 {
				maxBytes = min(maxBytes, converted)
			} else {
				maxBytes = converted
			}
		}
	case maxBytes <= 0:
		maxBytes = effectiveLimits(model, Limits{}, opts.BytesPerToken).MaxBytes
	}

	if maxBytes <= 0 || (len(text) <= maxBytes && maxTokens == 0) {
		if c, ok := newChunk(text, 0, len(text), 0); ok {
			return []Chunk{c}
		}
		return nil
	}

	// A budget narrower than one rune cannot hold a chunk without cutting a
	// rune in half, so floor it. Below this the question is not "where do I
	// split" but "what would a chunk even mean".
	maxBytes = max(maxBytes, utf8.UTFMax)

	var headings []heading
	var sectionStarts []int
	if opts.Structure == StructureMarkdown {
		headings = scanMarkdown(text)
		sectionStarts = make([]int, len(headings))
		for i, h := range headings {
			sectionStarts[i] = h.Start
		}
	}

	overlap := max(opts.Overlap, 0)
	// Clamp so every step advances: the window moves forward by maxBytes and
	// back by overlap, so an overlap at or above maxBytes would stall.
	overlap = min(overlap, maxBytes/2)

	// windowEnd is the furthest a chunk starting at start may reach.
	windowEnd := func(start int) int {
		end := min(start+maxBytes, len(text))
		if maxTokens > 0 {
			end = tokenExtent(tc, text, start, end, maxTokens)
		}
		return end
	}

	var chunks []Chunk
	lastStart := -1 // Start of the last emitted chunk, after trimming.
	for start := 0; start < len(text); {
		limit := windowEnd(start)
		if limit >= len(text) {
			if c, ok := newChunk(text, start, len(text), len(chunks)); ok {
				chunks = append(chunks, c)
			}
			break
		}

		end := breakPoint(text, start, limit, sectionStarts)
		if end <= start {
			// Only reachable on invalid UTF-8, where backing off to a rune
			// boundary can walk past the window entirely. Advance one rune
			// (one byte, for a byte that starts no rune) so the loop cannot
			// stall -- a hang here would be far worse than an ugly chunk.
			_, size := utf8.DecodeRuneInString(text[start:])
			end = start + max(size, 1)
		}
		if c, ok := newChunk(text, start, end, len(chunks)); ok {
			chunks = append(chunks, c)
			lastStart = c.Start
		}

		next := backUpToRune(text, end-overlap)
		// Progress is measured against the chunk just emitted, not against
		// the window: leading whitespace means a window can start earlier
		// than its chunk does, and an overlap that rewound to at or before
		// that point would make the next chunk a superset of this one --
		// duplicate content, and an embed call spent on it.
		if next <= start || next <= lastStart {
			next = end
		}
		start = next
	}

	chunks = absorbSliver(text, chunks, maxBytes, opts.MinBytes, tc, maxTokens)
	for i := range chunks {
		chunks[i].Headings = headingPathAt(headings, chunks[i].Start)
	}
	return chunks
}

// breakPoint picks where to end a chunk that starts at start and may run no
// further than limit. It scans the tail of the window for the best available
// separator, taking the first kind that lands in the acceptable range.
func breakPoint(text string, start, limit int, sectionStarts []int) int {
	// Only consider boundaries past this point; a break too near the start
	// wastes the budget.
	return breakPointAbove(text, start, limit, start+(limit-start)*minFillPercent/100, sectionStarts)
}

// breakPointAbove is breakPoint with a caller-supplied floor, so the tail
// rebalance can demand a boundary late enough to leave a legal final chunk.
func breakPointAbove(text string, start, limit, floor int, sectionStarts []int) int {
	// A section boundary outranks everything: ending a chunk where the
	// document ends a section beats ending it three lines into the next one.
	// Sections can start below the fill floor -- a short section is still a
	// better cut than a paragraph break inside the following one -- so only
	// forward progress is required.
	for i := len(sectionStarts) - 1; i >= 0; i-- {
		if b := sectionStarts[i]; b > start && b <= limit {
			return b
		}
	}

	window := text[start:limit]
	rel := limit - start

	// Paragraph, then line.
	if i := strings.LastIndex(window, "\n\n"); i >= 0 && start+i+2 > floor {
		return start + i + 2
	}
	if i := strings.LastIndexByte(window, '\n'); i >= 0 && start+i+1 > floor {
		return start + i + 1
	}
	// Sentence: terminal punctuation followed by a space, so "3.14" and
	// "e.g." inside a line are not mistaken for the end of a thought.
	for i := rel - 1; i > 0; i-- {
		if !isSpaceByte(window[i]) {
			continue
		}
		if p := window[i-1]; p == '.' || p == '!' || p == '?' {
			if start+i > floor {
				return start + i
			}
			break
		}
	}
	// Word.
	for i := rel - 1; i > 0; i-- {
		if isSpaceByte(window[i]) {
			if start+i > floor {
				return start + i
			}
			break
		}
	}
	// No boundary worth taking: cut at the limit, backing off to a rune
	// boundary so the chunk stays valid UTF-8.
	return backUpToRune(text, limit)
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

// backUpToRune moves i back to the nearest rune boundary at or before it.
func backUpToRune(text string, i int) int {
	if i <= 0 {
		return 0
	}
	if i >= len(text) {
		return len(text)
	}
	for i > 0 && !utf8.RuneStart(text[i]) {
		i--
	}
	return i
}

// newChunk trims whitespace from the span and reports whether anything is
// left. Offsets track the trim so Text stays equal to source[Start:End].
func newChunk(text string, start, end, ordinal int) (Chunk, bool) {
	for start < end && isSpaceByte(text[start]) {
		start++
	}
	for end > start && isSpaceByte(text[end-1]) {
		end--
	}
	if start >= end {
		return Chunk{}, false
	}
	return Chunk{Text: text[start:end], Start: start, End: end, Ordinal: ordinal}, true
}

// absorbSliver removes a too-short final chunk, either by merging it into its
// predecessor when the two fit the budget together, or by re-splitting the
// pair down the middle when they do not.
//
// The rebalance matters more than the merge. A sliver usually turns up behind
// a nearly-full chunk, so merging would overrun the budget and a merge-only
// implementation would decline exactly when it was needed. Rebalancing always
// applies: the pair spans at most two budgets, so a boundary in the middle
// leaves two legal chunks.
//
// MinBytes above half of MaxBytes cannot always be honoured -- a rebalanced
// chunk is about half the pair -- and the tail pair does not overlap after a
// rebalance, since it is recut from the source rather than stepped through.
func absorbSliver(text string, chunks []Chunk, maxBytes, minBytes int, tc TokenCounter, maxTokens int) []Chunk {
	if minBytes <= 0 || len(chunks) < 2 {
		return chunks
	}
	last := chunks[len(chunks)-1]
	if len(last.Text) >= minBytes {
		return chunks
	}
	prev := chunks[len(chunks)-2]

	if last.End-prev.Start <= maxBytes && fitsTokens(tc, text, prev.Start, last.End, maxTokens) {
		merged, ok := newChunk(text, prev.Start, last.End, prev.Ordinal)
		if !ok {
			return chunks
		}
		chunks = chunks[:len(chunks)-1]
		chunks[len(chunks)-1] = merged
		return chunks
	}

	// Recut the pair down the middle. The search runs backward from its
	// limit, so the limit is the midpoint rather than the end -- searching to
	// the end would just re-find the boundary that produced the sliver. The
	// floor keeps the second half within the budget; it is satisfiable because
	// the pair spans at most two budgets.
	lo := max(prev.Start, last.End-maxBytes)
	hi := min(last.End, prev.Start+maxBytes)
	mid := breakPointAbove(text, prev.Start, min((prev.Start+last.End)/2, hi), lo, nil)
	if mid <= prev.Start || mid >= last.End {
		return chunks
	}

	head, headOK := newChunk(text, prev.Start, mid, prev.Ordinal)
	tail, tailOK := newChunk(text, mid, last.End, last.Ordinal)
	if !headOK || !tailOK {
		return chunks
	}
	// Check the result rather than trusting the bounds: on invalid UTF-8 the
	// boundary search can back off below lo, which would hand back a tail
	// over budget. Leaving the sliver is the lesser failure.
	if len(head.Text) > maxBytes || len(tail.Text) > maxBytes {
		return chunks
	}
	if !fitsTokens(tc, text, head.Start, head.End, maxTokens) ||
		!fitsTokens(tc, text, tail.Start, tail.End, maxTokens) {
		return chunks
	}
	chunks[len(chunks)-2] = head
	chunks[len(chunks)-1] = tail
	return chunks
}

// TokenBudget returns the registered token budget for model, or 0 when none
// is registered. It reflects the static registry and RegisterLimits only; a
// per-embedder override lives on Config, where Config.Limits reports it.
func TokenBudget(model string) int {
	return LookupLimits(model).MaxTokens
}

// Limits reports the byte and token budget this configuration puts in force:
// the registered limits for the model, with any non-zero override applied,
// and a byte budget derived from MaxTokens when the model is unregistered.
func (c Config) Limits() Limits {
	return effectiveLimits(c.Model, Limits{MaxBytes: c.MaxBytes, MaxTokens: c.MaxTokens}, c.BytesPerToken)
}

// fitsTokens reports whether text[start:end] is within maxTokens. A zero
// budget or a missing counter means tokens are not being enforced.
func fitsTokens(tc TokenCounter, text string, start, end, maxTokens int) bool {
	if tc == nil || maxTokens <= 0 {
		return true
	}
	return tc.CountTokens(text[start:end]) <= maxTokens
}
