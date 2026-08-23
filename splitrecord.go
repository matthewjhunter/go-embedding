package embedding

import "unicode/utf8"

// RenderedChunk is one chunk of a record, ready to embed.
type RenderedChunk struct {
	// Text is the fully rendered chunk: task prefix, metadata fields, then
	// this chunk's slice of the body. It is what goes to the backend.
	Text string
	// Start and End are byte offsets into the *body* passed to SplitRecord --
	// not into Text.
	//
	// Rendering adds a header to every chunk, so offsets derived from Text
	// drift by the accumulated header bytes and late chunks report spans past
	// the end of the source. Body offsets are what make a chunk hit resolvable
	// to a passage of the document, and overlap makes them impossible to
	// recompute afterwards, so they are reported rather than implied.
	Start, End int
	// Ordinal is the chunk's position, from 0.
	Ordinal int
	// Headings is the heading path in force at this chunk under
	// StructureMarkdown; nil otherwise. See Chunk.Headings.
	Headings []string
}

// recordOverhead measures what rendering adds to every chunk: the task prefix,
// the field lines, and the blank line separating them from the body.
//
// It measures rather than assumes, and it measures *with* a body. FormatRecord
// trims its trailing newline when the body is empty but emits a blank-line
// separator when it is not, so the obvious len(format("")) undercounts by two
// bytes -- enough to put a tightly-budgeted chunk over a backend limit that
// rejects rather than truncates.
func recordOverhead(model string, task Task, fields []Field) int {
	const probe = "x"
	return len(FormatRecordForTask(model, task, fields, probe)) - len(probe)
}

// SplitRecord chunks a record: the body is split to fit the budget, and the
// metadata header is re-applied to every chunk.
//
// This is the record-shaped counterpart to Split, and it exists because every
// caller that has written it by hand has made the same two mistakes.
//
// The first is splitting the *rendered* text rather than the body. It leaves
// chunk 0 carrying the header while chunks 1..N are anonymous prose with
// nothing saying which document they belong to -- no error, just worse
// retrieval on exactly the long documents chunking was meant to rescue. It also
// drifts the offsets, since each rendering adds header bytes the source does
// not have.
//
// The second is not charging the header against the budget: size the body at
// the full budget, prepend a header, and every request goes over it.
//
// opts.MaxBytes bounds the *rendered* chunk, and opts.MaxTokens expresses the
// same bound in tokens (converted through BudgetForTokens, so it follows the
// model's observed bytes-per-token rather than a guess). With both set,
// whichever binds first wins -- the usual arrangement being a token target with
// a byte ceiling the backend imposes, where the ceiling can only ever make
// chunks smaller.
//
// Overlap and MinBytes apply to the body, as in Split.
func SplitRecord(model string, task Task, fields []Field, body string, opts SplitOptions) []RenderedChunk {
	format := func(b string) string { return FormatRecordForTask(model, task, fields, b) }

	// Resolve the rendered budget: the byte ceiling and the token target, with
	// whichever binds first winning.
	budget := opts.MaxBytes
	if opts.MaxTokens > 0 {
		if fromTokens := BudgetForTokens(model, opts.MaxTokens); fromTokens > 0 {
			if budget <= 0 || fromTokens < budget {
				budget = fromTokens
			}
		}
	}

	inner := opts
	inner.MaxTokens = 0 // already folded into the byte budget
	inner.Tokenizer = nil
	if budget > 0 {
		overhead := recordOverhead(model, task, fields)
		inner.MaxBytes = budget - overhead
		if inner.MaxBytes < utf8.UTFMax {
			// A header leaving no room for a body means the fields are
			// pathological. Emit what fits rather than splitting into slivers;
			// applyLimits still clips to the model's budget on the way out.
			inner.MaxBytes = utf8.UTFMax
		}
	} else {
		inner.MaxBytes = 0
	}

	chunks := Split(model, body, inner)
	out := make([]RenderedChunk, len(chunks))
	for i, c := range chunks {
		out[i] = RenderedChunk{
			Text:     format(c.Text),
			Start:    c.Start,
			End:      c.End,
			Ordinal:  c.Ordinal,
			Headings: c.Headings,
		}
	}
	return out
}
