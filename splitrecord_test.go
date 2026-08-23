package embedding

import (
	"strings"
	"testing"
)

const srModel = "nomic-embed-text"

func srFields() []Field {
	return []Field{{Key: "feed", Value: "Example Weekly"}, {Key: "title", Value: "On Retry Budgets"}}
}

func srBody() string {
	return strings.Repeat("The retry budget needs careful tuning. ", 200)
}

// The trap both callers hit first: splitting the *rendered* text leaves chunk 0
// with the header and chunks 1..N as anonymous prose. No error, just worse
// retrieval on exactly the long documents chunking was meant to rescue.
func TestSplitRecord_EveryChunkCarriesTheHeader(t *testing.T) {
	chunks := SplitRecord(srModel, TaskRetrievalDocument, srFields(), srBody(), SplitOptions{MaxBytes: 1024})

	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want several", len(chunks))
	}
	for i, c := range chunks {
		if !strings.Contains(c.Text, "On Retry Budgets") {
			t.Errorf("chunk %d lost the header: %q", i, c.Text[:min(60, len(c.Text))])
		}
		if !strings.HasPrefix(c.Text, "search_document:") {
			t.Errorf("chunk %d lost the task prefix", i)
		}
	}
}

// The other half of the same trap: offsets taken from the rendered text drift
// by the accumulated header bytes, so late chunks report spans past the end of
// the source.
func TestSplitRecord_OffsetsAddressTheBody(t *testing.T) {
	body := srBody()
	chunks := SplitRecord(srModel, TaskRetrievalDocument, srFields(), body, SplitOptions{MaxBytes: 1024})

	for _, c := range chunks {
		if c.Start < 0 || c.End > len(body) || c.Start >= c.End {
			t.Fatalf("chunk %d span [%d,%d) is not inside the %d-byte body",
				c.Ordinal, c.Start, c.End, len(body))
		}
		if got := body[c.Start:c.End]; !strings.Contains(c.Text, got) {
			t.Errorf("chunk %d: Text does not contain body[%d:%d]", c.Ordinal, c.Start, c.End)
		}
	}
}

// MaxBytes bounds the *rendered* chunk. Sizing the body at the full budget and
// then prepending a header puts every request over it.
func TestSplitRecord_BudgetBoundsTheRenderedChunk(t *testing.T) {
	const budget = 900
	chunks := SplitRecord(srModel, TaskRetrievalDocument, srFields(), srBody(), SplitOptions{MaxBytes: budget})

	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want several", len(chunks))
	}
	for _, c := range chunks {
		if len(c.Text) > budget {
			t.Errorf("chunk %d rendered to %d bytes, over the %d budget", c.Ordinal, len(c.Text), budget)
		}
	}
}

// The header is measured, not assumed. FormatRecord trims its trailing newline
// when the body is empty but adds a blank-line separator when it is not, so the
// obvious len(format("")) undercounts by two bytes -- enough to put a
// tightly-budgeted chunk over a hard backend limit.
func TestSplitRecord_HeaderMeasurementAccountsForTheSeparator(t *testing.T) {
	fields := srFields()
	naive := len(FormatRecordForTask(srModel, TaskRetrievalDocument, fields, ""))
	actual := recordOverhead(srModel, TaskRetrievalDocument, fields)

	if actual <= naive {
		t.Errorf("measured overhead %d does not exceed the naive %d; the separator is unaccounted",
			actual, naive)
	}
}

// A token target is converted through the observed ratio, so callers express
// chunk size in the unit it is reasoned about in.
func TestSplitRecord_TokenTargetIsHonoured(t *testing.T) {
	ResetCalibration()
	small := SplitRecord(srModel, TaskRetrievalDocument, srFields(), srBody(), SplitOptions{MaxTokens: 128})
	large := SplitRecord(srModel, TaskRetrievalDocument, srFields(), srBody(), SplitOptions{MaxTokens: 512})

	if len(small) <= len(large) {
		t.Errorf("a 128-token target gave %d chunks and 512 gave %d; the token target is inert",
			len(small), len(large))
	}
}

// Target and ceiling: whichever binds first wins, and the ceiling can only make
// chunks smaller.
func TestSplitRecord_CeilingClampsTheTokenTarget(t *testing.T) {
	ResetCalibration()
	const ceiling = 400
	chunks := SplitRecord(srModel, TaskRetrievalDocument, srFields(), srBody(),
		SplitOptions{MaxTokens: 4096, MaxBytes: ceiling})

	for _, c := range chunks {
		if len(c.Text) > ceiling {
			t.Errorf("chunk %d rendered to %d bytes, over the %d ceiling", c.Ordinal, len(c.Text), ceiling)
		}
	}
}

func TestSplitRecord_EmptyBodyYieldsNothing(t *testing.T) {
	if got := SplitRecord(srModel, TaskRetrievalDocument, srFields(), "  \n ", SplitOptions{MaxBytes: 1024}); len(got) != 0 {
		t.Errorf("got %d chunks for a whitespace body", len(got))
	}
}
