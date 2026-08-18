package embedding

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// checkInvariants asserts the properties every Split result must hold,
// whatever the input: chunks fit the budget, stay valid UTF-8, carry offsets
// that actually address their own text, and advance through the source.
func checkInvariants(t *testing.T, source string, chunks []Chunk, maxBytes int) {
	t.Helper()
	// Split cuts on rune boundaries but does not repair its input: garbage in
	// stays garbage out, so validity is only required of chunks cut from a
	// valid source.
	sourceValid := utf8.ValidString(source)
	for i, c := range chunks {
		if maxBytes > 0 && len(c.Text) > maxBytes {
			t.Errorf("chunk %d is %d bytes, over the %d budget", i, len(c.Text), maxBytes)
		}
		if sourceValid && !utf8.ValidString(c.Text) {
			t.Errorf("chunk %d is not valid UTF-8 -- a split landed mid-rune", i)
		}
		if c.Start < 0 || c.End > len(source) || c.Start > c.End {
			t.Fatalf("chunk %d has nonsense offsets [%d:%d] for a %d-byte source", i, c.Start, c.End, len(source))
		}
		if got := source[c.Start:c.End]; got != c.Text {
			t.Errorf("chunk %d: source[%d:%d] = %q, but Text = %q", i, c.Start, c.End, got, c.Text)
		}
		if c.Ordinal != i {
			t.Errorf("chunk %d has Ordinal %d", i, c.Ordinal)
		}
		if i > 0 && c.Start <= chunks[i-1].Start {
			t.Errorf("chunk %d starts at %d, not after chunk %d at %d -- no forward progress",
				i, c.Start, i-1, chunks[i-1].Start)
		}
	}
}

// stripSpace removes all whitespace, so two texts can be compared for content
// regardless of where the splitter trimmed.
func stripSpace(s string) string {
	return strings.Join(strings.Fields(s), "")
}

func TestSplit_ShortTextIsOneChunk(t *testing.T) {
	text := "a short document that fits comfortably"
	got := Split("nomic-embed-text", text, SplitOptions{})
	if len(got) != 1 {
		t.Fatalf("got %d chunks, want 1", len(got))
	}
	if got[0].Text != text {
		t.Errorf("Text = %q, want %q", got[0].Text, text)
	}
	checkInvariants(t, text, got, 6000)
}

func TestSplit_EmptyInput(t *testing.T) {
	for _, text := range []string{"", "   ", "\n\n\t "} {
		if got := Split("nomic-embed-text", text, SplitOptions{}); len(got) != 0 {
			t.Errorf("Split(%q) returned %d chunks, want 0", text, len(got))
		}
	}
}

// A model with no registered budget and no override cannot be split against
// anything, so the document passes through whole rather than being cut at an
// arbitrary size.
func TestSplit_UnknownModelDoesNotSplit(t *testing.T) {
	text := strings.Repeat("word ", 5000)
	got := Split("model-nobody-registered", text, SplitOptions{})
	if len(got) != 1 {
		t.Fatalf("got %d chunks, want 1", len(got))
	}
}

func TestSplit_PrefersParagraphBoundaries(t *testing.T) {
	para := strings.Repeat("sentence text here. ", 5) // 100 bytes
	text := para + "\n\n" + para + "\n\n" + para

	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 150})
	if len(got) < 3 {
		t.Fatalf("got %d chunks, want at least 3", len(got))
	}
	checkInvariants(t, text, got, 150)
	for i, c := range got {
		if strings.Contains(c.Text, "\n\n") {
			t.Errorf("chunk %d spans a paragraph break: %q", i, c.Text)
		}
	}
}

func TestSplit_FallsBackToSentenceThenWord(t *testing.T) {
	// One paragraph, so paragraph boundaries cannot help.
	text := strings.Repeat("This is a sentence of moderate length. ", 20)

	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 120})
	checkInvariants(t, text, got, 120)
	if len(got) < 5 {
		t.Fatalf("got %d chunks, want several", len(got))
	}
	// Sentence boundaries are available at this size, so no chunk should end
	// mid-word.
	for i, c := range got[:len(got)-1] {
		if !strings.HasSuffix(c.Text, ".") {
			t.Errorf("chunk %d does not end at a sentence boundary: %q", i, c.Text)
		}
	}
}

// A single unbroken token longer than the budget (base64, a minified blob)
// has no boundary to find. It must still be cut, and cut safely.
func TestSplit_HardCutsAnUnbreakableRun(t *testing.T) {
	text := strings.Repeat("A", 1000)
	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 100})
	checkInvariants(t, text, got, 100)
	if len(got) != 10 {
		t.Errorf("got %d chunks, want 10", len(got))
	}
	if joined := stripSpace(concat(got)); joined != text {
		t.Errorf("hard cutting lost or duplicated content: %d bytes vs %d", len(joined), len(text))
	}
}

// Invalid UTF-8 is passed through rather than repaired or rejected, and must
// not panic on the way.
func TestSplit_InvalidUTF8PassesThrough(t *testing.T) {
	text := "before \xe3 after " + strings.Repeat("padding ", 20)
	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 40})
	checkInvariants(t, text, got, 40)
	if want, have := stripSpace(text), stripSpace(concat(got)); want != have {
		t.Errorf("content changed: %q vs %q", want, have)
	}
}

func TestSplit_NeverSplitsMidRune(t *testing.T) {
	// Multibyte throughout, with no ASCII space to fall back to.
	text := strings.Repeat("日本語のテキスト", 200)
	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 100})
	checkInvariants(t, text, got, 100)
	if len(got) < 2 {
		t.Fatal("multibyte text was not split at all")
	}
}

func TestSplit_LosesNoContentWithoutOverlap(t *testing.T) {
	text := strings.Repeat("Alpha beta gamma delta. ", 100)
	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200})
	checkInvariants(t, text, got, 200)
	if want, have := stripSpace(text), stripSpace(concat(got)); want != have {
		t.Errorf("content changed: %d bytes in, %d bytes out", len(want), len(have))
	}
}

func TestSplit_OverlapRepeatsTheBoundary(t *testing.T) {
	text := strings.Repeat("Alpha beta gamma delta. ", 100)

	plain := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200})
	lapped := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200, Overlap: 60})
	checkInvariants(t, text, lapped, 200)

	if len(lapped) <= len(plain) {
		t.Errorf("overlap produced %d chunks, not more than the %d without it", len(lapped), len(plain))
	}
	for i := 1; i < len(lapped); i++ {
		if lapped[i].Start >= lapped[i-1].End {
			t.Errorf("chunk %d starts at %d, at or after the previous chunk's end %d -- no overlap",
				i, lapped[i].Start, lapped[i-1].End)
		}
	}
}

// An overlap at or beyond the budget would rewind further than each step
// advances. It must be clamped rather than looping forever.
func TestSplit_AbsurdOverlapStillTerminates(t *testing.T) {
	text := strings.Repeat("Alpha beta gamma delta. ", 100)
	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200, Overlap: 10_000})
	checkInvariants(t, text, got, 200)
	if len(got) == 0 {
		t.Fatal("no chunks")
	}
}

// The contract is that no trailing sliver survives, whether MinBytes gets
// there by merging the pair or by recutting it.
func TestSplit_MinBytesLeavesNoTrailingSliver(t *testing.T) {
	const maxBytes, minBytes = 100, 20

	// Several lengths, so both the merge path (pair fits the budget) and the
	// rebalance path (pair does not) are exercised.
	for _, repeats := range []int{18, 25, 34, 41, 50} {
		text := strings.Repeat("abcde ", repeats) + "xy"

		without := Split("nomic-embed-text", text, SplitOptions{MaxBytes: maxBytes})
		if last := without[len(without)-1]; len(last.Text) >= minBytes {
			continue // no sliver to fix at this length
		}

		with := Split("nomic-embed-text", text, SplitOptions{MaxBytes: maxBytes, MinBytes: minBytes})
		checkInvariants(t, text, with, maxBytes)
		if last := with[len(with)-1]; len(last.Text) < minBytes {
			t.Errorf("repeats=%d: tail chunk is still %d bytes, under the %d floor",
				repeats, len(last.Text), minBytes)
		}
		if want, have := stripSpace(text), stripSpace(concat(with)); want != have {
			t.Errorf("repeats=%d: fixing the tail changed the content", repeats)
		}
	}
}

// The cheap path: when the pair fits the budget together, they become one
// chunk rather than two rebalanced ones.
func TestSplit_MinBytesMergesWhenThePairFits(t *testing.T) {
	text := strings.Repeat("abcde ", 12) + "xy" // 74 bytes, two chunks at 50

	without := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 50})
	with := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 80, MinBytes: 40})
	checkInvariants(t, text, with, 80)
	if len(with) >= len(without) {
		t.Errorf("got %d chunks, want fewer than the %d without MinBytes", len(with), len(without))
	}
}

func TestSplit_UsesTheModelBudgetByDefault(t *testing.T) {
	text := strings.Repeat("Alpha beta gamma delta. ", 1000) // 24000 bytes
	got := Split("nomic-embed-text", text, SplitOptions{})
	checkInvariants(t, text, got, 6000)
	if len(got) < 4 {
		t.Errorf("got %d chunks for 24000 bytes against a 6000-byte budget", len(got))
	}
}

func TestConfig_Limits(t *testing.T) {
	cfg := Config{Model: "nomic-embed-text"}
	if got := cfg.Limits(); got != (Limits{MaxBytes: 6000, MaxTokens: 2000}) {
		t.Errorf("Limits() = %+v, want the registered budget", got)
	}

	cfg.MaxTokens = 512
	if got := cfg.Limits().MaxTokens; got != 512 {
		t.Errorf("MaxTokens = %d, want the override 512", got)
	}
}

func TestTokenBudget(t *testing.T) {
	if got := TokenBudget("nomic-embed-text"); got != 2000 {
		t.Errorf("TokenBudget = %d, want 2000", got)
	}
	if got := TokenBudget("nomic-embed-text:latest"); got != 2000 {
		t.Errorf("tagged variant: TokenBudget = %d, want 2000", got)
	}
	if got := TokenBudget("model-nobody-registered"); got != 0 {
		t.Errorf("unknown model: TokenBudget = %d, want 0", got)
	}
}

// concat joins chunk texts in order.
func concat(chunks []Chunk) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString(c.Text)
	}
	return b.String()
}

// FuzzSplit drives the invariants over arbitrary input. The splitter runs over
// whatever a corpus contains -- markup, binary-ish blobs, mixed scripts -- and
// an infinite loop or a mid-rune cut there is far more expensive to find in
// production than here.
func FuzzSplit(f *testing.F) {
	f.Add("hello world", 10, 0, 0, false)
	f.Add(strings.Repeat("para\n\ngraph ", 20), 30, 5, 3, false)
	f.Add("日本語のテキスト", 5, 2, 0, false)
	f.Add("", 0, 0, 0, false)
	f.Add("# H\n\nbody\n\n## H2\n\nmore body\n", 20, 4, 2, true)
	f.Add("Setext\n===\n\n```\n# fake\n```\n\n## Real\n", 16, 0, 0, true)

	f.Fuzz(func(t *testing.T, text string, maxBytes, overlap, minBytes int, structured bool) {
		// Keep the parameters in a sane range; the point is the text, and
		// enormous values only exercise the allocator.
		if maxBytes < 0 || maxBytes > 1<<16 || overlap < 0 || overlap > 1<<16 || minBytes < 0 || minBytes > 1<<16 {
			t.Skip()
		}
		structure := StructureNone
		if structured {
			structure = StructureMarkdown
		}
		got := Split("nomic-embed-text", text, SplitOptions{
			MaxBytes: maxBytes, Overlap: overlap, MinBytes: minBytes, Structure: structure,
		})
		// Mirror the two adjustments Split documents: an unset budget falls
		// back to the model's, and a sub-rune budget is floored.
		budget := maxBytes
		if budget <= 0 {
			budget = LookupLimits("nomic-embed-text").MaxBytes
		} else {
			budget = max(budget, utf8.UTFMax)
		}
		checkInvariants(t, text, got, budget)
	})
}
