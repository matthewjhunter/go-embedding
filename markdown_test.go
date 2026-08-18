package embedding

import (
	"reflect"
	"strings"
	"testing"
)

func TestScanMarkdown_ATXHeadings(t *testing.T) {
	text := "# Title\n\nbody text\n\n## Section\n\nmore\n\n###### Deep\n"
	got := scanMarkdown(text)

	want := []struct {
		level int
		title string
	}{
		{1, "Title"},
		{2, "Section"},
		{6, "Deep"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d headings, want %d: %+v", len(got), len(want), got)
	}
	for i, w := range want {
		if got[i].Level != w.level || got[i].Title != w.title {
			t.Errorf("heading %d = (%d, %q), want (%d, %q)", i, got[i].Level, got[i].Title, w.level, w.title)
		}
		if text[got[i].Start] != '#' {
			t.Errorf("heading %d Start=%d does not point at the heading line", i, got[i].Start)
		}
	}
}

func TestScanMarkdown_ATXEdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		text  string
		want  string // expected title, or "" for no heading
		level int
	}{
		{name: "trailing hashes are closing syntax", text: "## Section ##\n", want: "Section", level: 2},
		{name: "indented up to three spaces", text: "   ## Section\n", want: "Section", level: 2},
		{name: "four spaces is a code block", text: "    ## Section\n"},
		{name: "hash without a space is not a heading", text: "#hashtag\n"},
		{name: "seven hashes is not a heading", text: "####### too deep\n"},
		{name: "empty heading", text: "##\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := scanMarkdown(tt.text)
			if tt.want == "" {
				if len(got) != 0 {
					t.Fatalf("got %+v, want no headings", got)
				}
				return
			}
			if len(got) != 1 {
				t.Fatalf("got %d headings, want 1: %+v", len(got), got)
			}
			if got[0].Title != tt.want || got[0].Level != tt.level {
				t.Errorf("got (%d, %q), want (%d, %q)", got[0].Level, got[0].Title, tt.level, tt.want)
			}
		})
	}
}

func TestScanMarkdown_SetextHeadings(t *testing.T) {
	text := "Title Line\n=====\n\nsome body\n\nSection Line\n-----\n\nmore body\n"
	got := scanMarkdown(text)
	if len(got) != 2 {
		t.Fatalf("got %d headings, want 2: %+v", len(got), got)
	}
	if got[0].Level != 1 || got[0].Title != "Title Line" {
		t.Errorf("got (%d, %q), want (1, \"Title Line\")", got[0].Level, got[0].Title)
	}
	if got[1].Level != 2 || got[1].Title != "Section Line" {
		t.Errorf("got (%d, %q), want (2, \"Section Line\")", got[1].Level, got[1].Title)
	}
	// The heading starts at its text line, not at the underline.
	if !strings.HasPrefix(text[got[0].Start:], "Title Line") {
		t.Errorf("setext Start=%d does not point at the title line", got[0].Start)
	}
}

// A dashed line with no paragraph above it is a thematic break, not a heading.
func TestScanMarkdown_ThematicBreakIsNotAHeading(t *testing.T) {
	for _, text := range []string{"para\n\n---\n\nmore\n", "---\n\nbody\n"} {
		if got := scanMarkdown(text); len(got) != 0 {
			t.Errorf("scanMarkdown(%q) found %+v, want no headings", text, got)
		}
	}
}

// Markdown inside a fenced code block is code, not structure. Shell comments
// and diff markers would otherwise register as headings.
func TestScanMarkdown_IgnoresFencedCode(t *testing.T) {
	text := "# Real\n\n```sh\n# not a heading\necho hi\n```\n\n## Also Real\n\n~~~\n### nor this\n~~~\n"
	got := scanMarkdown(text)
	if len(got) != 2 {
		t.Fatalf("got %d headings, want 2: %+v", len(got), got)
	}
	if got[0].Title != "Real" || got[1].Title != "Also Real" {
		t.Errorf("got %q and %q", got[0].Title, got[1].Title)
	}
}

func TestScanMarkdown_UnclosedFenceSwallowsTheRest(t *testing.T) {
	// An unterminated fence means everything after it is code, which is what
	// a renderer would do too.
	text := "# Real\n\n```\n# not a heading\n\n## nor this\n"
	got := scanMarkdown(text)
	if len(got) != 1 || got[0].Title != "Real" {
		t.Fatalf("got %+v, want just the one real heading", got)
	}
}

func TestSplit_HeadingPathTracksTheStack(t *testing.T) {
	body := strings.Repeat("filler words here. ", 8) // ~152 bytes per section

	text := "# Doc\n\n" + body +
		"\n\n## Alpha\n\n" + body +
		"\n\n### Alpha Deep\n\n" + body +
		"\n\n## Beta\n\n" + body

	got := Split("nomic-embed-text", text, SplitOptions{
		MaxBytes: 200, Structure: StructureMarkdown,
	})
	checkInvariants(t, text, got, 200)

	// Every chunk must know its section, and "## Beta" must have popped the
	// "### Alpha Deep" level rather than nesting under it.
	var sawDeep, sawBeta bool
	for _, c := range got {
		if len(c.Headings) == 0 {
			t.Errorf("chunk %d at %d has no heading path: %q", c.Ordinal, c.Start, c.Text)
			continue
		}
		if c.Headings[0] != "Doc" {
			t.Errorf("chunk %d: path starts with %q, want \"Doc\"", c.Ordinal, c.Headings[0])
		}
		switch last := c.Headings[len(c.Headings)-1]; last {
		case "Alpha Deep":
			sawDeep = true
			if want := []string{"Doc", "Alpha", "Alpha Deep"}; !reflect.DeepEqual(c.Headings, want) {
				t.Errorf("chunk %d: path %v, want %v", c.Ordinal, c.Headings, want)
			}
		case "Beta":
			sawBeta = true
			if want := []string{"Doc", "Beta"}; !reflect.DeepEqual(c.Headings, want) {
				t.Errorf("chunk %d: path %v, want %v -- a sibling heading must pop the deeper level",
					c.Ordinal, c.Headings, want)
			}
		}
	}
	if !sawDeep || !sawBeta {
		t.Errorf("test did not reach the sections it checks (deep=%v beta=%v)", sawDeep, sawBeta)
	}
}

// A chunk that begins with a heading is inside that section, not the one
// before it.
func TestSplit_ChunkStartingAtAHeadingIsInsideIt(t *testing.T) {
	body := strings.Repeat("filler words here. ", 10)
	text := "# Doc\n\n" + body + "\n\n## Alpha\n\n" + body

	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 220, Structure: StructureMarkdown})
	for _, c := range got {
		if !strings.HasPrefix(c.Text, "## Alpha") {
			continue
		}
		want := []string{"Doc", "Alpha"}
		if !reflect.DeepEqual(c.Headings, want) {
			t.Errorf("chunk starting at the heading has path %v, want %v", c.Headings, want)
		}
		return
	}
	t.Skip("no chunk began at the heading; boundary choice changed")
}

// Breaking just before a heading is better than breaking three lines into the
// section, so a heading outranks a paragraph as a boundary.
func TestSplit_PrefersHeadingBoundaries(t *testing.T) {
	body := strings.Repeat("filler words here. ", 6)
	text := "## One\n\n" + body + "\n\n## Two\n\n" + body + "\n\n## Three\n\n" + body

	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 160, Structure: StructureMarkdown})
	checkInvariants(t, text, got, 160)

	starts := 0
	for _, c := range got {
		if strings.HasPrefix(c.Text, "## ") {
			starts++
		}
	}
	if starts < 2 {
		t.Errorf("only %d chunks begin at a heading; headings are not being preferred", starts)
	}
	for _, c := range got {
		if i := strings.Index(c.Text, "\n## "); i > 0 {
			t.Errorf("chunk %d runs past a heading it could have broken at: %q", c.Ordinal, c.Text)
		}
	}
}

// StructureNone is the default and must behave exactly as before.
func TestSplit_StructureNoneIgnoresMarkdown(t *testing.T) {
	body := strings.Repeat("filler words here. ", 8)
	text := "# Doc\n\n" + body + "\n\n## Alpha\n\n" + body

	plain := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200})
	explicit := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200, Structure: StructureNone})

	if !reflect.DeepEqual(plain, explicit) {
		t.Error("StructureNone differs from the zero value")
	}
	for _, c := range plain {
		if c.Headings != nil {
			t.Errorf("chunk %d carries a heading path without StructureMarkdown: %v", c.Ordinal, c.Headings)
		}
	}
}

func TestSplit_HeadingPathBeforeAnyHeading(t *testing.T) {
	text := strings.Repeat("preamble text. ", 30) + "\n\n# Later\n\n" + strings.Repeat("body. ", 30)
	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200, Structure: StructureMarkdown})
	if len(got[0].Headings) != 0 {
		t.Errorf("preamble chunk has path %v, want none", got[0].Headings)
	}
}

// The heading path is metadata; it must not be spliced into Text, or the
// offsets stop addressing the source.
func TestSplit_HeadingPathDoesNotAlterText(t *testing.T) {
	body := strings.Repeat("filler words here. ", 8)
	text := "# Doc\n\n" + body + "\n\n## Alpha\n\n" + body

	got := Split("nomic-embed-text", text, SplitOptions{MaxBytes: 200, Structure: StructureMarkdown})
	checkInvariants(t, text, got, 200)
	if want, have := stripSpace(text), stripSpace(concat(got)); want != have {
		t.Error("structure-aware splitting changed the content")
	}
}
