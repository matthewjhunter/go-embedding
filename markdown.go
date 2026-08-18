package embedding

import "strings"

// Structure tells Split how much of the input's shape to understand. The zero
// value treats input as flat text.
//
// Markdown is the only structured format supported, deliberately. It is
// trivial to scan, structured enough to break on, and every other format
// (HTML, PDF text, docs) converts to it -- so callers convert once rather
// than this library growing a parser per format.
type Structure int

const (
	// StructureNone treats the input as flat text: no heading paths, and
	// boundaries chosen purely by paragraph, sentence, and word.
	StructureNone Structure = iota
	// StructureMarkdown reads ATX and setext headings (ignoring anything
	// inside a fenced code block), records the heading path in force on each
	// chunk, and prefers to break where a section does.
	StructureMarkdown
)

// heading is one heading found in a markdown document.
type heading struct {
	// Start is the byte offset of the first character of the heading, which
	// for a setext heading is its text line rather than the underline.
	Start int
	// Level is 1-6.
	Level int
	// Title is the heading text, with syntax stripped.
	Title string
	// path is the full heading stack in force from this heading onward,
	// outermost first.
	path []string
}

// maxATXLevel is the deepest ATX heading markdown defines. Seven hashes is a
// paragraph beginning with hashes.
const maxATXLevel = 6

// scanMarkdown finds the headings in text, in document order, each carrying
// the heading path in force from it onward.
//
// Fenced code blocks are skipped: a shell comment or a diff marker inside one
// would otherwise register as a heading, which is a common way for a
// structure-aware splitter to produce nonsense on technical writing. An
// unterminated fence swallows the rest of the document, matching what a
// renderer does.
func scanMarkdown(text string) []heading {
	var (
		out       []heading
		stack     []heading
		fenceChar byte
		fenceLen  int
		prevStart = -1
		prevLine  string
	)

	for offset := 0; offset <= len(text); {
		lineEnd := strings.IndexByte(text[offset:], '\n')
		var line string
		next := len(text) + 1
		if lineEnd < 0 {
			line = text[offset:]
		} else {
			line = text[offset : offset+lineEnd]
			next = offset + lineEnd + 1
		}

		if fenceLen > 0 {
			if isFenceClose(line, fenceChar, fenceLen) {
				fenceLen = 0
			}
			prevStart, prevLine = -1, ""
			offset = next
			continue
		}
		if ch, n, ok := fenceOpen(line); ok {
			fenceChar, fenceLen = ch, n
			prevStart, prevLine = -1, ""
			offset = next
			continue
		}

		if level, title, ok := atxHeading(line); ok {
			out = append(out, pushHeading(&stack, heading{Start: offset, Level: level, Title: title}))
			prevStart, prevLine = -1, ""
			offset = next
			continue
		}

		// A setext underline promotes the paragraph line above it. Without a
		// paragraph line, a run of dashes is a thematic break.
		if level, ok := setextUnderline(line); ok && prevStart >= 0 {
			out = append(out, pushHeading(&stack, heading{
				Start: prevStart, Level: level, Title: strings.TrimSpace(prevLine),
			}))
			prevStart, prevLine = -1, ""
			offset = next
			continue
		}

		if strings.TrimSpace(line) == "" {
			prevStart, prevLine = -1, ""
		} else {
			prevStart, prevLine = offset, line
		}
		offset = next
	}
	return out
}

// pushHeading applies h to the stack -- a heading closes every section at or
// below its own level -- and returns it with its path filled in.
func pushHeading(stack *[]heading, h heading) heading {
	s := *stack
	for len(s) > 0 && s[len(s)-1].Level >= h.Level {
		s = s[:len(s)-1]
	}
	s = append(s, h)
	*stack = s

	path := make([]string, len(s))
	for i, e := range s {
		path[i] = e.Title
	}
	h.path = path
	return h
}

// atxHeading matches "### Title", allowing up to three leading spaces (four
// makes it an indented code block) and stripping any closing run of hashes.
// A hash run not followed by a space is not a heading, so "#hashtag" is prose.
func atxHeading(line string) (level int, title string, ok bool) {
	i := 0
	for i < len(line) && i < 4 && line[i] == ' ' {
		i++
	}
	if i == 4 {
		return 0, "", false
	}
	start := i
	for i < len(line) && line[i] == '#' {
		i++
	}
	level = i - start
	if level == 0 || level > maxATXLevel {
		return 0, "", false
	}
	rest := line[i:]
	if rest != "" && rest[0] != ' ' && rest[0] != '\t' {
		return 0, "", false
	}
	title = strings.TrimSpace(strings.TrimRight(strings.TrimSpace(rest), "#"))
	if title == "" {
		// CommonMark allows an empty heading, but an empty title contributes
		// nothing to a heading path and would show up as "" in the middle of
		// one. Treat it as prose.
		return 0, "", false
	}
	return level, title, true
}

// setextUnderline matches a run of "=" (level 1) or "-" (level 2).
func setextUnderline(line string) (level int, ok bool) {
	s := strings.TrimRight(strings.TrimLeft(line, " "), " \t")
	if len(s) == 0 {
		return 0, false
	}
	switch s[0] {
	case '=':
		level = 1
	case '-':
		level = 2
	default:
		return 0, false
	}
	for i := range len(s) {
		if s[i] != s[0] {
			return 0, false
		}
	}
	return level, true
}

// minFenceLen is the shortest run of backticks or tildes that opens a fence.
const minFenceLen = 3

// fenceOpen reports whether line opens a fenced code block, and with what.
func fenceOpen(line string) (char byte, length int, ok bool) {
	s := strings.TrimLeft(line, " ")
	if len(s) < minFenceLen || (s[0] != '`' && s[0] != '~') {
		return 0, 0, false
	}
	n := 0
	for n < len(s) && s[n] == s[0] {
		n++
	}
	if n < minFenceLen {
		return 0, 0, false
	}
	return s[0], n, true
}

// isFenceClose reports whether line closes a fence opened with char and length.
func isFenceClose(line string, char byte, length int) bool {
	c, n, ok := fenceOpen(line)
	return ok && c == char && n >= length
}

// headingPathAt returns the heading path in force at byte offset pos, or nil
// when pos precedes every heading. A heading beginning exactly at pos counts:
// a chunk that starts with "## Rollback" is inside that section, not the one
// before it.
func headingPathAt(headings []heading, pos int) []string {
	// Headings are in document order, so walk back from the last one at or
	// before pos. Documents have few headings relative to chunks; a linear
	// scan from the end is simpler than an index and fast enough.
	for i := len(headings) - 1; i >= 0; i-- {
		if headings[i].Start <= pos {
			return headings[i].path
		}
	}
	return nil
}
