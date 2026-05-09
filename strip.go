package embedding

import (
	"regexp"
	"strings"
)

// Stripping regexes are package-level vars so they compile once. Order of
// application in StripNonsemantic matters — markdown link syntax has to
// be reduced to its visible text BEFORE the bare URL pattern runs, or the
// URL inside [text](url) gets stripped while leaving the brackets behind.
var (
	// urlRegex matches absolute URLs that callers commonly inline as bare
	// text: http(s), ftp, mailto. The character class for the tail
	// excludes whitespace plus a few common closing punctuation marks
	// (), >, ]) that often follow a URL — including these in the match
	// produces stray punctuation in the output.
	urlRegex = regexp.MustCompile(`(?i)\b(?:https?|ftp|mailto):[^\s)>\]]+`)

	// htmlTagRegex strips HTML markup. Defense-in-depth: most feed
	// readers parse HTML to plain text upstream, but some leak balanced
	// or stray tags through — particularly for feeds that wrap each
	// paragraph in <p>...</p> without a parser. This is intentionally
	// permissive (greedy any-non-> match) and won't handle nested
	// CDATA-style escapes correctly; for embedder input that's fine.
	htmlTagRegex = regexp.MustCompile(`<[^>]+>`)

	// mdImageRegex matches Markdown image syntax: ![alt](url). The alt
	// text is preserved as the visible content; the URL is dropped.
	mdImageRegex = regexp.MustCompile(`!\[([^\]]*)\]\([^)]*\)`)

	// mdLinkRegex matches Markdown link syntax: [text](url). The visible
	// link text is preserved; the URL is dropped. Must run AFTER
	// mdImageRegex so the leading "!" of an image isn't left orphaned.
	mdLinkRegex = regexp.MustCompile(`\[([^\]]+)\]\([^)]*\)`)

	// wsRunRegex collapses runs of horizontal whitespace (spaces, tabs)
	// to a single space. Newlines are preserved by virtue of not being
	// in the character class — paragraph structure carries some
	// semantic signal.
	wsRunRegex = regexp.MustCompile(`[ \t]+`)

	// multiNewlineRegex collapses 3+ consecutive newlines to 2 (single
	// blank line). Runs of blank lines are pure formatting noise.
	multiNewlineRegex = regexp.MustCompile(`\n{3,}`)
)

// StripNonsemantic removes content from text that is high-byte/high-token
// cost but low-meaning for a semantic embedder. The aim is to fit more
// real prose into a model's context window without burning room on
// URLs, HTML tags, or markdown formatting that the embedder treats as
// noise tokens.
//
// Stripped:
//   - Bare URLs (http/https/ftp/mailto): removed entirely
//   - HTML tags: removed (content between tags preserved)
//   - Markdown image syntax ![alt](url): replaced by alt text
//   - Markdown link syntax [text](url): replaced by text
//   - Runs of horizontal whitespace: collapsed to single space
//   - 3+ consecutive newlines: collapsed to a single blank line
//
// Preserved:
//   - All other characters, paragraph structure (single newlines), and
//     punctuation. Markdown emphasis markers (*, _, **) are NOT stripped —
//     they're 1-2 bytes each, not meaningful headroom, and removing them
//     correctly across snake_case identifiers and emphasis pairs is
//     more error-prone than the savings justify.
//
// This is a lossy transform. Don't apply it to inputs where URLs or
// markup carry meaning (e.g. a search-indexed README where the user
// might query "github.com/owner/repo"). For RSS article bodies feeding
// a clustering or retrieval embedder, the loss is intentional — the
// embedder learns better signal from prose than from URL fragments.
func StripNonsemantic(text string) string {
	// Markdown link/image syntax first: the captured group becomes the
	// visible content, so subsequent URL stripping doesn't see the
	// already-extracted URL.
	text = mdImageRegex.ReplaceAllString(text, "$1")
	text = mdLinkRegex.ReplaceAllString(text, "$1")
	// Bare URLs that weren't wrapped in markdown link syntax.
	text = urlRegex.ReplaceAllString(text, "")
	// HTML tags after URL stripping in case any URL appeared inside an
	// attribute (the URL match runs first, then the tag remnant gets
	// cleaned up here).
	text = htmlTagRegex.ReplaceAllString(text, "")
	// Whitespace cleanup last so runs left behind by stripping (e.g.
	// "see <a href=...>here</a> for details" → "see  for details") are
	// collapsed to a single space.
	text = wsRunRegex.ReplaceAllString(text, " ")
	text = multiNewlineRegex.ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}
