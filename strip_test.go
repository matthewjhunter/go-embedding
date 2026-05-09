package embedding

import (
	"strings"
	"testing"
)

func TestStripNonsemantic(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "bare URL removed",
			in:   "Read more at https://example.com/path?q=1 for details.",
			want: "Read more at for details.",
		},
		{
			name: "multiple URLs",
			in:   "See http://a.com and https://b.org/page for info.",
			want: "See and for info.",
		},
		{
			name: "mailto link",
			in:   "Email mailto:user@example.com or call.",
			want: "Email or call.",
		},
		{
			name: "ftp URL",
			in:   "Mirror at ftp://files.example.com/pub/ exists.",
			want: "Mirror at exists.",
		},
		{
			name: "URL with trailing punctuation",
			in:   "(See https://example.com/x), then continue.",
			want: "(See ), then continue.",
		},
		{
			name: "markdown link preserves text",
			in:   "Check the [official docs](https://example.com/docs) for details.",
			want: "Check the official docs for details.",
		},
		{
			name: "markdown image preserves alt",
			in:   "Diagram: ![architecture overview](https://example.com/img.png) explains it.",
			want: "Diagram: architecture overview explains it.",
		},
		{
			name: "markdown image without alt is dropped",
			in:   "Picture: ![](https://example.com/img.png) and prose.",
			want: "Picture: and prose.",
		},
		{
			name: "html tags stripped",
			in:   "Some <p>paragraph</p> and <a href='x'>link</a> text.",
			want: "Some paragraph and link text.",
		},
		{
			name: "whitespace collapsed",
			in:   "lots   of    spaces\there",
			want: "lots of spaces here",
		},
		{
			name: "multiple newlines collapsed",
			in:   "para one.\n\n\n\n\npara two.",
			want: "para one.\n\npara two.",
		},
		{
			name: "single newlines preserved",
			in:   "line one\nline two\nline three",
			want: "line one\nline two\nline three",
		},
		{
			name: "leading/trailing whitespace trimmed",
			in:   "   \n\n  hello world  \n\n   ",
			want: "hello world",
		},
		{
			name: "snake_case identifiers preserved (no emphasis stripping)",
			in:   "The function get_user_id returns the *current* user.",
			want: "The function get_user_id returns the *current* user.",
		},
		{
			name: "no-op on plain prose",
			in:   "This is just normal text with no markup or URLs.",
			want: "This is just normal text with no markup or URLs.",
		},
		{
			name: "empty input",
			in:   "",
			want: "",
		},
		{
			name: "URL inside HTML attribute (combined cleanup)",
			in:   `Click <a href="https://example.com">here</a> now.`,
			want: "Click here now.",
		},
		{
			name: "RSS-style article snippet",
			in: `<p>Read the full article at <a href="https://example.com/2026/05/post">example.com</a> ` +
				`for more, or see the [research paper](https://arxiv.org/abs/2026.12345) directly. ` +
				`Tags: ![icon](https://cdn.example.com/tag.svg) <strong>policy</strong>.</p>`,
			want: "Read the full article at example.com for more, or see the research paper directly. Tags: icon policy.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StripNonsemantic(tt.in)
			if got != tt.want {
				t.Errorf("StripNonsemantic(%q):\n  got:  %q\n  want: %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestStripNonsemanticReducesBytes(t *testing.T) {
	// Verify the helper actually frees up budget for typical RSS content.
	// A sanity check that the regex passes don't accidentally bloat input.
	rssBody := strings.Repeat(
		`<p>This article was originally published at `+
			`<a href="https://example.com/path/to/post-with-long-slug">example.com</a>. `+
			`See also the [related research](https://arxiv.org/abs/9999.99999) and `+
			`![social card](https://cdn.example.com/cards/post-12345.jpg). `+
			`Read more on <a href="https://blog.example.com/related">our blog</a>.</p>`,
		5)

	stripped := StripNonsemantic(rssBody)
	if len(stripped) >= len(rssBody) {
		t.Errorf("StripNonsemantic should reduce typical RSS content; before=%d after=%d", len(rssBody), len(stripped))
	}
	// Ratio sanity: stripped content should be at most ~70% of original
	// for this URL-heavy synthetic input.
	if ratio := float64(len(stripped)) / float64(len(rssBody)); ratio > 0.70 {
		t.Errorf("expected stripping to remove >30%% of bytes for URL-heavy content, got ratio=%.2f", ratio)
	}
}
