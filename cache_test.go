package embedding

import (
	"context"
	"slices"
	"testing"
)

func TestNormalizeQuery(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"trim", "  hello  ", "hello"},
		{"lowercase", "Hello World", "hello world"},
		{"collapse internal whitespace", "foo   bar\tbaz", "foo bar baz"},
		{"newlines collapse", "foo\n\nbar", "foo bar"},
		{"mixed", "  Foo\tBar  Baz \n", "foo bar baz"},
		{"only whitespace", "   \t\n ", ""},
		{"unicode case", "MÉMOIRE", "mémoire"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeQuery(tt.in); got != tt.want {
				t.Errorf("normalizeQuery(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestCacheKey(t *testing.T) {
	fp := Fingerprint{Model: "nomic-embed-text", Dim: 768}

	// Queries that normalize identically share a key.
	if cacheKey(fp, "Hello World") != cacheKey(fp, "  hello   world ") {
		t.Error("expected normalized queries to produce identical keys")
	}

	// Distinct fingerprints never collide for the same query.
	cases := []struct {
		name string
		a, b Fingerprint
	}{
		{"different model", Fingerprint{Model: "a", Dim: 768}, Fingerprint{Model: "b", Dim: 768}},
		{"different dim", Fingerprint{Model: "a", Dim: 768}, Fingerprint{Model: "a", Dim: 1024}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if cacheKey(tc.a, "query") == cacheKey(tc.b, "query") {
				t.Errorf("expected distinct keys for %+v vs %+v", tc.a, tc.b)
			}
		})
	}

	// The NUL separators must stop the model name from bleeding into the
	// dimension field and forging a collision.
	if cacheKey(Fingerprint{Model: "a", Dim: 1}, "q") ==
		cacheKey(Fingerprint{Model: "a\x001", Dim: 1}, "q") {
		t.Error("separator collision between model and dim components")
	}
}

// countingEmbedder returns deterministic vectors and records every text it is
// asked to embed, so tests can assert which inputs actually reached the backend.
type countingEmbedder struct {
	fp     Fingerprint
	seen   []string
	vecFor func(text string) []float32
}

func (c *countingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		c.seen = append(c.seen, t)
		out[i] = c.vecFor(t)
	}
	return out, nil
}

func (c *countingEmbedder) Model() string            { return c.fp.Model }
func (c *countingEmbedder) Fingerprint() Fingerprint { return c.fp }

func newCountingEmbedder(model string, dim int) *countingEmbedder {
	return &countingEmbedder{
		fp: Fingerprint{Model: model, Dim: dim},
		vecFor: func(text string) []float32 {
			return []float32{float32(len(text))}
		},
	}
}

func TestQueryCacheSingleHits(t *testing.T) {
	e := newCountingEmbedder("m", 4)
	c := NewQueryCache(8)

	first, err := c.Single(context.Background(), e, "Hello World")
	if err != nil {
		t.Fatalf("Single: %v", err)
	}
	// A normalized variant must hit the same entry without re-embedding.
	second, err := c.Single(context.Background(), e, "  hello   world ")
	if err != nil {
		t.Fatalf("Single: %v", err)
	}
	if !slices.Equal(first, second) {
		t.Errorf("cached vector %v != original %v", second, first)
	}
	if len(e.seen) != 1 {
		t.Errorf("expected 1 embed call, got %d: %v", len(e.seen), e.seen)
	}
}

func TestQueryCacheEmbedMixedHitsAndMisses(t *testing.T) {
	e := newCountingEmbedder("m", 4)
	c := NewQueryCache(8)
	ctx := context.Background()

	// Warm "a" and "c".
	if _, err := c.Single(ctx, e, "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Single(ctx, e, "c"); err != nil {
		t.Fatal(err)
	}
	e.seen = nil

	got, err := c.Embed(ctx, e, []string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("expected 4 results, got %d", len(got))
	}
	for i, v := range got {
		if v == nil {
			t.Errorf("result %d is nil", i)
		}
	}
	// Only the misses ("b", "d") should have reached the backend.
	if !slices.Equal(e.seen, []string{"b", "d"}) {
		t.Errorf("embedded %v, want [b d]", e.seen)
	}
}

func TestQueryCacheEviction(t *testing.T) {
	e := newCountingEmbedder("m", 4)
	c := NewQueryCache(2)
	ctx := context.Background()

	if _, err := c.Single(ctx, e, "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Single(ctx, e, "b"); err != nil {
		t.Fatal(err)
	}
	// Touch "a" so "b" is the least-recently-used entry.
	if _, err := c.Single(ctx, e, "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Single(ctx, e, "c"); err != nil { // evicts "b"
		t.Fatal(err)
	}

	// Query survivors first (a hit re-adds and would itself evict), then the
	// evicted entry last.
	e.seen = nil
	if _, err := c.Single(ctx, e, "a"); err != nil { // still cached → no embed
		t.Fatal(err)
	}
	if _, err := c.Single(ctx, e, "c"); err != nil { // still cached → no embed
		t.Fatal(err)
	}
	if len(e.seen) != 0 {
		t.Errorf("survivors re-embedded %v, want none", e.seen)
	}
	if _, err := c.Single(ctx, e, "b"); err != nil { // evicted → re-embed
		t.Fatal(err)
	}
	if !slices.Equal(e.seen, []string{"b"}) {
		t.Errorf("re-embedded %v, want [b] (b should have been evicted)", e.seen)
	}
}

func TestQueryCacheFingerprintScoping(t *testing.T) {
	c := NewQueryCache(8)
	ctx := context.Background()

	m4 := newCountingEmbedder("m", 4)
	m8 := newCountingEmbedder("m", 8) // same model name, different dimension

	if _, err := c.Single(ctx, m4, "shared"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Single(ctx, m8, "shared"); err != nil {
		t.Fatal(err)
	}
	// Different fingerprints must not share an entry: both embedders see it.
	if len(m4.seen) != 1 || len(m8.seen) != 1 {
		t.Errorf("expected each fingerprint to embed once; m4=%v m8=%v", m4.seen, m8.seen)
	}
}

func TestQueryCacheNilDisabled(t *testing.T) {
	c := NewQueryCache(0)
	if c != nil {
		t.Fatalf("expected nil cache for capacity 0, got %v", c)
	}
	if c.Len() != 0 {
		t.Errorf("nil cache Len() = %d, want 0", c.Len())
	}

	e := newCountingEmbedder("m", 4)
	ctx := context.Background()

	// Methods on a nil cache must fall through to the embedder every time.
	if _, err := c.Single(ctx, e, "q"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Single(ctx, e, "q"); err != nil {
		t.Fatal(err)
	}
	if len(e.seen) != 2 {
		t.Errorf("nil cache should not cache; embed calls = %d, want 2", len(e.seen))
	}

	got, err := c.Embed(ctx, e, []string{"x", "y"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Errorf("Embed returned %d results, want 2", len(got))
	}
}
