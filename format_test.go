package embedding

import (
	"strings"
	"testing"
)

func TestFormatRecord_FieldsAndBody(t *testing.T) {
	got := FormatRecord(
		[]Field{
			{"feed", "Schneier on Security"},
			{"author", "Bruce Schneier"},
			{"categories", "cryptography, surveillance"},
			{"title", "How AI Will Change Cyber Defense"},
		},
		"Full article body goes here.",
	)
	want := "feed: Schneier on Security\n" +
		"author: Bruce Schneier\n" +
		"categories: cryptography, surveillance\n" +
		"title: How AI Will Change Cyber Defense\n" +
		"\n" +
		"Full article body goes here."
	if got != want {
		t.Errorf("FormatRecord:\n got %q\nwant %q", got, want)
	}
}

func TestFormatRecord_OrderPreserved(t *testing.T) {
	// Fields must appear in slice order, not alphabetical or any other.
	// The embedder learns recurring positional patterns, so stability
	// across calls is essential.
	got := FormatRecord(
		[]Field{
			{"z-key", "z"},
			{"a-key", "a"},
			{"m-key", "m"},
		},
		"body",
	)
	want := "z-key: z\na-key: a\nm-key: m\n\nbody"
	if got != want {
		t.Errorf("order not preserved:\n got %q\nwant %q", got, want)
	}
}

func TestFormatRecord_SkipEmptyValues(t *testing.T) {
	got := FormatRecord(
		[]Field{
			{"feed", "X"},
			{"author", ""}, // skipped
			{"title", "T"},
		},
		"body",
	)
	want := "feed: X\ntitle: T\n\nbody"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestFormatRecord_SkipEmptyKeys(t *testing.T) {
	got := FormatRecord(
		[]Field{
			{"", "stray-value"},
			{"feed", "X"},
		},
		"body",
	)
	want := "feed: X\n\nbody"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestFormatRecord_BodyOnly(t *testing.T) {
	got := FormatRecord(nil, "just the body")
	if got != "just the body" {
		t.Errorf("got %q, want %q", got, "just the body")
	}
}

func TestFormatRecord_FieldsOnly(t *testing.T) {
	// Trailing whitespace is undesirable when there is no body.
	got := FormatRecord([]Field{{"k", "v"}}, "")
	if got != "k: v" {
		t.Errorf("got %q, want %q (no trailing newline)", got, "k: v")
	}
}

func TestFormatRecord_Empty(t *testing.T) {
	if got := FormatRecord(nil, ""); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

func TestLookupTaskPrompter_UnknownModel(t *testing.T) {
	// Unknown models get a pass-through prompter — text returned unchanged.
	p := LookupTaskPrompter("entirely-fictional-model")
	got := p(TaskClustering, "hello")
	if got != "hello" {
		t.Errorf("unknown model prompter modified text: got %q", got)
	}
}

func TestLookupTaskPrompter_TagSuffixFallsBackToBase(t *testing.T) {
	// Same fall-back as LookupLimits: tagged variants share the base
	// model's task convention.
	cases := []string{
		"nomic-embed-text:latest",
		"nomic-embed-text:q4_0",
		"embeddinggemma:300m",
	}
	for _, model := range cases {
		t.Run(model, func(t *testing.T) {
			p := LookupTaskPrompter(model)
			got := p(TaskClustering, "x")
			if !strings.Contains(got, "clustering") {
				t.Errorf("tagged variant %q lost task prefix: got %q", model, got)
			}
			if !strings.HasSuffix(got, "\nx") {
				t.Errorf("tagged variant %q lost the prefix-body newline separator: got %q", model, got)
			}
		})
	}
}

func TestNomicTaskPrompter(t *testing.T) {
	cases := []struct {
		task Task
		want string
	}{
		{TaskRetrievalDocument, "search_document:\nhello"},
		{TaskRetrievalQuery, "search_query:\nhello"},
		{TaskClassification, "classification:\nhello"},
		{TaskClustering, "clustering:\nhello"},
		// Unsupported tasks pass through unchanged.
		{TaskSimilarity, "hello"},
		{TaskQuestionAnswering, "hello"},
		{TaskCodeRetrieval, "hello"},
	}
	for _, tc := range cases {
		t.Run(string(tc.task), func(t *testing.T) {
			got := nomicTaskPrompter(tc.task, "hello")
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestGemmaTaskPrompter(t *testing.T) {
	cases := []struct {
		task Task
		want string
	}{
		{TaskRetrievalDocument, "title: none | text:\nhello"},
		{TaskRetrievalQuery, "task: search result | query:\nhello"},
		{TaskClustering, "task: clustering | query:\nhello"},
		{TaskClassification, "task: classification | query:\nhello"},
		{TaskSimilarity, "task: sentence similarity | query:\nhello"},
		{TaskQuestionAnswering, "task: question answering | query:\nhello"},
		{TaskFactChecking, "task: fact checking | query:\nhello"},
		{TaskCodeRetrieval, "task: code retrieval | query:\nhello"},
	}
	for _, tc := range cases {
		t.Run(string(tc.task), func(t *testing.T) {
			got := gemmaTaskPrompter(tc.task, "hello")
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestFormatForTask(t *testing.T) {
	// Sanity check that FormatForTask routes through the registry.
	got := FormatForTask("nomic-embed-text", TaskClustering, "x")
	if got != "clustering:\nx" {
		t.Errorf("got %q, want %q", got, "clustering:\nx")
	}
	got = FormatForTask("embeddinggemma", TaskClustering, "x")
	if got != "task: clustering | query:\nx" {
		t.Errorf("got %q, want %q", got, "task: clustering | query:\nx")
	}
}

func TestFormatRecordForTask(t *testing.T) {
	// The full assembly: task prefix wraps the field-labels + body block.
	// The prefix ends with a newline so field labels start at column 0
	// below the prefix, giving the embedder stable positional patterns.
	got := FormatRecordForTask(
		"nomic-embed-text:latest",
		TaskClustering,
		[]Field{{"feed", "Schneier on Security"}, {"author", "Bruce Schneier"}},
		"Full body.",
	)
	want := "clustering:\nfeed: Schneier on Security\nauthor: Bruce Schneier\n\nFull body."
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestRegisterTaskPrompter_OverrideAndRemove(t *testing.T) {
	const name = "test-only-custom-model"
	t.Cleanup(func() { RegisterTaskPrompter(name, nil) })

	RegisterTaskPrompter(name, func(task Task, text string) string {
		return "CUSTOM[" + string(task) + "]: " + text
	})
	if got := FormatForTask(name, TaskClustering, "x"); got != "CUSTOM[clustering]: x" {
		t.Errorf("after RegisterTaskPrompter: got %q", got)
	}

	RegisterTaskPrompter(name, nil)
	if got := FormatForTask(name, TaskClustering, "x"); got != "x" {
		t.Errorf("after nil RegisterTaskPrompter (remove): got %q, want pass-through", got)
	}
}

func TestRegisterTaskPrompter_ExactWinsOverTagFallback(t *testing.T) {
	// If a tagged variant has its own registration, exact match wins
	// over base-name fallback.
	const tagged = "nomic-embed-text:custom"
	t.Cleanup(func() { RegisterTaskPrompter(tagged, nil) })

	RegisterTaskPrompter(tagged, func(_ Task, text string) string {
		return "TAGGED: " + text
	})
	if got := FormatForTask(tagged, TaskClustering, "x"); got != "TAGGED: x" {
		t.Errorf("explicit tagged registration not winning: got %q", got)
	}
	// Bare name still goes through nomic.
	if got := FormatForTask("nomic-embed-text", TaskClustering, "x"); got != "clustering:\nx" {
		t.Errorf("bare name unaffected by tagged registration: got %q", got)
	}
}

func TestFormatRetrievalDocument_GemmaWithTitle(t *testing.T) {
	got := FormatRetrievalDocument(
		"embeddinggemma",
		"My Real Title",
		[]Field{{"feed", "X"}, {"author", "Y"}},
		"Body.",
	)
	want := "title: My Real Title | text:\nfeed: X\nauthor: Y\n\nBody."
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestFormatRetrievalDocument_GemmaEmptyTitle(t *testing.T) {
	// Empty title falls back to the documented default ("none") so the
	// model still sees a syntactically valid prompt.
	got := FormatRetrievalDocument(
		"embeddinggemma",
		"",
		[]Field{{"feed", "X"}},
		"Body.",
	)
	want := "title: none | text:\nfeed: X\n\nBody."
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestFormatRetrievalDocument_GemmaTaggedVariant(t *testing.T) {
	// Tag-suffix fallback applies to document prompters too.
	got := FormatRetrievalDocument(
		"embeddinggemma:300m",
		"T",
		nil,
		"Body.",
	)
	want := "title: T | text:\nBody."
	if got != want {
		t.Errorf("tagged variant lost slot: got %q\nwant %q", got, want)
	}
}

func TestFormatRetrievalDocument_NomicNoSlot(t *testing.T) {
	// nomic-embed-text has no structural title slot. Title is promoted
	// to a leading field, then the standard search_document prefix
	// wraps the whole thing.
	got := FormatRetrievalDocument(
		"nomic-embed-text",
		"My Title",
		[]Field{{"feed", "X"}},
		"Body.",
	)
	want := "search_document:\ntitle: My Title\nfeed: X\n\nBody."
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestFormatRetrievalDocument_NomicEmptyTitleNoField(t *testing.T) {
	// No title supplied → no title field added (caller's existing
	// fields are unchanged).
	got := FormatRetrievalDocument(
		"nomic-embed-text",
		"",
		[]Field{{"feed", "X"}},
		"Body.",
	)
	want := "search_document:\nfeed: X\n\nBody."
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestFormatRetrievalDocument_UnknownModelFallsThrough(t *testing.T) {
	// No DocumentPrompter, no TaskPrompter → text passes through with
	// title promoted to a field.
	got := FormatRetrievalDocument(
		"unknown-model",
		"T",
		[]Field{{"feed", "X"}},
		"Body.",
	)
	want := "title: T\nfeed: X\n\nBody."
	if got != want {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestRegisterDocumentPrompter_OverrideAndRemove(t *testing.T) {
	const name = "test-only-doc-model"
	t.Cleanup(func() { RegisterDocumentPrompter(name, nil) })

	RegisterDocumentPrompter(name, func(title, text string) string {
		return "DOC[" + title + "]: " + text
	})
	if got := FormatRetrievalDocument(name, "T", nil, "B"); got != "DOC[T]: B" {
		t.Errorf("after RegisterDocumentPrompter: got %q", got)
	}

	RegisterDocumentPrompter(name, nil)
	// Removed → no slot → fallback path adds title as a field.
	if got := FormatRetrievalDocument(name, "T", nil, "B"); got != "title: T\n\nB" {
		t.Errorf("after nil RegisterDocumentPrompter (remove): got %q", got)
	}
}

func TestLookupDocumentPrompter_UnknownReturnsNil(t *testing.T) {
	if p := LookupDocumentPrompter("entirely-fictional-model"); p != nil {
		t.Errorf("unknown model: got non-nil prompter")
	}
}

func TestGemmaTaskPrompter_RetrievalDocumentDelegates(t *testing.T) {
	// gemmaTaskPrompter(TaskRetrievalDocument, ...) must produce the
	// same output as gemmaDocumentPrompter("", ...) — they share an
	// implementation now.
	taskOut := gemmaTaskPrompter(TaskRetrievalDocument, "x")
	docOut := gemmaDocumentPrompter("", "x")
	if taskOut != docOut {
		t.Errorf("delegation broken: task=%q doc=%q", taskOut, docOut)
	}
}
