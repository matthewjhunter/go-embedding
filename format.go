package embedding

import (
	"strings"
	"sync"
)

// Task identifies the use case for an embedding. Models that support
// task-specific instructional prompts (nomic-embed-text, EmbeddingGemma)
// emit a different prefix per task. Callers pick the Task that matches
// what they intend to do with the resulting vector.
//
// Unknown Task values pass through unchanged — a caller may use a Task
// the library has no built-in convention for; the registered prompter
// is responsible for handling or ignoring it.
type Task string

const (
	// TaskRetrievalDocument: indexing a document for later retrieval.
	// nomic: "search_document: ", Gemma: "title: none | text: ".
	TaskRetrievalDocument Task = "retrieval-document"

	// TaskRetrievalQuery: a query string to search against indexed
	// documents. nomic: "search_query: ",
	// Gemma: "task: search result | query: ".
	TaskRetrievalQuery Task = "retrieval-query"

	// TaskClustering: text to be grouped with semantically similar
	// neighbours. nomic: "clustering: ",
	// Gemma: "task: clustering | query: ".
	TaskClustering Task = "clustering"

	// TaskClassification: text to be assigned a class label.
	// nomic: "classification: ",
	// Gemma: "task: classification | query: ".
	TaskClassification Task = "classification"

	// TaskSimilarity: text for symmetric semantic-similarity scoring.
	// Gemma: "task: sentence similarity | query: ". nomic has no
	// dedicated similarity prefix; falls back to no-op pass-through.
	TaskSimilarity Task = "similarity"

	// TaskQuestionAnswering: query for a QA-style retrieval system.
	// Gemma: "task: question answering | query: ". nomic has no
	// dedicated QA prefix.
	TaskQuestionAnswering Task = "question-answering"

	// TaskFactChecking: query for a fact-verification system.
	// Gemma: "task: fact checking | query: ". nomic has no dedicated
	// fact-checking prefix.
	TaskFactChecking Task = "fact-checking"

	// TaskCodeRetrieval: code-aware retrieval query.
	// Gemma: "task: code retrieval | query: ". nomic has no dedicated
	// code-retrieval prefix.
	TaskCodeRetrieval Task = "code-retrieval"
)

// Field is one labeled metadata line in the assembled embed text. Stored
// as an ordered slice (not a map) because the embedder learns metadata
// keys as features and Go map iteration is non-deterministic — two calls
// with the same data must produce identical text and identical vectors.
type Field struct {
	Key, Value string
}

// FormatRecord assembles structured metadata into the conventional
// embed-text format: "key: value" lines in slice order, a blank line,
// then the body. Empty values are skipped (their key is dropped). Empty
// keys are also skipped.
//
// If body is empty, the trailing blank line is omitted. If both fields
// and body are empty, returns "".
//
// The returned text is suitable for direct use with Embed/Single, or for
// further wrapping via FormatForTask.
func FormatRecord(fields []Field, body string) string {
	var b strings.Builder
	for _, f := range fields {
		if f.Key == "" || f.Value == "" {
			continue
		}
		b.WriteString(f.Key)
		b.WriteString(": ")
		b.WriteString(f.Value)
		b.WriteByte('\n')
	}
	if body == "" {
		// Trim the trailing newline so a fields-only record doesn't end
		// with whitespace.
		return strings.TrimRight(b.String(), "\n")
	}
	if b.Len() > 0 {
		b.WriteByte('\n')
	}
	b.WriteString(body)
	return b.String()
}

// TaskPrompter wraps text with a model-specific instructional prompt for
// the given Task. Implementations should return text unchanged when the
// task has no registered convention for the model — never error, never
// drop content.
type TaskPrompter func(task Task, text string) string

// modelTaskPrompters is the registry of per-model task-prompt rules.
// Reads via LookupTaskPrompter, writes via RegisterTaskPrompter, both
// guarded by modelTaskPromptersMu.
var (
	modelTaskPromptersMu sync.RWMutex
	modelTaskPrompters   = map[string]TaskPrompter{
		"nomic-embed-text":    nomicTaskPrompter,
		"nomic-embed-text-v2": nomicTaskPrompter,
		"embeddinggemma":      gemmaTaskPrompter,
	}
)

// LookupTaskPrompter returns the registered prompter for model, or a
// no-op prompter (returns text unchanged) if no convention is registered.
//
// Tag-suffix fallback is applied identically to LookupLimits: a tagged
// model name like "nomic-embed-text:latest" resolves to the bare model's
// prompter when the tagged variant is not explicitly registered. Task
// prompts are an architectural property of the base model.
func LookupTaskPrompter(model string) TaskPrompter {
	modelTaskPromptersMu.RLock()
	defer modelTaskPromptersMu.RUnlock()
	if p, ok := modelTaskPrompters[model]; ok {
		return p
	}
	if i := strings.IndexByte(model, ':'); i > 0 {
		if p, ok := modelTaskPrompters[model[:i]]; ok {
			return p
		}
	}
	return passthroughPrompter
}

// RegisterTaskPrompter adds or overrides the prompter for model. Safe to
// call at any time. Pass nil to remove a registration (callers will then
// see a no-op pass-through).
func RegisterTaskPrompter(model string, p TaskPrompter) {
	modelTaskPromptersMu.Lock()
	defer modelTaskPromptersMu.Unlock()
	if p == nil {
		delete(modelTaskPrompters, model)
		return
	}
	modelTaskPrompters[model] = p
}

// FormatForTask wraps text with the model's task-specific prefix.
// Equivalent to LookupTaskPrompter(model)(task, text). For models without
// a registered convention, or tasks the model doesn't recognise, the
// text is returned unchanged.
func FormatForTask(model string, task Task, text string) string {
	return LookupTaskPrompter(model)(task, text)
}

// FormatRecordForTask combines FormatRecord and FormatForTask in one
// call: the metadata block is assembled first, then wrapped in the
// model's task prefix. The resulting layout is:
//
//	<task prefix>key1: value1
//	key2: value2
//
//	<body>
//
// Truncation (via Embed/applyLimits) trims from the end, so the task
// prefix and field labels are preserved while the body absorbs the cut.
func FormatRecordForTask(model string, task Task, fields []Field, body string) string {
	return FormatForTask(model, task, FormatRecord(fields, body))
}

// DocumentPrompter formats a retrieval-document embed text with a
// structural title slot — i.e. the model has a dedicated convention for
// where the title goes, distinct from the body. Implementations supply
// a default when title is empty (e.g. EmbeddingGemma's "title: none |
// text: ...").
//
// Models without a structural title slot do not register a
// DocumentPrompter; FormatRetrievalDocument falls back to threading the
// title through the field labels for those models.
type DocumentPrompter func(title, text string) string

// modelDocumentPrompters is the registry of per-model retrieval-document
// formatters. Reads via LookupDocumentPrompter, writes via
// RegisterDocumentPrompter, both guarded by modelDocumentPromptersMu.
var (
	modelDocumentPromptersMu sync.RWMutex
	modelDocumentPrompters   = map[string]DocumentPrompter{
		"embeddinggemma": gemmaDocumentPrompter,
		// nomic-embed-text has no structural title slot — its
		// search_document prefix wraps the entire input as one blob.
	}
)

// LookupDocumentPrompter returns the registered document prompter for
// model, or nil if the model has no structural title slot. Tag-suffix
// fallback is applied identically to LookupLimits and
// LookupTaskPrompter.
func LookupDocumentPrompter(model string) DocumentPrompter {
	modelDocumentPromptersMu.RLock()
	defer modelDocumentPromptersMu.RUnlock()
	if p, ok := modelDocumentPrompters[model]; ok {
		return p
	}
	if i := strings.IndexByte(model, ':'); i > 0 {
		if p, ok := modelDocumentPrompters[model[:i]]; ok {
			return p
		}
	}
	return nil
}

// RegisterDocumentPrompter adds or overrides the document prompter for
// model. Pass nil to remove a registration.
func RegisterDocumentPrompter(model string, p DocumentPrompter) {
	modelDocumentPromptersMu.Lock()
	defer modelDocumentPromptersMu.Unlock()
	if p == nil {
		delete(modelDocumentPrompters, model)
		return
	}
	modelDocumentPrompters[model] = p
}

// FormatRetrievalDocument assembles a retrieval-document embed text
// using the model's structural title slot when available. For models
// with a registered DocumentPrompter (e.g. EmbeddingGemma's
// "title: <T> | text: <body>"), the title is placed in the model's
// dedicated slot. For models without one (e.g. nomic-embed-text), the
// title is promoted to a leading "title: <T>" field and the standard
// TaskRetrievalDocument prefix is applied.
//
// Empty title is allowed: registered prompters supply a default (e.g.
// "none"), and the no-slot fallback simply omits the title field.
//
// Use this helper instead of FormatRecordForTask(..., TaskRetrievalDocument, ...)
// when the document has a meaningful title — it gives title-aware
// models a chance to use it structurally rather than as a free-form
// metadata line.
func FormatRetrievalDocument(model, title string, fields []Field, body string) string {
	if p := LookupDocumentPrompter(model); p != nil {
		return p(title, FormatRecord(fields, body))
	}
	if title != "" {
		fields = append([]Field{{"title", title}}, fields...)
	}
	return FormatRecordForTask(model, TaskRetrievalDocument, fields, body)
}

// passthroughPrompter is the prompter returned for unregistered models.
// It is also returned by LookupTaskPrompter as the zero-value behaviour.
func passthroughPrompter(_ Task, text string) string { return text }

// nomicTaskPrompter implements the nomic-embed-text task convention:
// a single colon-separated prefix prepended to the input. Documented in
// nomic-embed-text-v1.5's model card as the four canonical task tags.
// Unknown tasks pass through unchanged.
func nomicTaskPrompter(task Task, text string) string {
	switch task {
	case TaskRetrievalDocument:
		return "search_document: " + text
	case TaskRetrievalQuery:
		return "search_query: " + text
	case TaskClassification:
		return "classification: " + text
	case TaskClustering:
		return "clustering: " + text
	}
	return text
}

// gemmaTaskPrompter implements the EmbeddingGemma task convention: a
// pipe-separated header of the form "task: <description> | query: <text>"
// for query-style tasks, and "title: <title> | text: <text>" for
// document indexing. Source: EmbeddingGemma model card.
//
// TaskRetrievalDocument delegates to gemmaDocumentPrompter with an empty
// title (default "none"). Callers with a real title should call
// FormatRetrievalDocument to use the title slot meaningfully.
func gemmaTaskPrompter(task Task, text string) string {
	switch task {
	case TaskRetrievalDocument:
		return gemmaDocumentPrompter("", text)
	case TaskRetrievalQuery:
		return "task: search result | query: " + text
	case TaskClustering:
		return "task: clustering | query: " + text
	case TaskClassification:
		return "task: classification | query: " + text
	case TaskSimilarity:
		return "task: sentence similarity | query: " + text
	case TaskQuestionAnswering:
		return "task: question answering | query: " + text
	case TaskFactChecking:
		return "task: fact checking | query: " + text
	case TaskCodeRetrieval:
		return "task: code retrieval | query: " + text
	}
	return text
}

// gemmaDocumentPrompter formats EmbeddingGemma's structured retrieval-
// document prompt: "title: <title> | text: <text>". Empty title falls
// back to the documented default of "none".
func gemmaDocumentPrompter(title, text string) string {
	if title == "" {
		title = "none"
	}
	return "title: " + title + " | text: " + text
}
