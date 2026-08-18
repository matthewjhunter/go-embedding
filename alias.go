package embedding

import (
	"fmt"
	"strings"
	"sync"
)

// Model identity decides two things: which task prefixes the text is wrapped in,
// and how large an input the model accepts. Both are looked up by name, and both
// silently do nothing when the lookup misses -- a miss yields a passthrough
// prompter and a zero Limits, so the caller gets no prefix on a model trained to
// require one, and no budget at all.
//
// The second is the dangerous half. A caller sizing chunks from the registered
// budget will build inputs far past the model's context, and a backend that
// rejects oversize input hard rather than truncating (llama.cpp returns HTTP 500
// at its physical batch limit) then fails every long document, with nothing
// naming the cause.
//
// Misses are the normal case rather than the exception, because serving runtimes
// rename models. Ollama appends a `:tag`; Lemonade appends `-GGUF` and
// quantisation markers and takes a `user.` prefix on registration. None of those
// describe a different model, but all of them defeat an exact-match lookup.

var (
	modelAliasesMu sync.RWMutex
	modelAliases   = map[string]string{}
)

// RegisterModelAlias maps a served model name onto a canonical one, so prefixes
// and limits resolve for a model a runtime has renamed.
//
// The alias affects lookup only. It deliberately does not rename the model:
// Embedder.Model() keeps reporting the served name, because stored vectors are
// keyed by it and vectors from different quantisations or variants are not
// interchangeable. Resolving identity for prompting is a different question from
// asserting two models produce the same vectors.
//
// Safe for concurrent use, though the expected pattern is registering at startup.
func RegisterModelAlias(alias, canonical string) {
	modelAliasesMu.Lock()
	defer modelAliasesMu.Unlock()
	modelAliases[strings.ToLower(strings.TrimSpace(alias))] = strings.TrimSpace(canonical)
}

// ResetModelAliases clears every registered alias. For tests.
func ResetModelAliases() {
	modelAliasesMu.Lock()
	defer modelAliasesMu.Unlock()
	modelAliases = map[string]string{}
}

// lookupAlias returns the canonical name registered for a served name.
func lookupAlias(name string) (string, bool) {
	modelAliasesMu.RLock()
	defer modelAliasesMu.RUnlock()
	c, ok := modelAliases[strings.ToLower(strings.TrimSpace(name))]
	return c, ok
}

// canonicalModel reduces a served model name to the name the registries are
// keyed by: an explicit alias if one is registered, otherwise the name with the
// affixes serving runtimes add stripped.
//
// Only affixes that describe *packaging* are stripped -- a tag, the `-GGUF`
// format marker, Lemonade's `user.` registration prefix, and letter case. A
// version or mixture marker is left alone: `-v2` and `-moe` denote a different
// model, and resolving one model to another's prefixes and budget would be worse
// than not resolving at all. Everything beyond packaging needs an explicit alias,
// because only the operator knows whether two names mean the same weights.
func canonicalModel(name string) string {
	if c, ok := lookupAlias(name); ok {
		return c
	}
	s := strings.ToLower(strings.TrimSpace(name))
	s = strings.TrimPrefix(s, "user.")
	if i := strings.IndexByte(s, ':'); i > 0 {
		s = s[:i]
	}
	s = strings.TrimSuffix(s, ".gguf")
	s = strings.TrimSuffix(s, "-gguf")
	return s
}

// ModelInfo is what a lookup actually resolved to, so a caller can report it
// rather than guess. See LookupModel.
type ModelInfo struct {
	// Name is the model as the caller named it, and the key vectors are stored
	// under.
	Name string
	// Canonical is the name its prefixes and limits were resolved through.
	Canonical string
	// Limits is the effective registered budget, zero when none is known.
	Limits Limits
	// HasPrompts reports whether task prefixes will be applied. False means
	// text reaches the model unwrapped, which on a model trained with task
	// prefixes is a silent quality loss rather than an error.
	HasPrompts bool
}

// LookupModel reports how a model name resolves. ok is false when it resolves to
// neither prompts nor limits, which means the library knows nothing about it and
// both prefixing and budget enforcement are inert.
//
// This exists so "which prefix and which budget am I actually using?" is
// answerable at startup instead of inferred from bad results later.
func LookupModel(name string) (ModelInfo, bool) {
	canonical := canonicalModel(name)
	info := ModelInfo{
		Name:       name,
		Canonical:  canonical,
		Limits:     LookupLimits(name),
		HasPrompts: LookupTaskPrompter(name) != nil && FormatForTask(name, TaskClustering, "\x00") != "\x00",
	}
	return info, info.HasPrompts || info.Limits.MaxBytes > 0 || info.Limits.MaxTokens > 0
}

// checkStrictModel returns an error when cfg demands a recognised model and the
// configured one is not.
func checkStrictModel(cfg Config) error {
	if !cfg.StrictModel {
		return nil
	}
	if _, ok := LookupModel(cfg.Model); !ok {
		return fmt.Errorf("embedding: model %q is not recognised: no task prefixes and no input budget would be applied. "+
			"Register an alias (RegisterModelAlias or EMBEDDING_MODEL_ALIAS) mapping it to a known model, "+
			"or unset StrictModel to proceed without either", cfg.Model)
	}
	return nil
}

// parseModelAliases parses the alias env var: comma-separated served=canonical
// pairs, whitespace tolerated around both. A malformed entry is an error rather
// than a skip -- an alias that silently did not apply would reproduce exactly the
// quiet failure this is here to prevent.
func parseModelAliases(source, value string) ([]string, error) {
	var out []string
	for _, pair := range strings.Split(value, ",") {
		if strings.TrimSpace(pair) == "" {
			continue
		}
		alias, canonical, ok := strings.Cut(pair, "=")
		alias, canonical = strings.TrimSpace(alias), strings.TrimSpace(canonical)
		if !ok || alias == "" || canonical == "" {
			return nil, fmt.Errorf("embedding: %s: %q is not a served=canonical pair", source, strings.TrimSpace(pair))
		}
		out = append(out, alias, canonical)
	}
	return out, nil
}
