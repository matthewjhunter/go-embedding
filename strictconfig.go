package embedding

import (
	"fmt"
	"os"
)

// ConfigFromEnvPrefixStrict is ConfigFromEnvPrefix with StrictModel defaulted
// on: the configuration a long-running service wants.
//
// Model identity decides two things that fail *open* when the registry lookup
// misses: the task prefixes text is wrapped in, and the input budget requests
// are clipped to. A miss yields a passthrough prompter and a zero budget with
// no error, so the symptom is not a failure but quietly worse retrieval -- or,
// against a backend that rejects oversize input rather than truncating, a
// per-document failure with nothing naming the cause.
//
// Serving runtimes rename models freely -- Ollama appends a tag, Lemonade
// appends -GGUF and takes a user. prefix -- so a rename on the backend is
// enough to turn a resolved model into an unresolved one. Strict makes that
// stop startup instead.
//
// An operator who states a preference is never overridden, in either
// direction: the default applies only when neither {prefix}_STRICT_MODEL nor
// the bare EMBEDDING_STRICT_MODEL is set. Setting the value unconditionally
// would silently ignore an explicit opt-out, which is the trap this exists to
// avoid.
func ConfigFromEnvPrefixStrict(prefix string) (Config, error) {
	cfg, err := ConfigFromEnvPrefix(prefix)
	if err != nil {
		return cfg, err
	}
	if !strictModelSetByOperator(prefix) {
		cfg.StrictModel = true
	}
	return cfg, nil
}

// strictModelSetByOperator reports whether the strict-model choice was made
// explicitly, so a default is applied only when nobody stated a preference.
func strictModelSetByOperator(prefix string) bool {
	for _, k := range []string{prefix + envSuffixStrictModel, DefaultEnvPrefix + envSuffixStrictModel} {
		if _, ok := os.LookupEnv(k); ok {
			return true
		}
	}
	return false
}

// Describe returns one line reporting what a configured model name actually
// resolved to: the canonical name, whether task prefixes will be applied, and
// the byte budget in force.
//
// It exists so "which prefix and which budget am I actually using?" is
// answerable at startup rather than inferred from bad results weeks later. It
// returns the string rather than logging it, so the caller keeps its own
// logger and format.
func Describe(cfg Config) string {
	info, known := LookupModel(cfg.Model)
	if !known {
		return fmt.Sprintf("embedding model %q is unrecognised -- no task prefixes and no input budget", cfg.Model)
	}
	return fmt.Sprintf("embedding model %q resolved as %q (task prefixes: %v, budget: %d bytes)",
		cfg.Model, info.Canonical, info.HasPrompts, cfg.Limits().MaxBytes)
}
