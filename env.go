package embedding

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// DefaultEnvPrefix is the prefix used by ConfigFromEnv. Callers wanting to
// share one embedding configuration across multiple apps should set
// EMBEDDING_BASE_URL, EMBEDDING_MODEL, etc., once and have every app call
// ConfigFromEnv.
const DefaultEnvPrefix = "EMBEDDING"

// Env-var suffixes appended to the configured prefix.
const (
	envSuffixBackend = "_BACKEND"
	envSuffixBaseURL = "_BASE_URL"
	envSuffixAPIKey  = "_API_KEY"
	envSuffixModel   = "_MODEL"
	envSuffixStrict  = "_STRICT"
)

// ConfigFromEnv reads Config from EMBEDDING_BACKEND, EMBEDDING_BASE_URL,
// EMBEDDING_API_KEY, EMBEDDING_MODEL, and EMBEDDING_STRICT, falling back to
// DefaultConfig field-by-field for any unset (or empty) variable.
//
// The intent is "set the embedding configuration once, every app reads
// from the same env." If you need per-app namespaces use ConfigFromEnvPrefix.
func ConfigFromEnv() (Config, error) {
	return ConfigFromEnvPrefix(DefaultEnvPrefix)
}

// ConfigFromEnvPrefix is ConfigFromEnv with a caller-supplied prefix.
// E.g. ConfigFromEnvPrefix("MEMSTORE_EMBED") reads MEMSTORE_EMBED_BACKEND,
// MEMSTORE_EMBED_BASE_URL, etc. Returns an error only on parse failures
// (unknown backend, malformed bool); missing vars fall back to DefaultConfig.
func ConfigFromEnvPrefix(prefix string) (Config, error) {
	cfg := DefaultConfig()

	if v := lookup(prefix + envSuffixBackend); v != "" {
		switch strings.ToLower(v) {
		case string(BackendOllama):
			cfg.Backend = BackendOllama
		case string(BackendOpenAI):
			cfg.Backend = BackendOpenAI
		default:
			return Config{}, fmt.Errorf(
				"embedding: unknown backend %q in %s%s (want ollama|openai)",
				v, prefix, envSuffixBackend,
			)
		}
	}
	if v := lookup(prefix + envSuffixBaseURL); v != "" {
		cfg.BaseURL = v
	}
	if v := lookup(prefix + envSuffixAPIKey); v != "" {
		cfg.APIKey = v
	}
	if v := lookup(prefix + envSuffixModel); v != "" {
		cfg.Model = v
	}
	if v := lookup(prefix + envSuffixStrict); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return Config{}, fmt.Errorf(
				"embedding: invalid %s%s value %q: %w",
				prefix, envSuffixStrict, v, err,
			)
		}
		cfg.Strict = b
	}
	return cfg, nil
}

// lookup returns the value of the named env var, or "" if unset or empty.
// We treat empty-string as unset so a caller exporting `EMBEDDING_API_KEY=`
// to wipe a stale key from the parent shell behaves the same as not setting it.
func lookup(key string) string {
	return os.Getenv(key)
}
