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

	// envSuffixNormalizeScores is read only in the RERANK_* namespace; the
	// embedder config does not use it.
	envSuffixNormalizeScores = "_NORMALIZE_SCORES"
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

// ConfigFromEnvPrefix is ConfigFromEnv with a caller-supplied prefix and a
// per-field fallback chain: {prefix}_FOO → EMBEDDING_FOO → DefaultConfig.
//
// E.g. ConfigFromEnvPrefix("MEMSTORE_EMBED") reads MEMSTORE_EMBED_BASE_URL
// first; if unset, falls back to EMBEDDING_BASE_URL; if still unset, takes
// the value from DefaultConfig. This means one canonical env set
// (EMBEDDING_*) works for every app, and any app can override just the
// fields it needs to differ.
//
// Returns an error only on parse failures (unknown backend, malformed
// bool); missing vars fall back to DefaultConfig.
func ConfigFromEnvPrefix(prefix string) (Config, error) {
	cfg := DefaultConfig()

	if v, src := envCascade(prefix, envSuffixBackend); v != "" {
		switch strings.ToLower(v) {
		case string(BackendOllama):
			cfg.Backend = BackendOllama
		case string(BackendOpenAI):
			cfg.Backend = BackendOpenAI
		default:
			return Config{}, fmt.Errorf(
				"embedding: unknown backend %q in %s (want ollama|openai)",
				v, src,
			)
		}
	}
	if v, _ := envCascade(prefix, envSuffixBaseURL); v != "" {
		cfg.BaseURL = v
	}
	if v, _ := envCascade(prefix, envSuffixAPIKey); v != "" {
		cfg.APIKey = v
	}
	if v, _ := envCascade(prefix, envSuffixModel); v != "" {
		cfg.Model = v
	}
	if v, src := envCascade(prefix, envSuffixStrict); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return Config{}, fmt.Errorf(
				"embedding: invalid %s value %q: %w",
				src, v, err,
			)
		}
		cfg.Strict = b
	}
	return cfg, nil
}

// envCascade looks up suffix under prefix first, then under DefaultEnvPrefix.
// Returns the first non-empty value and the env-var name it came from. If
// prefix already equals DefaultEnvPrefix only one lookup is performed. Empty
// strings are treated as unset.
func envCascade(prefix, suffix string) (value, source string) {
	return envCascadeTo(prefix, DefaultEnvPrefix, suffix)
}

// envCascadeTo is envCascade with a caller-supplied canonical fallback prefix,
// so namespaces other than EMBEDDING (e.g. RERANK) can share the same lookup
// logic. It checks prefix+suffix, then canonical+suffix.
func envCascadeTo(prefix, canonical, suffix string) (value, source string) {
	key := prefix + suffix
	if v := os.Getenv(key); v != "" {
		return v, key
	}
	if prefix == canonical {
		return "", ""
	}
	ckey := canonical + suffix
	if v := os.Getenv(ckey); v != "" {
		return v, ckey
	}
	return "", ""
}
