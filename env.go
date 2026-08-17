package embedding

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
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

	envSuffixTimeout         = "_TIMEOUT"
	envSuffixPerInputTimeout = "_PER_INPUT_TIMEOUT"

	envSuffixMaxBytes  = "_MAX_BYTES"
	envSuffixMaxTokens = "_MAX_TOKENS"

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
	if v, src := envCascade(prefix, envSuffixTimeout); v != "" {
		d, err := parseTimeoutEnv(src, v)
		if err != nil {
			return Config{}, err
		}
		cfg.Timeout = d
	}
	if v, src := envCascade(prefix, envSuffixPerInputTimeout); v != "" {
		d, err := parseTimeoutEnv(src, v)
		if err != nil {
			return Config{}, err
		}
		cfg.PerInputTimeout = d
	}
	if v, src := envCascade(prefix, envSuffixMaxBytes); v != "" {
		n, err := parseBudgetEnv(src, v)
		if err != nil {
			return Config{}, err
		}
		cfg.MaxBytes = n
	}
	if v, src := envCascade(prefix, envSuffixMaxTokens); v != "" {
		n, err := parseBudgetEnv(src, v)
		if err != nil {
			return Config{}, err
		}
		cfg.MaxTokens = n
	}
	return cfg, nil
}

// parseBudgetEnv parses a non-negative integer budget env var.
func parseBudgetEnv(source, value string) (int, error) {
	n, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("embedding: invalid %s value %q (want a whole number): %w", source, value, err)
	}
	if n < 0 {
		return 0, fmt.Errorf("embedding: invalid %s value %q: a budget cannot be negative", source, value)
	}
	return n, nil
}

// parseTimeoutEnv parses a duration env var, accepting "none" to disable the
// deadline. A bare number is rejected rather than guessed at: "30" is as
// likely to mean seconds as milliseconds, and silently choosing wrong turns a
// deadline into either a hang or a stream of spurious timeouts.
func parseTimeoutEnv(source, value string) (time.Duration, error) {
	if strings.EqualFold(value, "none") {
		return NoTimeout, nil
	}
	d, err := time.ParseDuration(value)
	if err != nil {
		return 0, fmt.Errorf(
			"embedding: invalid %s value %q (want a duration like 30s, or \"none\"): %w",
			source, value, err,
		)
	}
	if d < 0 {
		return 0, fmt.Errorf(
			"embedding: invalid %s value %q: a negative duration is not a deadline (use \"none\" to disable)",
			source, value,
		)
	}
	return d, nil
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
