package embedding

import "testing"

// Both service callers wrote the same ~65 lines around ConfigFromEnvPrefix.
// The only part with a trap in it is the default: applying it unconditionally
// silently overrides an operator who explicitly opted out (#28).

func TestConfigFromEnvPrefixStrict_DefaultsStrict(t *testing.T) {
	t.Setenv("SVC_MODEL", "nomic-embed-text")

	cfg, err := ConfigFromEnvPrefixStrict("SVC")
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.StrictModel {
		t.Error("StrictModel is off; an unrecognised model would get no prefixes and no budget, silently")
	}
}

func TestConfigFromEnvPrefixStrict_HonoursAnExplicitOptOut(t *testing.T) {
	t.Setenv("SVC_MODEL", "nomic-embed-text")
	t.Setenv("SVC_STRICT_MODEL", "false")

	cfg, err := ConfigFromEnvPrefixStrict("SVC")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.StrictModel {
		t.Error("an explicit SVC_STRICT_MODEL=false was overridden by the default")
	}
}

// The bare name is the cascade this library documents, so a deployment-wide
// setting must not be overridden either.
func TestConfigFromEnvPrefixStrict_HonoursTheBareOptOut(t *testing.T) {
	t.Setenv("SVC_MODEL", "nomic-embed-text")
	t.Setenv("EMBEDDING_STRICT_MODEL", "false")

	cfg, err := ConfigFromEnvPrefixStrict("SVC")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.StrictModel {
		t.Error("an explicit EMBEDDING_STRICT_MODEL=false was overridden by the default")
	}
}

// An explicit true is honoured too -- the default must not be the only way to
// get strict, or "did the operator ask for this?" becomes unanswerable.
func TestConfigFromEnvPrefixStrict_HonoursAnExplicitOptIn(t *testing.T) {
	t.Setenv("SVC_MODEL", "nomic-embed-text")
	t.Setenv("SVC_STRICT_MODEL", "true")

	cfg, err := ConfigFromEnvPrefixStrict("SVC")
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.StrictModel {
		t.Error("an explicit opt-in did not take effect")
	}
}

func TestDescribe_ReportsWhatTheNameResolvedTo(t *testing.T) {
	got := Describe(Config{Model: "nomic-embed-text:latest"})

	for _, want := range []string{"nomic-embed-text:latest", "nomic-embed-text"} {
		if !contains(got, want) {
			t.Errorf("Describe = %q, missing %q", got, want)
		}
	}
}

func TestDescribe_SaysSoWhenTheModelIsUnknown(t *testing.T) {
	got := Describe(Config{Model: "not-a-real-embedding-model"})

	if !contains(got, "unrecognised") {
		t.Errorf("Describe = %q, want it to say the model is unrecognised", got)
	}
}

func contains(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
