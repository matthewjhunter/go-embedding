package embedding

import "testing"

// BudgetForTokens is the join this library was missing: it holds the
// observations (CalibrationFor) and the fallback (conservativeBytesPerToken)
// but never connected them, so every caller re-derived the conversion and the
// statistical judgement behind it (#30).

func TestBudgetForTokens_UsesTheConservativeFallbackWhenNothingObserved(t *testing.T) {
	ResetCalibration()

	got := BudgetForTokens("unobserved-model", 512)

	if want := 512 * conservativeBytesPerToken; got != want {
		t.Errorf("BudgetForTokens = %d, want %d (the conservative fallback)", got, want)
	}
}

// The whole point of calibrating: a corpus that runs denser or looser than the
// guess should move the budget.
func TestBudgetForTokens_PrefersObservedRatio(t *testing.T) {
	ResetCalibration()
	const model = "observed-model"
	// A ratio well above the fallback, observed enough times to be trusted.
	observe(t, model, 4.0, calibrationMinSamples)

	got := BudgetForTokens(model, 512)

	if fallback := 512 * conservativeBytesPerToken; got <= fallback {
		t.Errorf("BudgetForTokens = %d with a 4.0 ratio observed, want more than the %d fallback",
			got, fallback)
	}
}

// A handful of atypical documents early in a run must not set the budget for
// everything after them.
func TestBudgetForTokens_IgnoresTooFewSamples(t *testing.T) {
	ResetCalibration()
	const model = "barely-observed"
	observe(t, model, 4.0, calibrationMinSamples-1)

	got := BudgetForTokens(model, 512)

	if want := 512 * conservativeBytesPerToken; got != want {
		t.Errorf("BudgetForTokens = %d after %d samples, want the %d fallback until %d",
			got, calibrationMinSamples-1, want, calibrationMinSamples)
	}
}

// The statistical judgement worth owning once: a LOW bytes-per-token ratio
// means MORE tokens per byte, so the low percentile is the conservative end.
// Sizing from the mean pushes the densest documents past the token target --
// exactly the ones a strict backend rejects outright.
func TestBudgetForTokens_IsConservativeAgainstMixedDensity(t *testing.T) {
	ResetCalibration()
	const model = "mixed-density"
	// Mostly loose text with a dense minority: the mean sits well above the
	// tenth percentile.
	observe(t, model, 2.0, 5)
	observe(t, model, 6.0, calibrationMinSamples)

	cal, ok := CalibrationFor(model)
	if !ok || cal.Mean <= cal.P10 {
		t.Fatalf("test needs a mean above P10, got mean=%v p10=%v", cal.Mean, cal.P10)
	}

	got := BudgetForTokens(model, 512)

	if fromMean := int(512 * cal.Mean); got >= fromMean {
		t.Errorf("BudgetForTokens = %d, which is at or above the %d the mean would give; "+
			"the dense tail would overrun the token target", got, fromMean)
	}
}

func TestBudgetForTokens_NonPositiveTokensYieldsZero(t *testing.T) {
	if got := BudgetForTokens("m", 0); got != 0 {
		t.Errorf("BudgetForTokens(m, 0) = %d, want 0", got)
	}
}

// observe folds n observations of the given bytes-per-token ratio into model's
// calibration, via the same path a real request takes.
func observe(t *testing.T, model string, ratio float64, n int) {
	t.Helper()
	for range n {
		recordUsage(Usage{Model: model, Inputs: 1, Bytes: int(ratio * 1000), Tokens: 1000})
	}
}
