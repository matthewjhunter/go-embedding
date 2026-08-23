package embedding

// calibrationMinSamples is how many observations to require before trusting a
// measured bytes-per-token ratio over the built-in fallback.
//
// A handful of atypical documents early in a run would otherwise set the budget
// for everything after them -- and the first documents through a fresh process
// are exactly the ones most likely to be unrepresentative, because a backfill
// tends to walk a corpus in insertion order.
const calibrationMinSamples = 20

// BudgetForTokens returns the byte budget to use for a token target on this
// model.
//
// Split's own documentation says the model's full budget is the wrong target
// for retrieval and that 256-512 tokens is the working range -- but until now
// there was no way to turn that token figure into the bytes Split actually
// takes, so every caller invented the conversion. This library holds the
// observations and the fallback; this is the join between them.
//
// The ratio comes from what has actually been observed for the model
// (CalibrationFor), falling back to conservativeBytesPerToken until enough has
// been seen. That matters because the ratio is a property of the corpus as much
// as the model: stripped article text has been measured at 2.75 bytes per token
// against the ~4 that English prose suggests, and denser content runs lower
// still.
//
// P10 rather than the mean, deliberately. A *low* bytes-per-token ratio means
// *more* tokens per byte, so the tenth percentile is the conservative end of
// the distribution. Sizing from the mean lets the densest documents in a mixed
// corpus overrun the token target, and those are precisely the ones a backend
// that rejects oversize input rather than truncating fails outright.
//
// Returns 0 for a non-positive token target, which callers can treat as
// "no target".
func BudgetForTokens(model string, tokens int) int {
	if tokens <= 0 {
		return 0
	}
	ratio := float64(conservativeBytesPerToken)
	if cal, ok := CalibrationFor(model); ok && cal.Samples >= calibrationMinSamples && cal.P10 > 0 {
		ratio = cal.P10
	}
	return int(float64(tokens) * ratio)
}
