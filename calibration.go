package embedding

import (
	"slices"
	"sync"
)

// calibrationWindow is how many recent observations are kept per model. The
// point is a current picture of the corpus being embedded, not a lifetime
// average, so old observations age out. 4096 ratios is a few dozen KB per
// model and enough for a stable low percentile.
const calibrationWindow = 4096

// Calibration is a snapshot of how a model's tokenizer has actually behaved
// on the text sent to it: bytes of input per token of context consumed.
//
// This is measurement only. Nothing in the library derives a budget from it
// yet -- the static ratio in limits.go is still what converts tokens to bytes.
// Read it to find out what the ratio really is for your corpus before
// deciding what to do about it.
type Calibration struct {
	// Model is the model these observations came from.
	Model string
	// Samples is how many observations back this snapshot.
	Samples int
	// Clipped is how many observations were discarded because the backend
	// clipped the input, which undercounts its true token cost.
	//
	// Read this before trusting the rest. Clipped inputs are the densest
	// documents in a corpus, so a high count here means the retained sample
	// is biased toward text that tokenizes more easily than the average --
	// and the estimate is correspondingly optimistic.
	Clipped int
	// Min, P10, Median, and Mean are bytes per token across the retained
	// observations. A budget derived from this should use the low end rather
	// than the middle: underestimating the ratio clips input, while
	// overestimating only wastes a little of the window.
	Min    float64
	P10    float64
	Median float64
	Mean   float64
}

// calibrator accumulates bytes-per-token observations for one model.
type calibrator struct {
	ratios  []float64 // ring buffer of the most recent observations
	next    int
	full    bool
	clipped int
}

var (
	calibrationMu sync.Mutex
	calibrations  = map[string]*calibrator{}
)

// recordUsage folds one request's observation into the model's calibration.
//
// A clipped request is counted but not measured: the backend stopped reading
// at the context limit, so the tokens it reported cover only part of the bytes
// that were sent and the ratio would come out inflated.
//
// A multi-input request contributes its aggregate ratio, which is legitimate
// -- total bytes over total tokens is what the tokenizer did to that text.
// Clipping inside a batch cannot be detected (both protocols report one total
// for the request), so a batch carrying a clipped input inflates the ratio
// slightly. After chunking that case should not arise; before it, prefer
// single-input observations.
func recordUsage(u Usage) {
	if u.Tokens <= 0 || u.Bytes <= 0 {
		return
	}

	calibrationMu.Lock()
	defer calibrationMu.Unlock()

	c := calibrations[u.Model]
	if c == nil {
		c = &calibrator{ratios: make([]float64, calibrationWindow)}
		calibrations[u.Model] = c
	}
	if u.Clipped {
		c.clipped++
		return
	}

	c.ratios[c.next] = float64(u.Bytes) / float64(u.Tokens)
	c.next++
	if c.next == len(c.ratios) {
		c.next = 0
		c.full = true
	}
}

// CalibrationFor returns what has been observed of model's bytes-per-token
// ratio, and whether anything has been observed at all.
//
// Observations are in-process and not persisted: a restart starts over. They
// accumulate automatically from every embed request whose backend reported a
// token count.
func CalibrationFor(model string) (Calibration, bool) {
	calibrationMu.Lock()
	c := calibrations[model]
	if c == nil {
		calibrationMu.Unlock()
		return Calibration{}, false
	}
	n := c.next
	if c.full {
		n = len(c.ratios)
	}
	ratios := slices.Clone(c.ratios[:n])
	clipped := c.clipped
	calibrationMu.Unlock()

	out := Calibration{Model: model, Samples: len(ratios), Clipped: clipped}
	if len(ratios) == 0 {
		return out, clipped > 0
	}

	var sum float64
	for _, r := range ratios {
		sum += r
	}
	out.Mean = sum / float64(len(ratios))

	slices.Sort(ratios)
	out.Min = ratios[0]
	out.P10 = percentileOf(ratios, 10)
	out.Median = percentileOf(ratios, 50)
	return out, true
}

// percentileOf returns the p-th percentile of a sorted slice by nearest rank.
func percentileOf(sorted []float64, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := len(sorted) * p / 100
	return sorted[min(i, len(sorted)-1)]
}

// ResetCalibration discards every observation. Intended for tests and for a
// caller that wants a clean window around a specific run -- a re-embed, say --
// rather than a mix of that run and whatever preceded it.
func ResetCalibration() {
	calibrationMu.Lock()
	defer calibrationMu.Unlock()
	clear(calibrations)
}
