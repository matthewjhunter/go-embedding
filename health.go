package embedding

// A health check for an embedder: is it working, and is it working as
// configured?
//
// Those are two questions and both go wrong quietly. A backend can return
// well-formed 768-dimension vectors that carry almost no retrieval signal, and
// a correctly-served model can be named in a way the prefix registry does not
// recognise, so text reaches it unwrapped. Neither produces an error. Both
// produce worse results that look like the model being mediocre.
//
// The check embeds a small fixed corpus of four topics with two paraphrases
// each and asks one question: does this embedder rank a paraphrase above an
// unrelated sentence? That is the minimum an embedder has to do, it needs no
// labelled dataset, and it takes a few seconds.

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"
)

// healthGroups are the probe texts: four topics, two paraphrases each, sharing
// no vocabulary with the other topics and no common prefix with each other.
//
// A shared prefix is specifically avoided. An earlier version of this probe
// prepended the document task prefix to every sample by hand, and on a backend
// whose pooling favours early tokens that alone inflated every similarity and
// made a merely weak embedder look completely broken. Prefixes are applied
// through FormatForTask here, which is also what exercises the registry.
var healthGroups = [][2]string{
	{"The cat sat on the mat.", "A feline rested upon the rug."},
	{"PostgreSQL uses a write-ahead log for durability.", "Postgres guarantees crash safety through WAL."},
	{"Sear the steak in a hot pan for two minutes a side.", "Brown the beef quickly over high heat."},
	{"Voyager 1 crossed the heliopause in 2012.", "The probe left the sun's magnetic bubble over a decade ago."},
}

// Verdict is the health check's summary judgement.
type Verdict string

const (
	// VerdictHealthy: paraphrases rank above unrelated text with room to
	// spare.
	VerdictHealthy Verdict = "healthy"
	// VerdictWeak: the ordering holds but the gap is small, so ranking is
	// fragile on harder material than these samples.
	VerdictWeak Verdict = "weak"
	// VerdictBroken: the distributions overlap or barely separate. Some
	// unrelated pair scores above some paraphrase pair, which means retrieval
	// order is unreliable rather than merely blunt.
	VerdictBroken Verdict = "broken"
)

// HealthReport is what the check found.
type HealthReport struct {
	// Identity, from the prefix and limit registries.
	Model      string
	Canonical  string
	Recognised bool
	HasPrompts bool
	Limits     Limits

	// Vector shape.
	Dim int

	// Similarity statistics over the probe corpus, within this embedder's own
	// space. SameMean is over paraphrase pairs, DiffMean over pairs from
	// different topics.
	SameMean   float64
	DiffMean   float64
	Separation float64 // SameMean - DiffMean
	// WorstMargin is the weakest paraphrase pair minus the strongest
	// unrelated pair. It is the number that matters: positive means every
	// paraphrase outranks every unrelated pair, negative means the
	// distributions overlap and ranking is unreliable. Mean separation can
	// look respectable while this is negative.
	WorstMargin float64

	// Timing, measured warm.
	SingleLatency  time.Duration
	BatchLatency   time.Duration // for BatchSize texts
	BatchSize      int
	TextsPerSecond float64

	Verdict Verdict
	Notes   []string
}

// Thresholds for the verdict, chosen from measurements of four real
// embedders rather than from intuition. See ReferenceReports for the values;
// the three working models scored separations of 0.17 to 0.30 with positive
// margins, and the failing one scored 0.08 with a margin of -0.14.
const (
	healthySeparation = 0.15
	brokenSeparation  = 0.10
)

// CheckHealth embeds the probe corpus through e and reports what it found.
//
// It applies the document task prefix through FormatForTask, so an embedder
// whose served name is not in the registry is measured exactly as a caller
// would use it -- unwrapped -- and the report says so in HasPrompts.
func CheckHealth(ctx context.Context, e Embedder) (HealthReport, error) {
	model := e.Model()
	info, recognised := LookupModel(model)
	r := HealthReport{
		Model:      model,
		Canonical:  info.Canonical,
		Recognised: recognised,
		HasPrompts: info.HasPrompts,
		Limits:     info.Limits,
	}
	if !r.HasPrompts {
		r.Notes = append(r.Notes, "no task prefixes are registered for this model name: text reaches the model unwrapped. "+
			"If it was trained with prefixes, retrieval is degraded and nothing reports it. Register an alias.")
	}

	var texts []string
	var group []int
	for gi, g := range healthGroups {
		for _, t := range g {
			texts = append(texts, FormatForTask(model, TaskRetrievalDocument, t))
			group = append(group, gi)
		}
	}

	// Warm the backend so the timings measure the model rather than a load.
	if _, err := e.Embed(ctx, []string{"warmup"}); err != nil {
		return r, fmt.Errorf("embedding health: warmup failed: %w", err)
	}

	start := time.Now()
	vecs, err := e.Embed(ctx, texts)
	if err != nil {
		return r, fmt.Errorf("embedding health: embedding probe corpus failed: %w", err)
	}
	r.BatchLatency = time.Since(start)
	r.BatchSize = len(texts)
	if r.BatchLatency > 0 {
		r.TextsPerSecond = float64(len(texts)) / r.BatchLatency.Seconds()
	}
	if len(vecs) != len(texts) {
		return r, fmt.Errorf("embedding health: got %d vectors for %d inputs", len(vecs), len(texts))
	}

	start = time.Now()
	if _, err := e.Embed(ctx, texts[:1]); err != nil {
		return r, fmt.Errorf("embedding health: single embed failed: %w", err)
	}
	r.SingleLatency = time.Since(start)

	r.Dim = len(vecs[0])
	for i, v := range vecs {
		if len(v) != r.Dim {
			return r, fmt.Errorf("embedding health: vector %d has dimension %d, first had %d", i, len(v), r.Dim)
		}
		if norm(v) == 0 {
			r.Notes = append(r.Notes, fmt.Sprintf("vector %d is all zeros", i))
		}
	}

	var same, diff []float64
	for i := range vecs {
		for j := i + 1; j < len(vecs); j++ {
			c := CosineSimilarity(vecs[i], vecs[j])
			if group[i] == group[j] {
				same = append(same, c)
			} else {
				diff = append(diff, c)
			}
		}
	}
	if len(same) == 0 || len(diff) == 0 {
		return r, fmt.Errorf("embedding health: probe corpus produced no comparable pairs")
	}
	sort.Float64s(same)
	sort.Float64s(diff)
	r.SameMean = mean(same)
	r.DiffMean = mean(diff)
	r.Separation = r.SameMean - r.DiffMean
	r.WorstMargin = same[0] - diff[len(diff)-1]

	switch {
	case r.WorstMargin <= 0 || r.Separation < brokenSeparation:
		r.Verdict = VerdictBroken
		r.Notes = append(r.Notes, "an unrelated pair scores at or above a paraphrase pair: retrieval order is unreliable, "+
			"not merely blunt. Check the serving runtime before the model.")
	case r.Separation < healthySeparation:
		r.Verdict = VerdictWeak
	default:
		r.Verdict = VerdictHealthy
	}
	return r, nil
}

func mean(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

// ReferenceReports are measured results for known embedders on this exact
// probe corpus, so a caller can tell an unusual number from a normal one
// without having a second backend to compare against.
//
// Measured 2026-08-30 against Lemonade on an AMD Strix Halo (Ryzen AI MAX+
// 395): the first three on the llama.cpp Vulkan backend over the integrated
// GPU, the last on the FLM backend over the NPU. Rates are that machine's and
// are included for shape rather than as a target -- the NPU figure is two
// orders of magnitude off the others on identical inputs.
var ReferenceReports = []HealthReport{
	{Model: "embeddinggemma", Dim: 768, SameMean: 0.6884, DiffMean: 0.3873,
		Separation: 0.3011, WorstMargin: 0.1199, TextsPerSecond: 149.2, Verdict: VerdictHealthy},
	{Model: "nomic-embed-text-v2-moe", Dim: 768, SameMean: 0.6478, DiffMean: 0.3667,
		Separation: 0.2810, WorstMargin: 0.0491, TextsPerSecond: 396.5, Verdict: VerdictHealthy},
	{Model: "nomic-embed-text-v1", Dim: 768, SameMean: 0.7727, DiffMean: 0.5995,
		Separation: 0.1732, WorstMargin: 0.0702, TextsPerSecond: 552.1, Verdict: VerdictHealthy},
	{Model: "embed-gemma-300m-FLM (NPU)", Dim: 768, SameMean: 0.8171, DiffMean: 0.7349,
		Separation: 0.0822, WorstMargin: -0.1438, TextsPerSecond: 5.8, Verdict: VerdictBroken},
}
