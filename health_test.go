package embedding

import (
	"context"
	"math"
	"math/rand"
	"strings"
	"testing"
)

// healthFake produces vectors with a controllable amount of topic signal, so
// the verdict thresholds can be exercised without a backend.
type healthFake struct {
	model  string
	signal float64 // 0 = pure noise, 1 = topic and nothing else
	dim    int
}

func (f *healthFake) Model() string { return f.model }
func (f *healthFake) Fingerprint() Fingerprint {
	return Fingerprint{Model: f.model, Dim: f.dim}
}

// Embed places each text near a per-topic axis, mixed with deterministic noise
// keyed on the text so paraphrases are near each other but not identical.
func (f *healthFake) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, f.dim)
		rng := rand.New(rand.NewSource(int64(len(t)) * 7))
		for j := range v {
			v[j] = float32((1 - f.signal) * rng.NormFloat64())
		}
		v[f.topic(t)%f.dim] += float32(f.signal * 4)
		out[i] = v
	}
	return out, nil
}

// topic recognises the probe corpus by a distinctive word from each group.
func (f *healthFake) topic(t string) int {
	for i, kw := range []string{"cat", "feline", "PostgreSQL", "Postgres", "steak", "beef", "Voyager", "probe"} {
		if strings.Contains(t, kw) {
			return i / 2
		}
	}
	return 99
}

func TestCheckHealth_VerdictTracksSeparation(t *testing.T) {
	cases := []struct {
		name   string
		signal float64
		want   Verdict
	}{
		{"strong topic signal", 0.9, VerdictHealthy},
		{"no topic signal at all", 0.0, VerdictBroken},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r, err := CheckHealth(context.Background(), &healthFake{model: "embeddinggemma", signal: c.signal, dim: 64})
			if err != nil {
				t.Fatal(err)
			}
			if r.Verdict != c.want {
				t.Errorf("verdict = %q (separation %.4f, worst margin %.4f), want %q",
					r.Verdict, r.Separation, r.WorstMargin, c.want)
			}
		})
	}
}

// The report has to say when prefixes are missing, because that is the failure
// with no other symptom: the embedder works, the vectors are fine, and every
// result is quietly worse than it should be.
func TestCheckHealth_FlagsMissingPrefixes(t *testing.T) {
	known, err := CheckHealth(context.Background(), &healthFake{model: "embeddinggemma", signal: 0.9, dim: 64})
	if err != nil {
		t.Fatal(err)
	}
	if !known.HasPrompts || !known.Recognised {
		t.Errorf("embeddinggemma: recognised=%v prompts=%v, want both true", known.Recognised, known.HasPrompts)
	}
	if noteMentions(known.Notes, "no task prefixes") {
		t.Errorf("a recognised model was warned about prefixes: %v", known.Notes)
	}

	unknown, err := CheckHealth(context.Background(), &healthFake{model: "acme-embed-turbo-v3", signal: 0.9, dim: 64})
	if err != nil {
		t.Fatal(err)
	}
	if unknown.Recognised || unknown.HasPrompts {
		t.Errorf("acme-embed-turbo-v3 should be unrecognised, got recognised=%v prompts=%v", unknown.Recognised, unknown.HasPrompts)
	}
	if !noteMentions(unknown.Notes, "no task prefixes") {
		t.Errorf("unrecognised model produced no prefix warning: %v", unknown.Notes)
	}
}

// WorstMargin, not mean separation, is what the broken verdict turns on. This
// pins the distinction: an embedder can post a respectable-looking mean while
// an unrelated pair outranks a paraphrase.
func TestCheckHealth_ReportsShapeAndTiming(t *testing.T) {
	r, err := CheckHealth(context.Background(), &healthFake{model: "embeddinggemma", signal: 0.9, dim: 64})
	if err != nil {
		t.Fatal(err)
	}
	if r.Dim != 64 {
		t.Errorf("Dim = %d, want 64", r.Dim)
	}
	if r.BatchSize != 8 {
		t.Errorf("BatchSize = %d, want 8 (four topics, two texts each)", r.BatchSize)
	}
	if r.SingleLatency <= 0 || r.BatchLatency <= 0 || r.TextsPerSecond <= 0 {
		t.Errorf("timings not populated: single=%v batch=%v rate=%.1f", r.SingleLatency, r.BatchLatency, r.TextsPerSecond)
	}
	if math.Abs(r.Separation-(r.SameMean-r.DiffMean)) > 1e-9 {
		t.Errorf("Separation %.6f is not SameMean-DiffMean", r.Separation)
	}
}

// The published reference numbers are what a reader compares against, so they
// have to be self-consistent with the thresholds the verdict uses. A typo here
// would mislead every person running the check.
func TestReferenceReportsAgreeWithTheThresholds(t *testing.T) {
	for _, ref := range ReferenceReports {
		var want Verdict
		switch {
		case ref.WorstMargin <= 0 || ref.Separation < brokenSeparation:
			want = VerdictBroken
		case ref.Separation < healthySeparation:
			want = VerdictWeak
		default:
			want = VerdictHealthy
		}
		if ref.Verdict != want {
			t.Errorf("%s: recorded verdict %q but its numbers imply %q (sep %.4f, margin %.4f)",
				ref.Model, ref.Verdict, want, ref.Separation, ref.WorstMargin)
		}
		if math.Abs(ref.Separation-(ref.SameMean-ref.DiffMean)) > 5e-4 {
			t.Errorf("%s: separation %.4f does not match %.4f - %.4f",
				ref.Model, ref.Separation, ref.SameMean, ref.DiffMean)
		}
	}
}

func noteMentions(notes []string, sub string) bool {
	for _, n := range notes {
		if strings.Contains(n, sub) {
			return true
		}
	}
	return false
}
