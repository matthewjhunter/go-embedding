package embedding

import (
	"context"
	"math"
	"strings"
	"sync"
	"testing"
)

func TestCalibration_UnobservedModel(t *testing.T) {
	ResetCalibration()
	if _, ok := CalibrationFor("never-used"); ok {
		t.Error("reported a calibration for a model with no observations")
	}
}

func TestCalibration_Statistics(t *testing.T) {
	ResetCalibration()

	// Ratios of 1, 2, 3, ... 10 bytes per token.
	for i := 1; i <= 10; i++ {
		recordUsage(Usage{Model: "m", Bytes: 100 * i, Tokens: 100})
	}

	got, ok := CalibrationFor("m")
	if !ok {
		t.Fatal("no calibration recorded")
	}
	if got.Samples != 10 {
		t.Errorf("Samples = %d, want 10", got.Samples)
	}
	if got.Min != 1 {
		t.Errorf("Min = %v, want 1", got.Min)
	}
	if got.P10 != 2 {
		t.Errorf("P10 = %v, want 2", got.P10)
	}
	if got.Median != 6 {
		t.Errorf("Median = %v, want 6", got.Median)
	}
	if math.Abs(got.Mean-5.5) > 1e-9 {
		t.Errorf("Mean = %v, want 5.5", got.Mean)
	}
}

// A clipped request measures only the bytes the backend got through before it
// stopped reading, so its ratio is meaningless. It must be counted, not
// measured -- the count is how a reader knows the sample is biased.
func TestCalibration_ClippedIsCountedNotMeasured(t *testing.T) {
	ResetCalibration()

	recordUsage(Usage{Model: "m", Bytes: 2000, Tokens: 1000})                 // ratio 2
	recordUsage(Usage{Model: "m", Bytes: 90000, Tokens: 2000, Clipped: true}) // ratio 45 if counted

	got, _ := CalibrationFor("m")
	if got.Samples != 1 {
		t.Errorf("Samples = %d, want 1", got.Samples)
	}
	if got.Clipped != 1 {
		t.Errorf("Clipped = %d, want 1", got.Clipped)
	}
	if got.Mean != 2 {
		t.Errorf("Mean = %v, want 2 -- the clipped observation was folded in", got.Mean)
	}
}

// A model whose every observation was clipped still reports the clip count, so
// "nothing measurable happened" is distinguishable from "nothing happened".
func TestCalibration_AllClipped(t *testing.T) {
	ResetCalibration()
	recordUsage(Usage{Model: "m", Bytes: 9000, Tokens: 2000, Clipped: true})

	got, ok := CalibrationFor("m")
	if !ok {
		t.Fatal("want a calibration reporting the clipped count")
	}
	if got.Samples != 0 || got.Clipped != 1 {
		t.Errorf("Samples=%d Clipped=%d, want 0 and 1", got.Samples, got.Clipped)
	}
}

func TestCalibration_WindowEvictsOldObservations(t *testing.T) {
	ResetCalibration()

	// Fill the window with a ratio of 1, then overwrite it entirely with 5.
	for range calibrationWindow {
		recordUsage(Usage{Model: "m", Bytes: 100, Tokens: 100})
	}
	for range calibrationWindow {
		recordUsage(Usage{Model: "m", Bytes: 500, Tokens: 100})
	}

	got, _ := CalibrationFor("m")
	if got.Samples != calibrationWindow {
		t.Errorf("Samples = %d, want the window size %d", got.Samples, calibrationWindow)
	}
	if got.Min != 5 {
		t.Errorf("Min = %v, want 5 -- the old observations did not age out", got.Min)
	}
}

func TestCalibration_IgnoresUnusableObservations(t *testing.T) {
	ResetCalibration()
	recordUsage(Usage{Model: "m", Bytes: 100, Tokens: 0})
	recordUsage(Usage{Model: "m", Bytes: 0, Tokens: 100})

	if _, ok := CalibrationFor("m"); ok {
		t.Error("recorded an observation with no bytes or no tokens")
	}
}

func TestCalibration_PerModel(t *testing.T) {
	ResetCalibration()
	recordUsage(Usage{Model: "a", Bytes: 200, Tokens: 100})
	recordUsage(Usage{Model: "b", Bytes: 400, Tokens: 100})

	a, _ := CalibrationFor("a")
	b, _ := CalibrationFor("b")
	if a.Mean != 2 || b.Mean != 4 {
		t.Errorf("means crossed between models: a=%v b=%v", a.Mean, b.Mean)
	}
}

// Observations arrive from whatever goroutines the caller embeds on.
func TestCalibration_ConcurrentRecording(t *testing.T) {
	ResetCalibration()

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 200 {
				recordUsage(Usage{Model: "m", Bytes: 200, Tokens: 100})
				CalibrationFor("m")
			}
		}()
	}
	wg.Wait()

	got, _ := CalibrationFor("m")
	if got.Samples != 1600 {
		t.Errorf("Samples = %d, want 1600", got.Samples)
	}
}

// The whole point is that this happens by itself on the embedding path.
func TestCalibration_AccumulatesFromEmbedCalls(t *testing.T) {
	ResetCalibration()

	srv := tokenReportingServer(t, 50)
	e, err := New(Config{Backend: BackendOllama, BaseURL: srv.URL, Model: "nomic-embed-text"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// 100 bytes against the server's reported 50 tokens -> ratio 2.
	text := strings.Repeat("a", 100)
	if _, err := e.Embed(context.Background(), []string{text}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	got, ok := CalibrationFor("nomic-embed-text")
	if !ok {
		t.Fatal("embedding recorded no calibration")
	}
	if got.Samples != 1 {
		t.Fatalf("Samples = %d, want 1", got.Samples)
	}
	if got.Mean != 2 {
		t.Errorf("Mean = %v, want 2", got.Mean)
	}
}
