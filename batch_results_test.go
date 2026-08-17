package embedding

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
)

// okEmbed returns one vector per input, each carrying the input's length so
// tests can check index alignment.
func okEmbed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{float32(len(texts[i]))}
	}
	return out, nil
}

func TestBatchEmbedResults_HappyPath(t *testing.T) {
	f := &fakeEmbedder{model: "test", embed: okEmbed}

	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	got, err := BatchEmbedResults(context.Background(), f, texts, 2, nil)
	if err != nil {
		t.Fatalf("BatchEmbedResults: %v", err)
	}
	if len(got) != len(texts) {
		t.Fatalf("len(got)=%d, want %d", len(got), len(texts))
	}
	for i, r := range got {
		if r.Err != nil {
			t.Errorf("result %d: unexpected error %v", i, r.Err)
		}
		if len(r.Vector) != 1 || r.Vector[0] != float32(len(texts[i])) {
			t.Errorf("result %d: got vector %v, want [%d]", i, r.Vector, len(texts[i]))
		}
	}
}

// A single poisoned input must fail alone: the surrounding inputs still get
// vectors via the one-by-one fallback, and only the bad one carries an error.
func TestBatchEmbedResults_PoisonedInputFailsAlone(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			if len(texts) > 1 {
				return nil, &fakeError{"batch rejected"}
			}
			if texts[0] == "poison" {
				return nil, &fakeError{"input too large"}
			}
			return okEmbed(texts)
		},
	}

	texts := []string{"a", "poison", "ccc"}
	got, err := BatchEmbedResults(context.Background(), f, texts, 3, nil)
	if err != nil {
		t.Fatalf("BatchEmbedResults: %v", err)
	}
	for _, i := range []int{0, 2} {
		if got[i].Err != nil {
			t.Errorf("result %d: unexpected error %v", i, got[i].Err)
		}
		if got[i].Vector == nil {
			t.Errorf("result %d: want a vector from the fallback, got nil", i)
		}
	}
	if got[1].Vector != nil {
		t.Errorf("result 1: want nil vector, got %v", got[1].Vector)
	}
	if got[1].Err == nil {
		t.Fatal("result 1: want an error, got nil")
	}
	// The message a caller stores for diagnosis must be the individual
	// attempt's cause, not the generic batch failure.
	if !strings.Contains(got[1].Err.Error(), "input too large") {
		t.Errorf("result 1: error %q does not mention the per-input cause", got[1].Err)
	}
}

// ItemError.Index identifies the input across batch boundaries, so a caller
// zipping results back onto its own records can trust it.
func TestBatchEmbedResults_ItemErrorCarriesIndex(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			if len(texts) > 1 {
				return nil, &fakeError{"batch rejected"}
			}
			if texts[0] == "bad" {
				return nil, &fakeError{"nope"}
			}
			return okEmbed(texts)
		},
	}

	// "bad" is index 4, in the third batch of two.
	texts := []string{"a", "b", "c", "d", "bad", "f"}
	got, err := BatchEmbedResults(context.Background(), f, texts, 2, nil)
	if err != nil {
		t.Fatalf("BatchEmbedResults: %v", err)
	}
	var ie *ItemError
	if !errors.As(got[4].Err, &ie) {
		t.Fatalf("result 4: error %v is not an *ItemError", got[4].Err)
	}
	if ie.Index != 4 {
		t.Errorf("ItemError.Index = %d, want 4", ie.Index)
	}
}

// A dead backend fails every input with the same batch-level cause. Callers
// use ItemError.Batch to tell that apart from a scatter of bad inputs, so it
// must be populated and must not be swallowed by the fallback's own error.
func TestBatchEmbedResults_DeadBackendReportsBatchCause(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(_ []string) ([][]float32, error) {
			return nil, &fakeError{"connection refused"}
		},
	}

	got, err := BatchEmbedResults(context.Background(), f, []string{"a", "b"}, 2, nil)
	if err == nil {
		t.Fatal("want an error when every input fails")
	}
	if len(got) != 2 {
		t.Fatalf("len(got)=%d, want 2", len(got))
	}
	for i, r := range got {
		var ie *ItemError
		if !errors.As(r.Err, &ie) {
			t.Fatalf("result %d: error %v is not an *ItemError", i, r.Err)
		}
		if ie.Batch == nil {
			t.Errorf("result %d: ItemError.Batch is nil, want the batch-level cause", i)
		}
	}
}

// A poisoned input inside an otherwise healthy batch still records the batch
// failure, but the caller can see the batch recovered because its neighbours
// have vectors. What must NOT happen is Batch being set on an input whose
// batch never failed.
func TestBatchEmbedResults_NoBatchCauseWhenBatchSucceeded(t *testing.T) {
	f := &fakeEmbedder{model: "test", embed: okEmbed}

	got, err := BatchEmbedResults(context.Background(), f, []string{"a", "b"}, 2, nil)
	if err != nil {
		t.Fatalf("BatchEmbedResults: %v", err)
	}
	for i, r := range got {
		if r.Err != nil {
			t.Errorf("result %d: unexpected error %v", i, r.Err)
		}
	}
}

// A short response (fewer vectors than inputs, no error) is a batch-level
// anomaly too. Inputs that then fail individually must report it, so the
// caller is not left guessing why the fallback ran.
func TestBatchEmbedResults_ShortResponseRecordedAsBatchCause(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			if len(texts) > 1 {
				// One vector for two inputs — a short response.
				return [][]float32{{1}}, nil
			}
			return nil, &fakeError{"still broken"}
		},
	}

	got, err := BatchEmbedResults(context.Background(), f, []string{"a", "b"}, 2, nil)
	if err == nil {
		t.Fatal("want an error when every input fails")
	}
	var ie *ItemError
	if !errors.As(got[0].Err, &ie) {
		t.Fatalf("result 0: error %v is not an *ItemError", got[0].Err)
	}
	if !errors.Is(ie.Batch, ErrShortBatch) {
		t.Errorf("ItemError.Batch = %v, want ErrShortBatch", ie.Batch)
	}
}

// The error chain must survive wrapping so IsRetryable still classifies a
// permanent failure as permanent — that is how a caller decides whether to
// burn a retry budget on the input.
func TestBatchEmbedResults_PreservesPermanentError(t *testing.T) {
	permanent := &PermanentError{Err: &fakeError{"input length exceeds context"}, TooLong: true}
	f := &fakeEmbedder{
		model: "test",
		embed: func(_ []string) ([][]float32, error) { return nil, permanent },
	}

	got, _ := BatchEmbedResults(context.Background(), f, []string{"a"}, 1, nil)
	if len(got) != 1 {
		t.Fatalf("len(got)=%d, want 1", len(got))
	}
	if IsRetryable(got[0].Err) {
		t.Errorf("IsRetryable(%v) = true, want false — PermanentError was lost in wrapping", got[0].Err)
	}
	var pe *PermanentError
	if !errors.As(got[0].Err, &pe) || !pe.TooLong {
		t.Errorf("errors.As did not recover the TooLong PermanentError from %v", got[0].Err)
	}
}

// An input the backend accepts but returns no vector for is a failure, not a
// silent nil — otherwise it is indistinguishable from a skipped input.
func TestBatchEmbedResults_NoVectorIsAnError(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			if len(texts) > 1 {
				return nil, &fakeError{"batch rejected"}
			}
			return [][]float32{}, nil
		},
	}

	got, err := BatchEmbedResults(context.Background(), f, []string{"a", "b"}, 2, nil)
	if err == nil {
		t.Fatal("want an error when every input fails")
	}
	for i, r := range got {
		if r.Err == nil {
			t.Errorf("result %d: want an error for an empty response, got nil", i)
		}
	}
}

func TestBatchEmbedResults_EmptyTexts(t *testing.T) {
	f := &fakeEmbedder{
		model: "test",
		embed: func(_ []string) ([][]float32, error) {
			t.Error("Embed should not be called for empty input")
			return nil, nil
		},
	}
	got, err := BatchEmbedResults(context.Background(), f, nil, 10, nil)
	if err != nil {
		t.Fatalf("BatchEmbedResults on empty: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d results, want 0", len(got))
	}
}

func TestBatchEmbedResults_ProgressInvokedPerBatch(t *testing.T) {
	f := &fakeEmbedder{model: "test", embed: okEmbed}

	type progressCall struct{ done, total int }
	var calls []progressCall
	progress := func(done, total int) { calls = append(calls, progressCall{done, total}) }

	texts := []string{"a", "b", "c", "d", "e"}
	if _, err := BatchEmbedResults(context.Background(), f, texts, 2, progress); err != nil {
		t.Fatalf("BatchEmbedResults: %v", err)
	}
	want := []progressCall{{2, 5}, {4, 5}, {5, 5}}
	if len(calls) != len(want) {
		t.Fatalf("progress calls: got %d, want %d (%v)", len(calls), len(want), calls)
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Errorf("call %d: got %v, want %v", i, calls[i], want[i])
		}
	}
}

func TestBatchEmbedResults_ZeroBatchSizeUsesDefault(t *testing.T) {
	var f fakeEmbedder
	f.model = "test"
	f.embed = okEmbed

	texts := make([]string, 30)
	for i := range texts {
		texts[i] = "x"
	}
	if _, err := BatchEmbedResults(context.Background(), &f, texts, 0, nil); err != nil {
		t.Fatalf("BatchEmbedResults: %v", err)
	}
	if f.calls.Load() != 2 {
		t.Errorf("expected 2 batch calls with default size, got %d", f.calls.Load())
	}
}

// BatchEmbed keeps its documented contract — failures collapse to nil vectors
// — now that it is implemented on top of BatchEmbedResults.
func TestBatchEmbed_StillCollapsesFailuresToNil(t *testing.T) {
	var singleCalls atomic.Int32
	f := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			if len(texts) > 1 {
				return nil, &fakeError{"batch rejected"}
			}
			singleCalls.Add(1)
			if texts[0] == "poison" {
				return nil, &fakeError{"nope"}
			}
			return okEmbed(texts)
		},
	}

	got, err := BatchEmbed(context.Background(), f, []string{"a", "poison", "ccc"}, 3, nil)
	if err != nil {
		t.Fatalf("BatchEmbed: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len(got)=%d, want 3", len(got))
	}
	if got[0] == nil || got[2] == nil {
		t.Errorf("healthy inputs lost their vectors: %v", got)
	}
	if got[1] != nil {
		t.Errorf("failed input: got %v, want nil", got[1])
	}
	if singleCalls.Load() != 3 {
		t.Errorf("expected 3 single-call fallbacks, got %d", singleCalls.Load())
	}
}
