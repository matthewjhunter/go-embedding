package embedding

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// BatchEmbedItems is the record-shaped counterpart to BatchEmbedResults.
// Both callers wrote this flatten-and-regroup by hand and reached the same
// design, including the part that is only obvious in hindsight: one bad chunk
// must fail its whole record (#29).

// identityEmbedder returns a vector encoding each text, so regrouping can be
// checked by content rather than by count alone.
func identityEmbedder(fail func(text string) error) *fakeEmbedder {
	return &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			out := make([][]float32, len(texts))
			for i, txt := range texts {
				if fail != nil {
					if err := fail(txt); err != nil {
						return nil, err
					}
				}
				out[i] = []float32{float32(len(txt))}
			}
			return out, nil
		},
	}
}

func items(groups ...[]string) []BatchItem {
	out := make([]BatchItem, len(groups))
	for i, g := range groups {
		out[i] = BatchItem{Texts: g}
	}
	return out
}

func TestBatchEmbedItems_RegroupsVectorsOntoTheirItem(t *testing.T) {
	e := identityEmbedder(nil)
	got := BatchEmbedItems(context.Background(), e,
		items([]string{"a"}, []string{"bb", "ccc", "dddd"}, []string{"ee"}), 2)

	if len(got) != 3 {
		t.Fatalf("got %d results for 3 items", len(got))
	}
	want := []int{1, 3, 1}
	for i, n := range want {
		if got[i].Err != nil {
			t.Fatalf("item %d: %v", i, got[i].Err)
		}
		if len(got[i].Vectors) != n {
			t.Errorf("item %d got %d vectors, want %d", i, len(got[i].Vectors), n)
		}
	}
	// Item 1's chunks were "bb","ccc","dddd" -- lengths 2,3,4, in order.
	for j, want := range []float32{2, 3, 4} {
		if got[1].Vectors[j][0] != want {
			t.Errorf("item 1 vector %d = %v, want %v -- vectors are misaligned",
				j, got[1].Vectors[j][0], want)
		}
	}
}

// The rule that is not obvious until it bites: a half-embedded record looks
// *complete* to the retry query -- its marker is set, so the backfill skips it
// -- while leaving a permanent hole in the middle of it.
func TestBatchEmbedItems_OneBadChunkFailsItsWholeItem(t *testing.T) {
	e := identityEmbedder(func(text string) error {
		if strings.Contains(text, "poison") {
			return errors.New("input too long")
		}
		return nil
	})

	got := BatchEmbedItems(context.Background(), e,
		items([]string{"fine"}, []string{"ok", "poison", "ok2"}, []string{"also fine"}), 8)

	if got[1].Err == nil {
		t.Error("item 1 contained an unembeddable chunk but reported success")
	}
	if len(got[1].Vectors) != 0 {
		t.Errorf("item 1 kept %d vectors alongside an error; a partial set must never be returned",
			len(got[1].Vectors))
	}
	for _, i := range []int{0, 2} {
		if got[i].Err != nil {
			t.Errorf("item %d failed because a *different* item was poisoned: %v", i, got[i].Err)
		}
		if len(got[i].Vectors) == 0 {
			t.Errorf("item %d got no vectors", i)
		}
	}
}

// An item with nothing to embed is a deterministic skip, not a failure.
func TestBatchEmbedItems_EmptyItemIsASkipNotAnError(t *testing.T) {
	got := BatchEmbedItems(context.Background(), identityEmbedder(nil),
		items([]string{"a"}, nil, []string{"b"}), 4)

	if got[1].Err != nil {
		t.Errorf("empty item reported an error: %v", got[1].Err)
	}
	if len(got[1].Vectors) != 0 {
		t.Errorf("empty item produced %d vectors", len(got[1].Vectors))
	}
}

// Chunks are batched across items, so batchSize counts inputs rather than
// records -- one request per record would defeat the point.
func TestBatchEmbedItems_BatchesAcrossItems(t *testing.T) {
	e := identityEmbedder(nil)
	BatchEmbedItems(context.Background(), e,
		items([]string{"a"}, []string{"b"}, []string{"c"}, []string{"d"}, []string{"e"}, []string{"f"}), 3)

	if n := e.calls.Load(); n != 2 {
		t.Errorf("made %d embed calls for 6 chunks at batch size 3, want 2", n)
	}
}

// A backend that breaks the index-alignment contract must fail every item
// loudly rather than silently zipping the wrong vectors onto the wrong records.
func TestBatchEmbedItems_ShortResponseFailsEveryItem(t *testing.T) {
	e := &fakeEmbedder{
		model: "test",
		embed: func(texts []string) ([][]float32, error) {
			return make([][]float32, len(texts)-1), nil // one short, no error
		},
	}

	got := BatchEmbedItems(context.Background(), e, items([]string{"a", "b"}, []string{"c"}), 8)

	for i, r := range got {
		if r.Err == nil {
			t.Errorf("item %d reported success against a short response", i)
		}
	}
}

func TestBatchEmbedItems_NoItemsIsNoCalls(t *testing.T) {
	e := identityEmbedder(nil)
	if got := BatchEmbedItems(context.Background(), e, nil, 4); len(got) != 0 {
		t.Errorf("got %d results for no items", len(got))
	}
	if n := e.calls.Load(); n != 0 {
		t.Errorf("made %d embed calls for no items", n)
	}
}
