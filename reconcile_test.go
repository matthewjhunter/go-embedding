package embedding

import (
	"errors"
	"testing"
)

// A fingerprint is progressively knowable. Model and Recipe come from
// configuration and are known before a single vector is read; Dim is only known
// once the backend has answered. CheckFingerprint assumed all three at once,
// which is why it has no callers -- at store-open the current Dim is always
// zero, so the comparison could never be run there.
//
// Reconcile models that: it compares what both sides actually know, and returns
// the fingerprint to persist.

func TestReconcile_FirstAdoptionRecordsWithoutError(t *testing.T) {
	cur := Fingerprint{Model: "nomic-embed-text", Recipe: "abc123"}

	got, err := Reconcile(Fingerprint{}, cur)
	if err != nil {
		t.Fatalf("nothing was stored yet, so there is nothing to conflict with: %v", err)
	}
	if got != cur {
		t.Errorf("Reconcile = %+v, want %+v", got, cur)
	}
}

// The case CheckFingerprint could not serve: at store-open the dimension is not
// known yet, and that must not read as a conflict with the stored one.
func TestReconcile_UnknownDimAtOpenKeepsTheStoredOne(t *testing.T) {
	stored := Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "abc123"}
	atOpen := Fingerprint{Model: "nomic-embed-text", Recipe: "abc123"} // Dim unknown

	got, err := Reconcile(stored, atOpen)
	if err != nil {
		t.Fatalf("an unknown dimension was treated as a conflict: %v", err)
	}
	if got.Dim != 768 {
		t.Errorf("Dim = %d, want the stored 768 carried forward", got.Dim)
	}
}

// The other moment: the first embed learns the dimension, which must end up in
// what gets persisted.
func TestReconcile_LearnedDimIsRecorded(t *testing.T) {
	stored := Fingerprint{Model: "nomic-embed-text", Recipe: "abc123"} // no Dim yet
	afterEmbed := Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "abc123"}

	got, err := Reconcile(stored, afterEmbed)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dim != 768 {
		t.Errorf("Dim = %d, want the newly observed 768", got.Dim)
	}
}

func TestReconcile_ConflictsAreErrors(t *testing.T) {
	base := Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "abc123"}
	cases := map[string]Fingerprint{
		"model":  {Model: "embeddinggemma", Dim: 768, Recipe: "abc123"},
		"dim":    {Model: "nomic-embed-text", Dim: 1024, Recipe: "abc123"},
		"recipe": {Model: "nomic-embed-text", Dim: 768, Recipe: "def456"},
	}
	for field, cur := range cases {
		t.Run(field, func(t *testing.T) {
			_, err := Reconcile(base, cur)
			if err == nil {
				t.Fatalf("a changed %s was accepted; stored vectors would be mixed with new ones", field)
			}
			var me *MismatchError
			if !errors.As(err, &me) {
				t.Errorf("error is %T, want *MismatchError", err)
			}
		})
	}
}

// A recipe change is the one that used to be invisible: model and dim are
// unchanged, so nothing in the old fingerprint noticed.
func TestReconcile_RecipeChangeIsVisibleWhereModelAndDimAreNot(t *testing.T) {
	stored := Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "old"}
	cur := Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "new"}

	if stored.Model != cur.Model || stored.Dim != cur.Dim {
		t.Fatal("test needs model and dim identical")
	}
	if _, err := Reconcile(stored, cur); err == nil {
		t.Error("a recipe-only change was accepted -- exactly the silent case this exists to catch")
	}
}

// A store written before recipes existed has none recorded. That is unknown,
// not a conflict: forcing a re-embed on everyone at upgrade would be worse than
// detecting the *next* change.
func TestReconcile_MissingStoredRecipeIsAdoptedNotRejected(t *testing.T) {
	stored := Fingerprint{Model: "nomic-embed-text", Dim: 768} // pre-recipe store
	cur := Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "abc123"}

	got, err := Reconcile(stored, cur)
	if err != nil {
		t.Fatalf("an absent stored recipe was treated as a conflict: %v", err)
	}
	if got.Recipe != "abc123" {
		t.Errorf("Recipe = %q, want it adopted so the next change is caught", got.Recipe)
	}
}

func TestReconcile_MatchingFingerprintIsUnchanged(t *testing.T) {
	fp := Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "abc123"}

	got, err := Reconcile(fp, fp)
	if err != nil {
		t.Fatal(err)
	}
	if got != fp {
		t.Errorf("Reconcile = %+v, want it unchanged", got)
	}
}

// The error has to say which field moved, or the operator cannot tell an
// incompatible model from a recipe they can clear and rebuild.
func TestMismatchError_NamesTheRecipe(t *testing.T) {
	_, err := Reconcile(
		Fingerprint{Model: "m", Dim: 768, Recipe: "old"},
		Fingerprint{Model: "m", Dim: 768, Recipe: "new"},
	)
	if err == nil {
		t.Fatal("expected an error")
	}
	msg := err.Error()
	for _, want := range []string{"old", "new"} {
		if !contains(msg, want) {
			t.Errorf("error %q does not mention %q", msg, want)
		}
	}
}
