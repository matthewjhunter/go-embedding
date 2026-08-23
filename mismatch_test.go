package embedding

import (
	"errors"
	"testing"
)

// The three mismatches want different remedies. A dimension change means the
// stored vectors cannot be compared at all. A model change means they compare
// but the results are garbage. A recipe change means they compare and the
// results are merely degraded -- and it is the one a caller can resolve alone,
// by clearing vectors and letting a backfill rebuild, because it follows a
// deliberate config edit rather than an accident.

func recipeChange() error {
	_, err := Reconcile(
		Fingerprint{Model: "m", Dim: 768, Recipe: "old"},
		Fingerprint{Model: "m", Dim: 768, Recipe: "new"},
	)
	return err
}

func TestMismatch_RecipeChangeMatchesOnlyItsSentinel(t *testing.T) {
	err := recipeChange()

	if !errors.Is(err, ErrRecipeChanged) {
		t.Error("a recipe change does not match ErrRecipeChanged")
	}
	for name, sentinel := range map[string]error{"model": ErrModelChanged, "dim": ErrDimChanged} {
		if errors.Is(err, sentinel) {
			t.Errorf("a recipe change also matched the %s sentinel", name)
		}
	}
}

func TestMismatch_ModelChangeMatchesItsSentinel(t *testing.T) {
	_, err := Reconcile(
		Fingerprint{Model: "old-model", Dim: 768, Recipe: "r"},
		Fingerprint{Model: "new-model", Dim: 768, Recipe: "r"},
	)
	if !errors.Is(err, ErrModelChanged) {
		t.Error("a model change does not match ErrModelChanged")
	}
	if errors.Is(err, ErrRecipeChanged) {
		t.Error("a model change matched ErrRecipeChanged; a caller would clear vectors on an accident")
	}
}

func TestMismatch_DimChangeMatchesItsSentinel(t *testing.T) {
	_, err := Reconcile(
		Fingerprint{Model: "m", Dim: 768, Recipe: "r"},
		Fingerprint{Model: "m", Dim: 1024, Recipe: "r"},
	)
	if !errors.Is(err, ErrDimChanged) {
		t.Error("a dimension change does not match ErrDimChanged")
	}
}

// A model swap usually moves the dimension too. Both must match, or a caller
// branching on one of them silently mishandles the other.
func TestMismatch_SeveralFieldsMatchSeveralSentinels(t *testing.T) {
	_, err := Reconcile(
		Fingerprint{Model: "nomic-embed-text", Dim: 768, Recipe: "r"},
		Fingerprint{Model: "embeddinggemma", Dim: 1024, Recipe: "r"},
	)
	for name, sentinel := range map[string]error{"model": ErrModelChanged, "dim": ErrDimChanged} {
		if !errors.Is(err, sentinel) {
			t.Errorf("a combined model+dim change did not match the %s sentinel", name)
		}
	}
}

// The existing type must keep working for callers that inspect the fingerprints.
func TestMismatch_StillUnwrapsToMismatchError(t *testing.T) {
	var me *MismatchError
	if !errors.As(recipeChange(), &me) {
		t.Fatal("no longer unwraps to *MismatchError")
	}
	if me.Stored.Recipe != "old" || me.Current.Recipe != "new" {
		t.Errorf("fingerprints lost: stored=%q current=%q", me.Stored.Recipe, me.Current.Recipe)
	}
}

// The query that decides the caller's behaviour, and the one that is easy to
// get subtly wrong by forgetting to assert the absence of the others.
func TestRecipeOnly(t *testing.T) {
	if !RecipeOnly(recipeChange()) {
		t.Error("RecipeOnly is false for a recipe-only change")
	}

	_, both := Reconcile(
		Fingerprint{Model: "a", Dim: 768, Recipe: "old"},
		Fingerprint{Model: "b", Dim: 768, Recipe: "new"},
	)
	if RecipeOnly(both) {
		t.Error("RecipeOnly is true when the model moved too; vectors would be cleared on an accident")
	}

	if RecipeOnly(nil) {
		t.Error("RecipeOnly(nil) is true")
	}
	if RecipeOnly(errors.New("unrelated")) {
		t.Error("RecipeOnly is true for an unrelated error")
	}
}
