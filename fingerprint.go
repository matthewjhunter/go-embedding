package embedding

import "fmt"

// Fingerprint identifies what a stored vector was produced by. It is stronger
// than Model() alone because two model versions can share a name while
// producing incompatible vectors (e.g. nomic-embed-text v1 vs v2), and because
// the same model produces vectors in a different region of the space when the
// text reaching it is assembled differently.
//
// Its fields are *progressively knowable*, and the zero value of each means
// "not known yet" rather than "empty":
//
//   - Model and Recipe come from configuration, so both are known before a
//     single vector is read.
//   - Dim is populated only after the first successful Embed call.
//
// That is why Reconcile rather than a single equality check: at store-open --
// the moment a mismatch is most worth catching -- the current Dim is always 0,
// so comparing whole fingerprints could never work there.
type Fingerprint struct {
	Model string
	Dim   int
	// Recipe identifies everything besides the model and dimension that
	// determines a vector: the task prefix, the chunk size, the header layout.
	// See RecipeFingerprint. Empty means unknown, which is what a store written
	// before recipes were recorded looks like.
	Recipe string
}

// MismatchError reports a Fingerprint comparison failure.
type MismatchError struct {
	Stored  Fingerprint
	Current Fingerprint
}

func (e *MismatchError) Error() string {
	return fmt.Sprintf(
		"embedding: fingerprint mismatch: stored=%s/%d/%s, current=%s/%d/%s",
		e.Stored.Model, e.Stored.Dim, recipeOrUnknown(e.Stored.Recipe),
		e.Current.Model, e.Current.Dim, recipeOrUnknown(e.Current.Recipe),
	)
}

func recipeOrUnknown(r string) string {
	if r == "" {
		return "recipe?"
	}
	return r
}

// CheckFingerprint returns nil if stored and current match exactly, or a
// *MismatchError otherwise.
//
// Prefer Reconcile. This compares whole fingerprints, including fields neither
// side may know yet, so it cannot be used at store-open: the current Dim is 0
// until the first Embed call, and would read as a conflict with the stored one.
// That is why it has no callers -- both known consumers hand-rolled a narrower
// comparison instead.
func CheckFingerprint(stored, current Fingerprint) error {
	if stored != current {
		return &MismatchError{Stored: stored, Current: current}
	}
	return nil
}

// Reconcile compares what stored and current both know, and returns the
// fingerprint that should be persisted.
//
// A field is compared only when both sides know it. The zero value means "not
// known yet", which is the normal state rather than an error case: at
// store-open the current Dim has not been observed, and a store written before
// recipes were recorded has no Recipe. Treating either as a conflict would fire
// on every upgrade and every startup, and a check that cries wolf is one
// callers route around -- which is exactly how CheckFingerprint ended up unused.
//
// Where the stored side does not know a field and the current side does, the
// value is adopted into the result. So the first run records, and the run after
// that is checked. A recipe introduced by an upgrade is therefore adopted
// silently and the *next* change is caught, which is the right trade: forcing a
// re-embed on every existing store at upgrade would be worse than detecting one
// change later.
//
// A field both sides know and disagree on is a *MismatchError. The caller
// decides what to do about it -- for a model or dimension change the stored
// vectors cannot be compared at all, while for a recipe change they can be
// compared but should not be mixed, and clearing them lets a backfill rebuild.
// Neither remedy is something this library can perform.
func Reconcile(stored, current Fingerprint) (Fingerprint, error) {
	out := stored
	mismatch := false

	if current.Model != "" {
		if stored.Model != "" && stored.Model != current.Model {
			mismatch = true
		}
		out.Model = current.Model
	}
	if current.Dim != 0 {
		if stored.Dim != 0 && stored.Dim != current.Dim {
			mismatch = true
		}
		out.Dim = current.Dim
	}
	if current.Recipe != "" {
		if stored.Recipe != "" && stored.Recipe != current.Recipe {
			mismatch = true
		}
		out.Recipe = current.Recipe
	}

	if mismatch {
		return Fingerprint{}, &MismatchError{Stored: stored, Current: current}
	}
	return out, nil
}
