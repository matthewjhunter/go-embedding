package embedding

import (
	"errors"
	"fmt"
)

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

// The three ways a fingerprint can move. They are separate sentinels because
// they want different remedies, and a caller that cannot tell them apart has to
// pick the most conservative one for all three.
//
// A dimension change means the stored vectors cannot be compared at all -- the
// arithmetic does not work. A model change means they compare and the results
// are garbage. Both usually mean something changed underneath the deployment
// rather than in it: a tag moved, a backend swapped a model. Clearing vectors
// automatically would hide that.
//
// A recipe change means the vectors compare and the results are merely
// degraded, and it follows a deliberate edit -- a task prefix added, a chunk
// size tuned. That is the one a caller can resolve alone, by clearing vectors
// and letting a backfill rebuild them.
//
// Match with errors.Is. A single mismatch can carry several of these at once:
// swapping a model usually moves the dimension too, so branching on one alone
// silently mishandles the other. See RecipeOnly.
var (
	ErrModelChanged  = errors.New("embedding: model changed")
	ErrDimChanged    = errors.New("embedding: dimension changed")
	ErrRecipeChanged = errors.New("embedding: recipe changed")
)

// MismatchError reports a Fingerprint comparison failure. It wraps one sentinel
// per field that moved, so errors.Is identifies which, while errors.As still
// yields the fingerprints themselves.
type MismatchError struct {
	Stored  Fingerprint
	Current Fingerprint
	// changed is the sentinel per field that moved, in field order.
	changed []error
}

// Unwrap exposes the per-field sentinels to errors.Is.
func (e *MismatchError) Unwrap() []error { return e.changed }

// RecipeOnly reports whether err is a mismatch in which the recipe moved and
// nothing else.
//
// It exists because that is the query a caller's behaviour turns on -- clear
// and rebuild, or stop and ask a human -- and it is the one easiest to get
// subtly wrong with errors.Is alone, since answering it correctly means
// asserting the *absence* of the other two rather than the presence of one.
func RecipeOnly(err error) bool {
	return errors.Is(err, ErrRecipeChanged) &&
		!errors.Is(err, ErrModelChanged) &&
		!errors.Is(err, ErrDimChanged)
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
	var changed []error

	if current.Model != "" {
		if stored.Model != "" && stored.Model != current.Model {
			changed = append(changed, ErrModelChanged)
		}
		out.Model = current.Model
	}
	if current.Dim != 0 {
		if stored.Dim != 0 && stored.Dim != current.Dim {
			changed = append(changed, ErrDimChanged)
		}
		out.Dim = current.Dim
	}
	if current.Recipe != "" {
		if stored.Recipe != "" && stored.Recipe != current.Recipe {
			changed = append(changed, ErrRecipeChanged)
		}
		out.Recipe = current.Recipe
	}

	if len(changed) > 0 {
		return Fingerprint{}, &MismatchError{Stored: stored, Current: current, changed: changed}
	}
	return out, nil
}
