package embedding

import "fmt"

// Fingerprint identifies an embedding-model+vector-shape combination. It is
// stronger than Model() alone because two model versions can share a name
// while producing incompatible vectors (e.g. nomic-embed-text v1 vs v2).
//
// Dim is populated after the first successful Embed call; it is 0 until then.
type Fingerprint struct {
	Model string
	Dim   int
}

// MismatchError reports a Fingerprint comparison failure.
type MismatchError struct {
	Stored  Fingerprint
	Current Fingerprint
}

func (e *MismatchError) Error() string {
	return fmt.Sprintf(
		"embedding: fingerprint mismatch: stored=%s/%d, current=%s/%d",
		e.Stored.Model, e.Stored.Dim,
		e.Current.Model, e.Current.Dim,
	)
}

// CheckFingerprint returns nil if stored and current match, or a
// *MismatchError otherwise. Callers should run this on store-open against
// the persisted fingerprint to catch silent garbage-ranking caused by an
// embedding-model change.
func CheckFingerprint(stored, current Fingerprint) error {
	if stored != current {
		return &MismatchError{Stored: stored, Current: current}
	}
	return nil
}
