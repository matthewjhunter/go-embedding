package embedding

import "testing"

// Fingerprint covers model and dim, but vectors also depend on the task
// prefix, the chunk size and the caller's header layout. A recipe change that
// leaves model and dim alone is invisible today, so a store keeps serving
// old-recipe vectors against new-recipe queries (#32).

func TestRecipeFingerprint_TaskChangeMovesIt(t *testing.T) {
	a := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, SplitOptions{MaxTokens: 512})
	b := RecipeFingerprint("nomic-embed-text", TaskClustering, SplitOptions{MaxTokens: 512})

	if a == b {
		t.Error("changing the task did not move the recipe; those vectors sit in different regions")
	}
}

func TestRecipeFingerprint_ChunkSizeChangeMovesIt(t *testing.T) {
	a := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, SplitOptions{MaxTokens: 512})
	b := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, SplitOptions{MaxTokens: 256})

	if a == b {
		t.Error("changing the chunk target did not move the recipe")
	}
}

// The caller owns its header layout, so it contributes that itself.
func TestRecipeFingerprint_CallerExtrasMoveIt(t *testing.T) {
	opts := SplitOptions{MaxTokens: 512}
	a := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, opts, "header-v1")
	b := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, opts, "header-v2")

	if a == b {
		t.Error("caller extras did not move the recipe")
	}
}

// A model whose prefixes resolve differently produces different vectors even
// for the same task.
func TestRecipeFingerprint_PrefixAvailabilityMovesIt(t *testing.T) {
	known := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, SplitOptions{MaxTokens: 512})
	unknown := RecipeFingerprint("no-such-model-anywhere", TaskRetrievalDocument, SplitOptions{MaxTokens: 512})

	if known == unknown {
		t.Error("a model with no registered prompter shares a recipe with one that has prefixes")
	}
}

// Packaging is not recipe. An Ollama tag names the same weights, so it must not
// invalidate a corpus.
func TestRecipeFingerprint_TagIsNotARecipeChange(t *testing.T) {
	bare := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, SplitOptions{MaxTokens: 512})
	tagged := RecipeFingerprint("nomic-embed-text:latest", TaskRetrievalDocument, SplitOptions{MaxTokens: 512})

	if bare != tagged {
		t.Error("an Ollama tag moved the recipe; that would force a needless re-embed")
	}
}

// Tuning knobs that nudge vectors marginally must NOT trip it, or callers learn
// to ignore the signal.
func TestRecipeFingerprint_OverlapTuningDoesNotTripIt(t *testing.T) {
	a := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, SplitOptions{MaxTokens: 512, Overlap: 64})
	b := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, SplitOptions{MaxTokens: 512, Overlap: 128})

	if a != b {
		t.Error("an overlap tweak moved the recipe; a signal that fires on tuning gets ignored")
	}
}

// Stability matters: the same inputs must hash the same across processes, or
// every restart looks like a recipe change.
func TestRecipeFingerprint_IsStable(t *testing.T) {
	opts := SplitOptions{MaxTokens: 512}
	if a, b := RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, opts),
		RecipeFingerprint("nomic-embed-text", TaskRetrievalDocument, opts); a != b {
		t.Errorf("same inputs hashed to %q and %q", a, b)
	}
}
