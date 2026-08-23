package embedding

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

// RecipeFingerprint returns a short stable hash of everything besides the model
// name and vector dimension that determines a stored vector.
//
// Fingerprint records model and dim. Neither caller's vectors are determined by
// those alone: they also depend on the task prefix the text was wrapped in, the
// chunk size it was split at, and the header layout rendered into each chunk.
// Change any of those and the model and dim are unchanged, so a store keeps
// serving old-recipe vectors against new-recipe queries -- strictly worse than
// either recipe used consistently, and nothing reports it.
//
// A caller stores this alongside model and dim, and compares on startup. What
// to do about a mismatch is the caller's decision -- for a recipe change the
// useful answer is usually "clear the vectors and let the backfill re-run",
// which is a storage action this library cannot take.
//
// extra is for what the library cannot see: the caller's own header layout,
// field order, or any transformation applied before the text arrives here. Pass
// a short version marker ("header-v2") rather than the layout itself.
//
// # What moves it, and what deliberately does not
//
// Moves it: the task, whether the model resolves to task prefixes at all, and
// the configured chunk target. Those change where vectors sit in the space.
//
// Does not move it: packaging. An Ollama tag or a -GGUF suffix names the same
// weights, so the model is canonicalised first -- otherwise a serving-side
// rename would force a needless re-embed of the whole corpus.
//
// Also does not move it: overlap and the minimum-chunk floor. Those nudge
// vectors marginally rather than relocating them, and a signal that fires on
// every tuning tweak is one callers learn to ignore.
//
// The *configured* chunk target is used, not the resolved byte budget.
// BudgetForTokens moves as observations accumulate, so hashing the resolved
// figure would make the recipe drift during normal operation.
func RecipeFingerprint(model string, task Task, opts SplitOptions, extra ...string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "model=%s\n", canonicalModel(model))
	fmt.Fprintf(&b, "task=%s\n", task)
	fmt.Fprintf(&b, "prompts=%t\n", LookupTaskPrompter(model) != nil && FormatForTask(model, task, "\x00") != "\x00")
	fmt.Fprintf(&b, "maxTokens=%d\n", opts.MaxTokens)
	fmt.Fprintf(&b, "maxBytes=%d\n", opts.MaxBytes)
	fmt.Fprintf(&b, "structure=%d\n", opts.Structure)
	for _, e := range extra {
		fmt.Fprintf(&b, "extra=%s\n", e)
	}
	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:8])
}
