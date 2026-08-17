package embedding

import (
	"context"
	"time"
)

// Request deadline defaults. They are deliberately generous: the point is to
// catch a backend that has stopped responding, not to police a slow one. A
// 25-input batch gets 30s + 25*2s = 80s, against roughly 8.5s measured for
// that batch on an Intel Arc A380 running nomic-embed-text at production text
// lengths.
const (
	// DefaultTimeout is the base per-request deadline used when
	// Config.Timeout is zero.
	DefaultTimeout = 30 * time.Second
	// DefaultPerInputTimeout is the per-input allowance added to the base
	// when Config.PerInputTimeout is zero.
	DefaultPerInputTimeout = 2 * time.Second
)

// NoTimeout disables a deadline. As Config.Timeout it makes requests
// unbounded, leaving cancellation entirely to the caller's context (the
// behaviour before deadlines existed). As Config.PerInputTimeout it disables
// only the per-input scaling, giving every request the flat base budget.
const NoTimeout time.Duration = -1

// timeouts is the resolved per-request deadline policy shared by the HTTP
// backends. A negative base means deadlines are disabled.
type timeouts struct {
	base     time.Duration
	perInput time.Duration
}

// resolveTimeouts turns the configured (possibly zero) durations into a
// policy. Zero means "use the default" rather than "disable", so a caller who
// never thinks about timeouts still gets one; NoTimeout is the explicit opt
// out.
func resolveTimeouts(base, perInput time.Duration) timeouts {
	if base < 0 {
		return timeouts{base: NoTimeout}
	}
	if base == 0 {
		base = DefaultTimeout
	}
	switch {
	case perInput < 0:
		perInput = 0
	case perInput == 0:
		perInput = DefaultPerInputTimeout
	}
	return timeouts{base: base, perInput: perInput}
}

// disabled reports whether this policy sets no deadline at all.
func (t timeouts) disabled() bool { return t.base < 0 }

// budget returns the deadline for a request carrying n inputs. Request cost
// scales with the batch, so the budget does too — one flat number would either
// trip on a large batch or be too loose to catch a hang on a small one.
func (t timeouts) budget(n int) time.Duration {
	return t.base + t.perInput*time.Duration(n)
}

// apply derives the request context for a call carrying n inputs. The returned
// cancel func must always be called. A caller's own deadline, if shorter,
// still wins — context.WithTimeout takes the earlier of the two — and caller
// cancellation still propagates.
//
// The deadline is applied per HTTP request rather than per API call on
// purpose. BatchEmbedResults retries a failed batch one input at a time, and a
// budget spanning the whole call would already be spent by the time the
// fallback ran, making it useless exactly when it is needed.
func (t timeouts) apply(ctx context.Context, n int) (context.Context, context.CancelFunc) {
	if t.disabled() {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, t.budget(n))
}
