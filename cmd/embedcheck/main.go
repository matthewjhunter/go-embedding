// Command embedcheck reports whether an embedding backend is working, and
// whether it is working as configured.
//
// Those are two separate questions and both fail quietly. A backend can return
// well-formed vectors of the right dimension that carry almost no retrieval
// signal, and a correctly-served model can be named in a way the prefix
// registry does not recognise, so text reaches it unwrapped. Neither raises an
// error; both just make every result worse.
//
//	embedcheck --base-url http://host:13305/api --model embeddinggemma
//
// Configuration otherwise comes from the same EMBEDDING_* environment the
// library reads, so checking what a service actually uses needs no flags:
//
//	embedcheck
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	embedding "github.com/matthewjhunter/go-embedding"
)

func main() {
	cfg, err := embedding.ConfigFromEnv()
	if err != nil {
		// Not fatal: the flags below can supply everything, and a broken
		// environment is a thing this tool should help diagnose rather than
		// refuse over.
		fmt.Fprintf(os.Stderr, "embedcheck: environment config: %v\n", err)
		cfg = embedding.DefaultConfig()
	}
	backend := flag.String("backend", string(cfg.Backend), "backend: openai|ollama")
	baseURL := flag.String("base-url", cfg.BaseURL, "backend base URL")
	model := flag.String("model", cfg.Model, "model name as the backend serves it")
	apiKey := flag.String("api-key", cfg.APIKey, "API key, if the backend needs one")
	timeout := flag.Duration("timeout", 5*time.Minute, "overall deadline")
	quiet := flag.Bool("quiet", false, "print only the verdict line")
	flag.Parse()

	cfg.Backend = embedding.Backend(*backend)
	cfg.BaseURL, cfg.Model, cfg.APIKey = *baseURL, *model, *apiKey
	// Strict mode is off here on purpose: an unrecognised model is exactly
	// what this tool exists to report, so refusing to run would withhold the
	// answer someone came for.
	cfg.StrictModel = false

	e, err := embedding.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embedcheck: %v\n", err)
		os.Exit(2)
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	r, err := embedding.CheckHealth(ctx, e)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embedcheck: %v\n", err)
		os.Exit(2)
	}

	if !*quiet {
		report(os.Stdout, r)
	}
	fmt.Printf("VERDICT: %s\n", r.Verdict)

	// Exit status so this can gate a deploy: 0 healthy, 1 weak, 3 broken.
	switch r.Verdict {
	case embedding.VerdictBroken:
		os.Exit(3)
	case embedding.VerdictWeak:
		os.Exit(1)
	}
}

func report(out *os.File, r embedding.HealthReport) {
	w := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)

	fmt.Fprintf(w, "MODEL\n")
	fmt.Fprintf(w, "  served name\t%s\n", r.Model)
	fmt.Fprintf(w, "  resolves to\t%s\n", r.Canonical)
	fmt.Fprintf(w, "  recognised\t%v\n", r.Recognised)
	fmt.Fprintf(w, "  task prefixes\t%v\n", yesNo(r.HasPrompts, "applied", "NONE -- text reaches the model unwrapped"))
	if r.Limits.MaxBytes > 0 || r.Limits.MaxTokens > 0 {
		fmt.Fprintf(w, "  input budget\t%d bytes / %d tokens\n", r.Limits.MaxBytes, r.Limits.MaxTokens)
	} else {
		fmt.Fprintf(w, "  input budget\tunknown -- oversized inputs are not trimmed\n")
	}
	fmt.Fprintf(w, "  dimension\t%d\n", r.Dim)

	fmt.Fprintf(w, "\nDISCRIMINATION\t(four topics, two paraphrases each, within this model's own space)\n")
	fmt.Fprintf(w, "  paraphrase pairs\t%+.4f mean cosine\n", r.SameMean)
	fmt.Fprintf(w, "  unrelated pairs\t%+.4f mean cosine\n", r.DiffMean)
	fmt.Fprintf(w, "  separation\t%+.4f\n", r.Separation)
	fmt.Fprintf(w, "  worst margin\t%+.4f\t%s\n", r.WorstMargin,
		yesNo(r.WorstMargin > 0, "every paraphrase outranks every unrelated pair",
			"an unrelated pair outranks a paraphrase -- ranking is unreliable"))

	fmt.Fprintf(w, "\nTIMING\t(warm)\n")
	fmt.Fprintf(w, "  single text\t%v\n", r.SingleLatency.Round(time.Millisecond))
	fmt.Fprintf(w, "  batch of %d\t%v\n", r.BatchSize, r.BatchLatency.Round(time.Millisecond))
	fmt.Fprintf(w, "  throughput\t%.1f texts/s\n", r.TextsPerSecond)

	fmt.Fprintf(w, "\nFOR COMPARISON\t(measured on one machine; see ReferenceReports)\n")
	fmt.Fprintf(w, "  model\tsame\tdiff\tsep\tmargin\trate\tverdict\n")
	for _, ref := range embedding.ReferenceReports {
		fmt.Fprintf(w, "  %s\t%+.4f\t%+.4f\t%+.4f\t%+.4f\t%.1f/s\t%s\n",
			ref.Model, ref.SameMean, ref.DiffMean, ref.Separation, ref.WorstMargin,
			ref.TextsPerSecond, ref.Verdict)
	}

	if len(r.Notes) > 0 {
		fmt.Fprintf(w, "\nNOTES\n")
		for _, n := range r.Notes {
			fmt.Fprintf(w, "  - %s\n", n)
		}
	}
	fmt.Fprintln(w)
	w.Flush()
}

func yesNo(b bool, yes, no string) string {
	if b {
		return yes
	}
	return no
}
