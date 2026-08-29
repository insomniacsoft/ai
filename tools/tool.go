// Package tools carries what a provider's HOSTED TOOLS cost.
//
// A hosted tool -- web search, file search -- is billed separately from the
// model that called it, per invocation rather than per token, and every model
// pays the same rate for it. That is why it is neither a field on a model nor
// a catalog keyed by one: a caller asking "what did this search cost" has a
// tool name and a count, and no model id that would help.
//
// The tokens are still the model's. A provider's own note on the web-search
// rate reads "+ Search content tokens billed at model rates", so a call that
// searched is billed twice over: once here, per call, and once through the
// model's own catalog for the text it read. Anything totalling spend has to
// add both, and a total carrying only one of them looks exactly like a correct
// total.
package tools

// Tool is one hosted tool, with what a provider charges to invoke it.
//
// The rates here are a DEFAULT, not an authority -- the same stance the model
// catalogs take. A vendor's rate card changes without notice and this file is
// a snapshot of the day it was generated, so a caller that lets an operator
// state a rate must let that rate win. What this removes is the obligation to
// state one at all.
type Tool struct {
	// ID is the catalog key, as the provider names the tool.
	ID string
	// Name is the human-readable name, for a chooser or a report.
	Name string
	// Provider is the service this tool is served by.
	Provider string
	// APIModel is the id the provider's own API uses, which is not always the
	// catalog key.
	APIModel string

	// Currency is the ISO 4217 code the cost fields are denominated in.
	Currency string

	// CostPer1KCalls is the cost of one THOUSAND invocations, in Currency.
	//
	// Per thousand, and named for it, because that is the unit the provider
	// publishes -- OpenAI's page reads "$10.00 / 1k calls". Converting to a
	// per-million rate here to match the token catalogs would bury a factor of
	// a thousand inside a generated file, in the one direction nobody audits:
	// a rate quietly a thousand times too small understates a bill and looks
	// like a small bill. A caller whose arithmetic wants per-million multiplies
	// at the point it does so, where the factor is visible and testable.
	//
	// Zero means the source publishes no per-call rate for this tool -- not
	// that invoking it is free. See CallRate.
	CostPer1KCalls float64
}

// CallRate returns what one thousand invocations cost, and whether the catalog
// publishes a rate at all.
//
// Two values for the same reason realtime.Model.Rate returns two: a missing
// rate handed back as a bare zero is spend recorded as free, and a total that
// silently omits every search a household ran reads exactly like a total that
// counted them and found nothing.
func (t Tool) CallRate() (float64, bool) {
	if t.CostPer1KCalls <= 0 {
		return 0, false
	}
	return t.CostPer1KCalls, true
}
