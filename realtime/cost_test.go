package realtime

import "testing"

// TestBillableTreatsCachedTokensAsASubset pins the measured payload shape.
// Getting this backwards — adding cached to the modality count instead of
// subtracting — inflates every bill, and does so invisibly because the total
// still looks plausible.
func TestBillableTreatsCachedTokensAsASubset(t *testing.T) {
	// Verbatim from the live API on 2026-08-23.
	u := &Usage{
		TotalTokens: 352, InputTokens: 123, OutputTokens: 229,
		InputDetails:  InputTokenDetails{TextTokens: 123, CachedTokens: 64},
		OutputDetails: OutputTokenDetails{TextTokens: 57, AudioTokens: 172},
	}
	u.InputDetails.CachedDetails.TextTokens = 64

	got := u.Billable()
	want := map[TokenClass]int64{
		ClassTextInput: 59, ClassTextInputCached: 64,
		ClassTextOutput: 57, ClassAudioOutput: 172,
	}
	for c, w := range want {
		if got[c] != w {
			t.Errorf("Billable()[%s] = %d, want %d", c, got[c], w)
		}
	}
	if total := sum(got); total != u.InputTokens+u.OutputTokens {
		t.Errorf("billable classes sum to %d, but the provider reported %d tokens",
			total, u.InputTokens+u.OutputTokens)
	}
}

// TestABreakdownlessUsageIsBilledHigh: a provider that reports totals without
// the per-class split must still be accounted for, and erring expensive means
// a spending guard refuses too early rather than too late.
func TestABreakdownlessUsageIsBilledHigh(t *testing.T) {
	u := &Usage{TotalTokens: 300, InputTokens: 100, OutputTokens: 200}
	got := u.Billable()
	if got[ClassAudioInput] != 100 || got[ClassAudioOutput] != 200 {
		t.Errorf("Billable() = %v; a usage with no breakdown was not attributed", got)
	}
}

// TestReasoningTokensAreNotBilledTwice pins the subset relationship measured
// on gpt-realtime-2.1: text 176 + audio 343 == output_tokens 519, with
// reasoning 42 already inside the text figure.
func TestReasoningTokensAreNotBilledTwice(t *testing.T) {
	u := &Usage{
		TotalTokens: 650, InputTokens: 131, OutputTokens: 519,
		InputDetails:  InputTokenDetails{TextTokens: 131},
		OutputDetails: OutputTokenDetails{TextTokens: 176, AudioTokens: 343, ReasoningTokens: 42},
	}
	got := u.Billable()
	if got[ClassTextOutput] != 176 {
		t.Errorf("text output billed as %d, want 176 — reasoning tokens are inside it", got[ClassTextOutput])
	}
	if total := sum(got); total != u.InputTokens+u.OutputTokens {
		t.Errorf("billable sums to %d, provider reported %d", total, u.InputTokens+u.OutputTokens)
	}
}
