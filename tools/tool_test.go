package tools

import "testing"

// TestAToolWithNoPublishedRateIsNotFree is the whole reason CallRate returns
// two values. A zero handed back as a rate is spend recorded as free, and a
// month's total that silently omitted every search a household ran looks
// exactly like one that counted them.
func TestAToolWithNoPublishedRateIsNotFree(t *testing.T) {
	for _, tc := range []struct {
		name string
		tool Tool
	}{
		{"nothing published", Tool{ID: "web-search"}},
		{"an explicit zero", Tool{ID: "web-search", CostPer1KCalls: 0}},
		{"a negative, which is not a price", Tool{ID: "web-search", CostPer1KCalls: -1}},
	} {
		if rate, ok := tc.tool.CallRate(); ok {
			t.Errorf("%s: CallRate() = (%v, true), want it reported as unpublished", tc.name, rate)
		}
	}
}

func TestAPublishedRateIsReturnedInTheUnitItWasPublishedIn(t *testing.T) {
	// $10.00 / 1k calls, as the provider's own page states it. Not 10000: a
	// per-million rate here would put a factor of a thousand inside a
	// generated file, understating every bill computed from it.
	tool := Tool{ID: "web-search", Currency: "USD", CostPer1KCalls: 10}
	rate, ok := tool.CallRate()
	if !ok {
		t.Fatal("CallRate() reported a published rate as missing")
	}
	if rate != 10 {
		t.Errorf("CallRate() = %v, want 10 -- the unit is per thousand calls", rate)
	}
}
