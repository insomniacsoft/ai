package model

import "testing"

func TestGemini35FlashLiteCatalogEntry(t *testing.T) {
	m, ok := GeminiModels[Gemini35FlashLite]
	if !ok {
		t.Fatal("Gemini35FlashLite is missing from GeminiModels")
	}
	if m.APIModel != "gemini-3.5-flash-lite" {
		t.Errorf("APIModel = %q, want gemini-3.5-flash-lite", m.APIModel)
	}
	if m.CostPer1MIn != 0.30 || m.CostPer1MInCached != 0.03 || m.CostPer1MOut != 2.50 {
		t.Errorf("pricing = input %.2f cached %.2f output %.2f, want 0.30/0.03/2.50", m.CostPer1MIn, m.CostPer1MInCached, m.CostPer1MOut)
	}
}
