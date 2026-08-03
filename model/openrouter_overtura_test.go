package model

import (
	"slices"
	"testing"
)

func TestOverturaOpenRouterImageModelsRemainAvailable(t *testing.T) {
	ids := []ID{
		OpenRouterFlux2Max,
		OpenRouterFlux2Pro,
		OpenRouterFlux2Klein,
		OpenRouterFlux2Flex,
		OpenRouterRiverflowV2Pro,
		OpenRouterRiverflowV2Fast,
	}
	for _, id := range ids {
		m, ok := OpenRouterImageGenerationModels[id]
		if !ok {
			t.Errorf("OpenRouterImageGenerationModels[%q] is missing", id)
			continue
		}
		if len(m.OutputModalities) != 1 || m.OutputModalities[0] != "image" {
			t.Errorf("OpenRouterImageGenerationModels[%q].OutputModalities = %v, want [image]", id, m.OutputModalities)
		}
	}
}

func TestOverturaOpenRouterTierMetadata(t *testing.T) {
	flex := OpenRouterImageGenerationModels[OpenRouterFlux2Flex]
	if !slices.Contains(flex.SupportedAspectRatios, "4:5") {
		t.Fatalf("FLUX.2 Flex ratios = %v, want 4:5", flex.SupportedAspectRatios)
	}
	if got := flex.Pricing["16:9"]["4K"]; got != 0.24 {
		t.Fatalf("FLUX.2 Flex 16:9/4K price = %v, want 0.24", got)
	}
	if flex.DefaultSize != "1K" || flex.DefaultQuality != "1K" {
		t.Fatalf("FLUX.2 Flex defaults = %q/%q, want 1K/1K", flex.DefaultSize, flex.DefaultQuality)
	}

	fast := OpenRouterImageGenerationModels[OpenRouterRiverflowV2Fast]
	if got := fast.Pricing["1:1"]["1K"]; got != 0.0049 {
		t.Fatalf("Riverflow V2 Fast 1:1/1K price = %v, want 0.0049", got)
	}
}
