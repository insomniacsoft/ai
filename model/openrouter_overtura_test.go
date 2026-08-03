package model

import "testing"

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
