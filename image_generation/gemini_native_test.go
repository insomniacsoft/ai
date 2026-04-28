package image_generation

import "testing"

func TestGeminiImageConfig_PassesAspectRatioAndImageSize(t *testing.T) {
	cfg := geminiImageConfig(GenerationOptions{
		Size:      "1024x1792",
		ImageSize: "2K",
	})
	if cfg == nil {
		t.Fatal("expected image config")
	}
	if cfg.AspectRatio != "9:16" {
		t.Errorf("AspectRatio = %q, want 9:16", cfg.AspectRatio)
	}
	if cfg.ImageSize != "2K" {
		t.Errorf("ImageSize = %q, want 2K", cfg.ImageSize)
	}
}

func TestGeminiImageConfig_EmptyOptionsOmitConfig(t *testing.T) {
	if cfg := geminiImageConfig(GenerationOptions{}); cfg != nil {
		t.Fatalf("expected nil config for empty options, got %#v", cfg)
	}
}
