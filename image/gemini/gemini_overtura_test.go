package gemini

import (
	"testing"

	"github.com/joakimcarlsson/ai/image"
	"github.com/joakimcarlsson/ai/model"
)

// TestIsNativeModel verifies the native-vs-Imagen routing: Gemini-image models
// (which support conversational editing) route to the native backend; others
// fall through to Imagen.
func TestIsNativeModel(t *testing.T) {
	native := []model.ID{
		model.Gemini25FlashImage,
		model.Gemini3ProImage,
		model.Gemini31FlashImagePreview,
	}
	for _, id := range native {
		if !isNativeModel(model.ImageGenerationModel{ID: id}) {
			t.Errorf("isNativeModel(%s) = false, want true", id)
		}
	}
	if isNativeModel(model.ImageGenerationModel{ID: "imagen-4.0-generate-001"}) {
		t.Error("isNativeModel(imagen) = true, want false (Imagen routes to imagenClient)")
	}
}

// TestMapToAspectRatio verifies pixel sizes normalize to ratios and ratio-format
// strings pass through.
func TestMapToAspectRatio(t *testing.T) {
	cases := map[string]string{
		"1024x1024": "1:1",
		"512x512":   "1:1",
		"1024x1792": "9:16",
		"1536x1024": "16:9",
		"4:3":       "4:3", // passthrough
		"":          "1:1", // empty default
	}
	for in, want := range cases {
		if got := mapToAspectRatio(in); got != want {
			t.Errorf("mapToAspectRatio(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestGeminiImageConfig_NilWhenUnset verifies image_config is omitted entirely
// when neither Size nor ImageSize is set.
func TestGeminiImageConfig_NilWhenUnset(t *testing.T) {
	if cfg := geminiImageConfig(image.GenerationOptions{}); cfg != nil {
		t.Errorf("expected nil config when size unset, got %+v", cfg)
	}
	cfg := geminiImageConfig(image.GenerationOptions{Size: "16:9", ImageSize: "2K"})
	if cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if cfg.AspectRatio != "16:9" {
		t.Errorf("AspectRatio = %q, want 16:9", cfg.AspectRatio)
	}
	if cfg.ImageSize != "2K" {
		t.Errorf("ImageSize = %q, want 2K", cfg.ImageSize)
	}
}

// TestImagenClientRejectsEditing verifies the Imagen backend reports and enforces
// no editing support.
func TestImagenClientRejectsEditing(t *testing.T) {
	c := &imagenClient{}
	if c.SupportsEditing() {
		t.Error("imagenClient.SupportsEditing() = true, want false")
	}
	if _, err := c.EditImage(t.Context(), "x"); err != image.ErrEditNotSupported {
		t.Errorf("EditImage err = %v, want ErrEditNotSupported", err)
	}
}

// TestNativeClientSupportsEditing verifies the native backend reports editing
// support.
func TestNativeClientSupportsEditing(t *testing.T) {
	c := &nativeClient{}
	if !c.SupportsEditing() {
		t.Error("nativeClient.SupportsEditing() = false, want true")
	}
}
