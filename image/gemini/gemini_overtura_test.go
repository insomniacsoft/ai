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

// pngBytes is a minimal valid PNG (signature + IHDR) so http.DetectContentType
// classifies it as image/png.
var pngBytes = []byte{
	0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a,
	0x00, 0x00, 0x00, 0x0d, 'I', 'H', 'D', 'R',
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89,
}

// TestBuildEditParts_MultiReferenceNotDropped is the regression for the
// "input image required for editing" failure on every media_ids reference:
// a caller passing WithInputImages (the multi-reference path) populated only
// the plural field, which the old singular-only edit path ignored. Each
// reference must become its own inline image part, with the prompt last.
func TestBuildEditParts_MultiReferenceNotDropped(t *testing.T) {
	parts, err := buildEditParts("blend these", image.GenerationOptions{
		InputImages: [][]byte{pngBytes, pngBytes},
	})
	if err != nil {
		t.Fatalf("buildEditParts(InputImages) err = %v, want nil", err)
	}
	if len(parts) != 3 {
		t.Fatalf("parts = %d, want 3 (2 images + prompt)", len(parts))
	}
	for i := 0; i < 2; i++ {
		if parts[i].InlineData == nil || parts[i].InlineData.MIMEType != "image/png" {
			t.Errorf("parts[%d] not a png inline image: %+v", i, parts[i])
		}
	}
	if parts[2].Text != "blend these" {
		t.Errorf("last part text = %q, want the prompt", parts[2].Text)
	}
}

// TestBuildEditParts_SingleFallback verifies the legacy single InputImage path
// still produces one image part + the prompt.
func TestBuildEditParts_SingleFallback(t *testing.T) {
	parts, err := buildEditParts("edit it", image.GenerationOptions{InputImage: pngBytes})
	if err != nil {
		t.Fatalf("buildEditParts(InputImage) err = %v, want nil", err)
	}
	if len(parts) != 2 || parts[0].InlineData == nil || parts[1].Text != "edit it" {
		t.Fatalf("unexpected parts for single input: %+v", parts)
	}
}

// TestBuildEditParts_NoInputErrors verifies an edit with no source image is
// rejected (and the message names both option helpers, matching the other
// providers).
func TestBuildEditParts_NoInputErrors(t *testing.T) {
	_, err := buildEditParts("x", image.GenerationOptions{})
	if err == nil {
		t.Fatal("buildEditParts(empty) err = nil, want error")
	}
}

// TestBuildEditParts_UnsupportedTypeErrors verifies a non-image source is
// rejected rather than forwarded.
func TestBuildEditParts_UnsupportedTypeErrors(t *testing.T) {
	_, err := buildEditParts("x", image.GenerationOptions{
		InputImages: [][]byte{[]byte("not an image at all, just text")},
	})
	if err == nil {
		t.Fatal("buildEditParts(non-image) err = nil, want error")
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
