package image

import (
	"testing"

	"github.com/joakimcarlsson/ai/model"
)

// TestApplyGenerationOptionsWithModelDefaults_SeedsDefaults verifies an unset
// per-call surface inherits the model's Size/Quality and the provider default
// response format, with N defaulting to 1.
func TestApplyGenerationOptionsWithModelDefaults_SeedsDefaults(t *testing.T) {
	m := model.ImageGenerationModel{DefaultSize: "1024x1024", DefaultQuality: "high"}

	got := ApplyGenerationOptionsWithModelDefaults(m, "b64_json")

	if got.Size != "1024x1024" {
		t.Errorf("Size = %q, want 1024x1024", got.Size)
	}
	if got.Quality != "high" {
		t.Errorf("Quality = %q, want high", got.Quality)
	}
	if got.ResponseFormat != "b64_json" {
		t.Errorf("ResponseFormat = %q, want b64_json", got.ResponseFormat)
	}
	if got.N != 1 {
		t.Errorf("N = %d, want 1", got.N)
	}
}

// TestApplyGenerationOptionsWithModelDefaults_OverlaysOptions verifies caller
// options override the seeded model defaults.
func TestApplyGenerationOptionsWithModelDefaults_OverlaysOptions(t *testing.T) {
	m := model.ImageGenerationModel{DefaultSize: "1024x1024", DefaultQuality: "high"}

	got := ApplyGenerationOptionsWithModelDefaults(m, "url",
		WithSize("512x512"),
		WithQuality("low"),
		WithN(3),
	)

	if got.Size != "512x512" {
		t.Errorf("Size = %q, want 512x512 (caller override)", got.Size)
	}
	if got.Quality != "low" {
		t.Errorf("Quality = %q, want low (caller override)", got.Quality)
	}
	if got.N != 3 {
		t.Errorf("N = %d, want 3 (caller override)", got.N)
	}
}

// TestWithInputImages_SkipsEmpty verifies empty reference blobs are filtered out.
func TestWithInputImages_SkipsEmpty(t *testing.T) {
	got := ApplyGenerationOptions(WithInputImages([][]byte{
		[]byte("a"), nil, {}, []byte("b"),
	}))

	if len(got.InputImages) != 2 {
		t.Fatalf("InputImages len = %d, want 2 (empties filtered)", len(got.InputImages))
	}
}
