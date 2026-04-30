package image_generation

import (
	"reflect"
	"testing"
)

// TestImageEditOptionsRoundTrip verifies the v0.18.5-overtura.5 additions
// land in GenerationOptions.
func TestImageEditOptionsRoundTrip(t *testing.T) {
	opts := GenerationOptions{}
	WithOutputFormat("webp")(&opts)
	WithOutputCompression(85)(&opts)
	WithInputImages([][]byte{
		[]byte("ref-1"),
		[]byte("ref-2"),
		[]byte("ref-3"),
	})(&opts)

	if opts.OutputFormat != "webp" {
		t.Errorf("OutputFormat = %q, want webp", opts.OutputFormat)
	}
	if opts.OutputCompression != 85 {
		t.Errorf("OutputCompression = %d, want 85", opts.OutputCompression)
	}
	if len(opts.InputImages) != 3 {
		t.Fatalf("InputImages length = %d, want 3", len(opts.InputImages))
	}
	wantBytes := [][]byte{
		[]byte("ref-1"),
		[]byte("ref-2"),
		[]byte("ref-3"),
	}
	if !reflect.DeepEqual(opts.InputImages, wantBytes) {
		t.Errorf("InputImages = %v, want %v", opts.InputImages, wantBytes)
	}
}

// TestImageEditOptionsFiltersEmptyReferences ensures nil/empty entries are dropped.
func TestImageEditOptionsFiltersEmptyReferences(t *testing.T) {
	opts := GenerationOptions{}
	WithInputImages([][]byte{
		[]byte("a"),
		nil,
		{},
		[]byte("b"),
	})(&opts)
	if len(opts.InputImages) != 2 {
		t.Errorf("expected 2 non-empty refs, got %d: %v", len(opts.InputImages), opts.InputImages)
	}
}
