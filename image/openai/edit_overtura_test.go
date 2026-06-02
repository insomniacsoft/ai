package openai

import (
	"bytes"
	"testing"

	"github.com/joakimcarlsson/ai/image"
)

// pngBytes is a minimal PNG header (magic bytes) that http.DetectContentType
// classifies as image/png.
var pngBytes = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0}

// jpegBytes is a minimal JPEG header classified as image/jpeg.
var jpegBytes = []byte{0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0}

// TestNamedImageReader_DetectsContentType verifies the multipart fix: the reader
// reports a real image filename + content type from magic bytes (without it the
// SDK sends anonymous_file + octet-stream and gpt-image-1 rejects the upload).
func TestNamedImageReader_DetectsContentType(t *testing.T) {
	r := newNamedImageReader(pngBytes)
	if r.ContentType() != "image/png" {
		t.Errorf("ContentType = %q, want image/png", r.ContentType())
	}
	if r.Filename() != "image.png" {
		t.Errorf("Filename = %q, want image.png", r.Filename())
	}
}

// TestBuildImageEditUnion_Single verifies a single InputImage maps to OfFile.
func TestBuildImageEditUnion_Single(t *testing.T) {
	u, err := buildImageEditUnion(image.GenerationOptions{InputImage: pngBytes})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u.OfFile == nil {
		t.Error("expected OfFile to be set for single input image")
	}
	if u.OfFileArray != nil {
		t.Error("expected OfFileArray to be nil for single input image")
	}
}

// TestBuildImageEditUnion_Multi verifies multiple InputImages map to OfFileArray.
func TestBuildImageEditUnion_Multi(t *testing.T) {
	u, err := buildImageEditUnion(image.GenerationOptions{
		InputImages: [][]byte{pngBytes, jpegBytes},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(u.OfFileArray) != 2 {
		t.Errorf("OfFileArray len = %d, want 2", len(u.OfFileArray))
	}
}

// TestBuildImageEditUnion_RejectsBadMIME verifies a non-image blob fails the
// whole call rather than silently uploading garbage.
func TestBuildImageEditUnion_RejectsBadMIME(t *testing.T) {
	_, err := buildImageEditUnion(image.GenerationOptions{
		InputImage: bytes.Repeat([]byte("not an image "), 8),
	})
	if err == nil {
		t.Fatal("expected error for unsupported MIME, got nil")
	}
}

// TestExtForMIME verifies the extension mapping used to name multipart parts.
func TestExtForMIME(t *testing.T) {
	cases := map[string]string{
		"image/png":  ".png",
		"image/jpeg": ".jpg",
		"image/webp": ".webp",
		"image/gif":  ".gif",
		"text/plain": ".bin",
	}
	for mime, want := range cases {
		if got := extForMIME(mime); got != want {
			t.Errorf("extForMIME(%q) = %q, want %q", mime, got, want)
		}
	}
}

// TestIsGPTImage verifies gpt-image-* models are detected so response_format is
// not sent (the API rejects it for those models).
func TestIsGPTImage(t *testing.T) {
	for _, m := range []string{"gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini"} {
		if !isGPTImage(m) {
			t.Errorf("isGPTImage(%q) = false, want true", m)
		}
	}
	if isGPTImage("dall-e-3") {
		t.Error("isGPTImage(dall-e-3) = true, want false")
	}
}
