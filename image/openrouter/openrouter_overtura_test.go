package openrouter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/joakimcarlsson/ai/image"
	"github.com/joakimcarlsson/ai/model"
)

// roundTripFunc captures the outgoing request and returns a canned response.
type roundTripFunc struct {
	captured *http.Request
	body     []byte
	respJSON string
	status   int
}

func (f *roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	f.captured = req
	if req.Body != nil {
		f.body, _ = io.ReadAll(req.Body)
	}
	status := f.status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(strings.NewReader(f.respJSON)),
		Header:     make(http.Header),
	}, nil
}

func newTestClient(rt *roundTripFunc, m model.ImageGenerationModel) image.ImageGeneration {
	return NewGeneration(
		WithAPIKey("sk-or-test"),
		WithModel(m),
		WithHTTPClient(&http.Client{Transport: rt}),
	)
}

var pngBytes = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0}

const okImageResp = `{"choices":[{"message":{"content":"","images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}}]}`

// TestGenerateImage_RequestShape verifies the request carries model, the prompt
// message, resolved modalities, and image_config (aspect_ratio + image_size).
func TestGenerateImage_RequestShape(t *testing.T) {
	rt := &roundTripFunc{respJSON: okImageResp}
	m := model.ImageGenerationModel{APIModel: "black-forest-labs/flux.2-pro", OutputModalities: []string{"image"}}
	c := newTestClient(rt, m)

	resp, err := c.GenerateImage(context.Background(), "a cat", image.WithImageSize("2K"), image.WithSize("16:9"))
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if len(resp.Images) != 1 || resp.Images[0].ImageBase64 != "AAAA" {
		t.Fatalf("expected one image AAAA, got %+v", resp.Images)
	}

	var body map[string]any
	if err := json.Unmarshal(rt.body, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if body["model"] != "black-forest-labs/flux.2-pro" {
		t.Errorf("model = %v", body["model"])
	}
	mods, _ := body["modalities"].([]any)
	if len(mods) != 1 || mods[0] != "image" {
		t.Errorf("modalities = %v, want [image] (model OutputModalities)", body["modalities"])
	}
	cfg, _ := body["image_config"].(map[string]any)
	if cfg["aspect_ratio"] != "16:9" || cfg["image_size"] != "2K" {
		t.Errorf("image_config = %v, want aspect_ratio=16:9 image_size=2K", cfg)
	}
	if got := rt.captured.Header.Get("Authorization"); got != "Bearer sk-or-test" {
		t.Errorf("Authorization = %q", got)
	}
}

// TestEditImage_RequestShape verifies edit sends a multi-part content array with
// a data-URI image_url block, and that caller modalities override the model.
func TestEditImage_RequestShape(t *testing.T) {
	rt := &roundTripFunc{respJSON: okImageResp}
	m := model.ImageGenerationModel{APIModel: "sourceful/riverflow-v2-pro", OutputModalities: []string{"image"}}
	c := newTestClient(rt, m)

	_, err := c.EditImage(context.Background(), "make it blue",
		image.WithInputImage(pngBytes),
		image.WithModalities("image", "text"),
	)
	if err != nil {
		t.Fatalf("EditImage: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(rt.body, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	mods, _ := body["modalities"].([]any)
	if len(mods) != 2 {
		t.Errorf("modalities = %v, want caller override [image text]", body["modalities"])
	}
	msgs, _ := body["messages"].([]any)
	msg0, _ := msgs[0].(map[string]any)
	content, _ := msg0["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content blocks = %d, want 2 (text + image_url)", len(content))
	}
	imgBlock, _ := content[1].(map[string]any)
	if imgBlock["type"] != "image_url" {
		t.Errorf("second block type = %v, want image_url", imgBlock["type"])
	}
	iu, _ := imgBlock["image_url"].(map[string]any)
	if url, _ := iu["url"].(string); !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Errorf("image url = %q, want data:image/png;base64, prefix", url)
	}
}

// TestEditImage_RequiresInputImage verifies an edit with no source image errors.
func TestEditImage_RequiresInputImage(t *testing.T) {
	rt := &roundTripFunc{respJSON: okImageResp}
	c := newTestClient(rt, model.ImageGenerationModel{APIModel: "x"})
	if _, err := c.EditImage(context.Background(), "x"); err == nil {
		t.Fatal("expected error when no input image provided")
	}
}

// TestResolveModalities verifies precedence: caller > model > default.
func TestResolveModalities(t *testing.T) {
	if got := resolveModalities([]string{"image"}, []string{"image", "text"}); len(got) != 2 {
		t.Errorf("caller should win, got %v", got)
	}
	if got := resolveModalities([]string{"image"}, nil); len(got) != 1 || got[0] != "image" {
		t.Errorf("model should win when caller silent, got %v", got)
	}
	if got := resolveModalities(nil, nil); len(got) != 2 {
		t.Errorf("default [image text] when both silent, got %v", got)
	}
}

// TestResolveImageSize verifies caller ImageSize wins over quality derivation.
func TestResolveImageSize(t *testing.T) {
	if got := resolveImageSize("4K", "low"); got != "4K" {
		t.Errorf("caller size should win, got %q", got)
	}
	if got := resolveImageSize("", "high"); got != "2K" {
		t.Errorf("high quality -> 2K, got %q", got)
	}
	if got := resolveImageSize("", ""); got != "1K" {
		t.Errorf("empty quality -> 1K, got %q", got)
	}
}

// TestExtractBase64FromDataURI verifies stripping of the data-URI prefix.
func TestExtractBase64FromDataURI(t *testing.T) {
	if got := extractBase64FromDataURI("data:image/png;base64,XYZ"); got != "XYZ" {
		t.Errorf("got %q, want XYZ", got)
	}
	if got := extractBase64FromDataURI("rawbase64"); got != "rawbase64" {
		t.Errorf("got %q, want passthrough", got)
	}
}

// TestStreamingUnsupported verifies both streaming entry points return the
// streaming-not-supported sentinel.
func TestStreamingUnsupported(t *testing.T) {
	c := &Client{}
	if err := c.GenerateImageStreaming(context.Background(), "x", nil); err != image.ErrStreamingNotSupported {
		t.Errorf("generate streaming err = %v", err)
	}
	if err := c.EditImageStreaming(context.Background(), "x", nil); err != image.ErrStreamingNotSupported {
		t.Errorf("edit streaming err = %v", err)
	}
	if !c.SupportsEditing() {
		t.Error("SupportsEditing() = false, want true")
	}
}
