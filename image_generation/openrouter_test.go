package image_generation

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/joakimcarlsson/ai/model"
)

// roundTripperFunc lets a test stub HTTP responses without spinning up a server.
type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

// captureRequest stubs the OpenRouter HTTP layer and returns the JSON body
// the client tried to send. The returned response is a minimal valid
// chat-completions response carrying a single tiny base64 PNG.
func captureRequest(t *testing.T) (*http.Client, *map[string]any) {
	t.Helper()
	body := map[string]any{}
	captured := &body
	rt := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		_ = json.Unmarshal(raw, captured)

		respBody := `{"choices":[{"message":{"content":"","images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}}]}`
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(bytes.NewReader([]byte(respBody))),
			Header:     make(http.Header),
		}, nil
	})
	return &http.Client{Transport: rt}, captured
}

// captureRequestWithError stubs an OpenRouter HTTP error response. The
// response body is returned to the caller so wrapping/redaction can be
// asserted.
func captureRequestWithError(t *testing.T, status int, errBody string) *http.Client {
	t.Helper()
	rt := roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: status,
			Body:       io.NopCloser(strings.NewReader(errBody)),
			Header:     make(http.Header),
		}, nil
	})
	return &http.Client{Transport: rt}
}

// fluxFlexModel returns the FLUX.2 Flex model entry for tests.
func fluxFlexModel() model.ImageGenerationModel {
	return model.OpenRouterImageGenerationModels[model.OpenRouterFlux2Flex]
}

// riverflowFastModel returns the Riverflow V2 Fast model entry for tests.
func riverflowFastModel() model.ImageGenerationModel {
	return model.OpenRouterImageGenerationModels[model.OpenRouterRiverflowV2Fast]
}

// TestOpenRouter_Generate_ImageOnlyModalities asserts that callers can pass
// modalities: ["image"] to image-only models so OpenRouter does not reject
// the request for asking for text output it cannot produce.
func TestOpenRouter_Generate_ImageOnlyModalities(t *testing.T) {
	httpClient, captured := captureRequest(t)
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     "test-key",
		model:      fluxFlexModel(),
		httpClient: httpClient,
	})

	_, err := client.generate(context.Background(), "a cat on a desk",
		WithModalities("image"),
	)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	mods, ok := (*captured)["modalities"].([]any)
	if !ok {
		t.Fatalf("modalities missing or wrong type: %T", (*captured)["modalities"])
	}
	if len(mods) != 1 || mods[0] != "image" {
		t.Errorf("expected modalities=[\"image\"], got %v", mods)
	}
}

// TestOpenRouter_Generate_DefaultModalitiesFromModel asserts image-only
// OpenRouter model metadata drives the default request shape. Direct library
// callers should not need an app-side WithModalities workaround.
func TestOpenRouter_Generate_DefaultModalitiesFromModel(t *testing.T) {
	httpClient, captured := captureRequest(t)
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     "test-key",
		model:      fluxFlexModel(),
		httpClient: httpClient,
	})

	_, err := client.generate(context.Background(), "a cat on a desk")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	mods, ok := (*captured)["modalities"].([]any)
	if !ok {
		t.Fatalf("modalities missing")
	}
	if len(mods) != 1 || mods[0] != "image" {
		t.Errorf("expected default modalities=[image], got %v", mods)
	}
}

func TestOpenRouter_Generate_DefaultModalitiesFallback(t *testing.T) {
	httpClient, captured := captureRequest(t)
	m := fluxFlexModel()
	m.OutputModalities = nil
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     "test-key",
		model:      m,
		httpClient: httpClient,
	})

	_, err := client.generate(context.Background(), "a cat on a desk")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	mods, ok := (*captured)["modalities"].([]any)
	if !ok {
		t.Fatalf("modalities missing")
	}
	if len(mods) != 2 || mods[0] != "image" || mods[1] != "text" {
		t.Errorf("expected legacy fallback modalities=[image,text], got %v", mods)
	}
}

// TestOpenRouter_Generate_ImageSizeOverridesQuality asserts that the new
// WithImageSize option overrides the legacy quality→image_size mapping. This
// is the key contract for OpenRouter image-only models that price by tier
// (1K/2K/4K) and need an explicit knob separate from quality.
func TestOpenRouter_Generate_ImageSizeOverridesQuality(t *testing.T) {
	httpClient, captured := captureRequest(t)
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     "test-key",
		model:      riverflowFastModel(),
		httpClient: httpClient,
	})

	_, err := client.generate(context.Background(), "a cat",
		WithModalities("image"),
		WithQuality("low"), // would map to "0.5K" via legacy path
		WithImageSize("2K"),
	)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	cfg, ok := (*captured)["image_config"].(map[string]any)
	if !ok {
		t.Fatalf("image_config missing or wrong type")
	}
	if got := cfg["image_size"]; got != "2K" {
		t.Errorf("expected image_size=2K (override), got %v", got)
	}
}

// TestOpenRouter_Generate_ImageSizeFallbackFromQuality asserts the legacy
// quality→image_size mapping still applies when WithImageSize is not set,
// preserving backward compatibility for existing callers.
func TestOpenRouter_Generate_ImageSizeFallbackFromQuality(t *testing.T) {
	httpClient, captured := captureRequest(t)
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     "test-key",
		model:      riverflowFastModel(),
		httpClient: httpClient,
	})

	_, err := client.generate(context.Background(), "a cat",
		WithQuality("hd"),
	)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	cfg, ok := (*captured)["image_config"].(map[string]any)
	if !ok {
		t.Fatalf("image_config missing")
	}
	if got := cfg["image_size"]; got != "2K" {
		t.Errorf("expected image_size=2K (from quality=hd), got %v", got)
	}
}

// TestOpenRouter_Generate_ImageConfigTopLevel asserts image_config is sent
// as a top-level field in the request body, sibling to messages/modalities,
// not nested inside a message. Per OpenRouter image generation docs.
func TestOpenRouter_Generate_ImageConfigTopLevel(t *testing.T) {
	httpClient, captured := captureRequest(t)
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     "test-key",
		model:      fluxFlexModel(),
		httpClient: httpClient,
	})

	_, err := client.generate(context.Background(), "a cat",
		WithModalities("image"),
		WithSize("16:9"),
		WithImageSize("1K"),
	)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	if _, ok := (*captured)["image_config"]; !ok {
		t.Errorf("image_config must be at top level, request body: %v", *captured)
	}
	// image_config must NOT appear inside any message
	if msgs, ok := (*captured)["messages"].([]any); ok {
		for _, m := range msgs {
			if msg, ok := m.(map[string]any); ok {
				if _, leaked := msg["image_config"]; leaked {
					t.Errorf("image_config leaked into a message: %v", msg)
				}
			}
		}
	}
}

// TestOpenRouter_Generate_AspectRatioFromSize asserts the size→aspect_ratio
// mapping handles both ratio strings (passthrough) and pixel strings (mapped).
func TestOpenRouter_Generate_AspectRatioFromSize(t *testing.T) {
	cases := []struct {
		size, want string
	}{
		{"16:9", "16:9"},
		{"9:16", "9:16"},
		{"4:3", "4:3"},
		{"1024x1024", "1:1"},
		{"1024x1792", "9:16"},
		{"1792x1024", "16:9"},
		{"", "1:1"},
	}
	for _, c := range cases {
		httpClient, captured := captureRequest(t)
		client := newOpenRouterClient(imageGenerationClientOptions{
			apiKey:     "test-key",
			model:      fluxFlexModel(),
			httpClient: httpClient,
		})
		_, err := client.generate(context.Background(), "a cat",
			WithModalities("image"),
			WithSize(c.size),
		)
		if err != nil {
			t.Fatalf("generate(size=%q): %v", c.size, err)
		}
		cfg := (*captured)["image_config"].(map[string]any)
		if got := cfg["aspect_ratio"]; got != c.want {
			t.Errorf("size=%q: aspect_ratio=%v, want %v", c.size, got, c.want)
		}
	}
}

// TestOpenRouter_Edit_RespectsModalities asserts the edit path also reads
// modalities from options instead of hardcoding image+text. This matches
// generate() behavior so image-only multi-turn editing works.
func TestOpenRouter_Edit_RespectsModalities(t *testing.T) {
	httpClient, captured := captureRequest(t)
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     "test-key",
		model:      fluxFlexModel(),
		httpClient: httpClient,
	})

	_, err := client.edit(context.Background(), "remove the background",
		WithInputImage(testImagePNG()),
		WithModalities("image"),
	)
	if err != nil {
		t.Fatalf("edit: %v", err)
	}

	mods, ok := (*captured)["modalities"].([]any)
	if !ok {
		t.Fatalf("modalities missing in edit body")
	}
	if len(mods) != 1 || mods[0] != "image" {
		t.Errorf("edit: expected modalities=[image], got %v", mods)
	}
}

// TestOpenRouter_Generate_ErrorDoesNotLeakAPIKey asserts that 4xx errors
// from OpenRouter are wrapped without including the bearer token.
func TestOpenRouter_Generate_ErrorDoesNotLeakAPIKey(t *testing.T) {
	apiKey := "sk-or-secret-key-do-not-leak-12345"
	httpClient := captureRequestWithError(t, 400, `{"error":{"message":"image-only model cannot return text"}}`)
	client := newOpenRouterClient(imageGenerationClientOptions{
		apiKey:     apiKey,
		model:      fluxFlexModel(),
		httpClient: httpClient,
	})

	_, err := client.generate(context.Background(), "a cat",
		WithModalities("image"),
	)
	if err == nil {
		t.Fatal("expected error from 400 response")
	}
	if strings.Contains(err.Error(), apiKey) {
		t.Errorf("error message contains API key: %v", err)
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("error should reference status code 400: %v", err)
	}
}

// TestOpenRouterImageRegistry_AllModelsHaveAPIModelAndPricing asserts every
// restored OpenRouter image entry has the metadata Overtura's app catalog
// will read at registration time.
func TestOpenRouterImageRegistry_AllModelsHaveAPIModelAndPricing(t *testing.T) {
	for id, m := range model.OpenRouterImageGenerationModels {
		if m.Provider != model.ProviderOpenRouter {
			t.Errorf("%s: provider=%s, want openrouter", id, m.Provider)
		}
		if m.APIModel == "" {
			t.Errorf("%s: empty APIModel", id)
		}
		if !strings.Contains(m.APIModel, "/") {
			t.Errorf("%s: APIModel %q missing provider prefix (e.g., black-forest-labs/...)", id, m.APIModel)
		}
		if len(m.SupportedSizes) == 0 {
			t.Errorf("%s: empty SupportedSizes", id)
		}
		if m.DefaultSize == "" {
			t.Errorf("%s: empty DefaultSize", id)
		}
		if len(m.SupportedQualities) == 0 {
			t.Errorf("%s: empty SupportedQualities (image_size tiers)", id)
		}
		if m.DefaultQuality == "" {
			t.Errorf("%s: empty DefaultQuality", id)
		}
		if len(m.Pricing) == 0 {
			t.Errorf("%s: empty Pricing — would silently under-bill", id)
		}
		if len(m.OutputModalities) != 1 || m.OutputModalities[0] != "image" {
			t.Errorf("%s: OutputModalities=%v, want [image]", id, m.OutputModalities)
		}
	}
}

// TestOpenRouterImageRegistry_VerifiedAPIModelIDs asserts the provider API
// IDs match the live OpenRouter catalog as verified during planning. Catches
// accidental typos and silent renames.
func TestOpenRouterImageRegistry_VerifiedAPIModelIDs(t *testing.T) {
	want := map[model.ID]string{
		model.OpenRouterFlux2Max:        "black-forest-labs/flux.2-max",
		model.OpenRouterFlux2Pro:        "black-forest-labs/flux.2-pro",
		model.OpenRouterFlux2Klein:      "black-forest-labs/flux.2-klein-4b",
		model.OpenRouterFlux2Flex:       "black-forest-labs/flux.2-flex",
		model.OpenRouterRiverflowV2Pro:  "sourceful/riverflow-v2-pro",
		model.OpenRouterRiverflowV2Fast: "sourceful/riverflow-v2-fast",
	}
	for id, apiModel := range want {
		entry, ok := model.OpenRouterImageGenerationModels[id]
		if !ok {
			t.Fatalf("missing entry: %s", id)
		}
		if entry.APIModel != apiModel {
			t.Errorf("%s: APIModel=%q, want %q", id, entry.APIModel, apiModel)
		}
	}
}

// testImagePNG returns a minimal 1x1 PNG sufficient for input MIME validation.
func testImagePNG() []byte {
	return []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
		0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
		0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
		0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
		0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
		0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
		0x44, 0xAE, 0x42, 0x60, 0x82,
	}
}
