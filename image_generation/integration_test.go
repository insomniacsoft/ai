package image_generation_test

import (
	"context"
	"os"
	"testing"
	"time"

	ig "github.com/joakimcarlsson/ai/image_generation"
	"github.com/joakimcarlsson/ai/model"
)

// testImage returns a minimal valid 1x1 red PNG (67 bytes).
func testImage() []byte {
	return []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
		0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB
		0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
		0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, // compressed
		0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, // pixel data
		0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
		0x44, 0xAE, 0x42, 0x60, 0x82,
	}
}

func skipIfNoKey(t *testing.T, envVar string) string {
	t.Helper()
	key := os.Getenv(envVar)
	if key == "" {
		t.Skipf("skipping: %s not set", envVar)
	}
	return key
}

// --- OpenAI Generate ---

func TestOpenAI_GenerateImage(t *testing.T) {
	apiKey := skipIfNoKey(t, "OPENAI_API_KEY")
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client, err := ig.NewImageGeneration(model.ProviderOpenAI,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.OpenAIImageGenerationModels[model.GPTImage1]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	resp, err := client.GenerateImage(ctx, "A small red circle on white background",
		ig.WithSize("1024x1024"),
		ig.WithQuality("low"),
	)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(resp.Images) == 0 {
		t.Fatal("no images returned")
	}
	if resp.Images[0].ImageURL == "" && resp.Images[0].ImageBase64 == "" {
		t.Fatal("image has no URL or base64 data")
	}
	t.Logf("OpenAI generate OK: model=%s, images=%d", resp.Model, len(resp.Images))
}

// --- OpenAI Edit ---

func TestOpenAI_EditImage_GPTImage1_NotDirectAPI(t *testing.T) {
	// GPT-Image-1 editing does NOT go through /v1/images/edits.
	// It goes through the Responses API (/v1/responses) where a text model
	// (gpt-4o, gpt-5.4-mini) uses the built-in image_generation tool with
	// input images from the conversation. That path is handled by the agent
	// pipeline (providers/openai_responses.go), not this image_generation package.
	//
	// The /v1/images/edits endpoint only supports direct editing for DALL-E 2
	// (multipart form-data). The openai-go SDK always sends multipart.
	apiKey := skipIfNoKey(t, "OPENAI_API_KEY")

	client, err := ig.NewImageGeneration(model.ProviderOpenAI,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.OpenAIImageGenerationModels[model.GPTImage1]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	_, err = client.EditImage(context.Background(), "Make the background blue",
		ig.WithInputImage(testImage()),
	)
	// Expect error — GPT-Image-1 editing requires Responses API, not Images API
	if err == nil {
		t.Fatal("expected error (GPT-Image-1 edit requires Responses API, not Images.Edit)")
	}
	t.Logf("GPT-Image-1 direct edit correctly fails: %v", err)
}

// --- OpenAI Edit without input image ---

func TestOpenAI_EditImage_NoInput(t *testing.T) {
	apiKey := skipIfNoKey(t, "OPENAI_API_KEY")

	client, err := ig.NewImageGeneration(model.ProviderOpenAI,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.OpenAIImageGenerationModels[model.GPTImage1]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	_, err = client.EditImage(context.Background(), "Make it blue")
	if err == nil {
		t.Fatal("expected error for edit without input image")
	}
	t.Logf("OpenAI edit no-input error (expected): %v", err)
}

// --- OpenAI SupportsEditing ---

func TestOpenAI_SupportsEditing(t *testing.T) {
	apiKey := skipIfNoKey(t, "OPENAI_API_KEY")

	client, err := ig.NewImageGeneration(model.ProviderOpenAI,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.OpenAIImageGenerationModels[model.GPTImage1]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	if !client.SupportsEditing() {
		t.Fatal("OpenAI GPT-Image-1 should support editing")
	}
}

// --- Gemini Native Generate ---

func TestGeminiNative_GenerateImage(t *testing.T) {
	apiKey := skipIfNoKey(t, "GEMINI_API_KEY")
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client, err := ig.NewImageGeneration(model.ProviderGemini,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.GeminiImageGenerationModels[model.Gemini25FlashImage]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	resp, err := client.GenerateImage(ctx, "A small green triangle on white background")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(resp.Images) == 0 {
		t.Fatal("no images returned")
	}
	if resp.Images[0].ImageBase64 == "" {
		t.Fatal("no base64 data in response")
	}
	t.Logf("Gemini native generate OK: model=%s, images=%d, b64len=%d",
		resp.Model, len(resp.Images), len(resp.Images[0].ImageBase64))
}

// --- Gemini Native Edit ---

func TestGeminiNative_EditImage(t *testing.T) {
	apiKey := skipIfNoKey(t, "GEMINI_API_KEY")

	client, err := ig.NewImageGeneration(model.ProviderGemini,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.GeminiImageGenerationModels[model.Gemini25FlashImage]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	if !client.SupportsEditing() {
		t.Fatal("Gemini native should support editing")
	}

	// Generate a real image first (the API rejects tiny synthetic PNGs)
	genCtx, genCancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer genCancel()

	genResp, err := client.GenerateImage(genCtx, "A plain red square on white background")
	if err != nil {
		t.Fatalf("generate source image: %v", err)
	}
	sourceImage, err := ig.DecodeBase64Image(genResp.Images[0].ImageBase64)
	if err != nil {
		t.Fatalf("decode source: %v", err)
	}
	t.Logf("Source image: %d bytes", len(sourceImage))

	// Now edit it
	editCtx, editCancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer editCancel()

	resp, err := client.EditImage(editCtx, "Change the background to blue",
		ig.WithInputImage(sourceImage),
	)
	if err != nil {
		t.Fatalf("edit: %v", err)
	}
	if len(resp.Images) == 0 {
		t.Fatal("no images returned")
	}
	t.Logf("Gemini native edit OK: model=%s, images=%d", resp.Model, len(resp.Images))
}

// --- Gemini Imagen Generate (regression) ---

func TestGeminiImagen_GenerateImage(t *testing.T) {
	apiKey := skipIfNoKey(t, "GEMINI_API_KEY")
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	client, err := ig.NewImageGeneration(model.ProviderGemini,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.GeminiImageGenerationModels[model.Imagen3]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	if client.SupportsEditing() {
		t.Fatal("Imagen should NOT support editing")
	}

	resp, err := client.GenerateImage(ctx, "A small blue square")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(resp.Images) == 0 {
		t.Fatal("no images returned")
	}
	t.Logf("Gemini Imagen generate OK: model=%s, images=%d", resp.Model, len(resp.Images))
}

// --- Gemini Imagen Edit should fail ---

func TestGeminiImagen_EditImage_NotSupported(t *testing.T) {
	apiKey := skipIfNoKey(t, "GEMINI_API_KEY")

	client, err := ig.NewImageGeneration(model.ProviderGemini,
		ig.WithAPIKey(apiKey),
		ig.WithModel(model.GeminiImageGenerationModels[model.Imagen3]),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	_, err = client.EditImage(context.Background(), "edit this",
		ig.WithInputImage(testImage()),
	)
	if err != ig.ErrEditNotSupported {
		t.Fatalf("expected ErrEditNotSupported, got: %v", err)
	}
	t.Log("Imagen edit correctly returns ErrEditNotSupported")
}

// --- OpenRouter Generate ---

func TestOpenRouter_GenerateImage(t *testing.T) {
	apiKey := skipIfNoKey(t, "OPENROUTER_API_KEY")
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Use a model that supports image generation via OpenRouter
	orModel := model.ImageGenerationModel{
		ID:             "openrouter-gpt5-image",
		Name:           "GPT-5 Image via OpenRouter",
		Provider:       model.ProviderOpenRouter,
		APIModel:       "openai/gpt-5-image",
		DefaultSize:    "1:1",
		DefaultQuality: "standard",
	}

	client, err := ig.NewImageGeneration(model.ProviderOpenRouter,
		ig.WithAPIKey(apiKey),
		ig.WithModel(orModel),
	)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}

	resp, err := client.GenerateImage(ctx, "A small red dot on white background")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(resp.Images) == 0 {
		t.Fatal("no images returned")
	}
	t.Logf("OpenRouter generate OK: model=%s, images=%d", resp.Model, len(resp.Images))
}
