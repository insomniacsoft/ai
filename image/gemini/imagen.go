package gemini

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/joakimcarlsson/ai/image"
	"github.com/joakimcarlsson/ai/model"
	"google.golang.org/genai"
)

// imagenClient implements generation via Models.GenerateImages (the Imagen
// family). It is generation-only: editing and streaming are unsupported.
type imagenClient struct {
	client  *genai.Client
	options Options
}

// Model returns the configured image generation model.
func (g *imagenClient) Model() model.ImageGenerationModel { return g.options.model }

// SupportsEditing reports that the Imagen client does not support editing.
func (g *imagenClient) SupportsEditing() bool { return false }

func (g *imagenClient) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if g.options.timeout != nil {
		return context.WithTimeout(ctx, *g.options.timeout)
	}
	return ctx, func() {}
}

// GenerateImage creates one or more images from a text prompt via Imagen.
func (g *imagenClient) GenerateImage(
	ctx context.Context,
	prompt string,
	options ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	opts := image.ApplyGenerationOptionsWithModelDefaults(g.options.model, "b64_json", options...)

	config := &genai.GenerateImagesConfig{NumberOfImages: int32(opts.N)}
	if opts.Size != "" && opts.Size != "1:1" {
		config.AspectRatio = opts.Size
	}

	ctx, cancel := g.withTimeout(ctx)
	defer cancel()

	response, err := g.client.Models.GenerateImages(ctx, g.options.model.APIModel, prompt, config)
	if err != nil {
		return nil, fmt.Errorf("failed to generate image: %w", err)
	}

	results := make([]image.GenerationResult, 0, len(response.GeneratedImages))
	for _, img := range response.GeneratedImages {
		results = append(results, image.GenerationResult{
			ImageBase64: base64.StdEncoding.EncodeToString(img.Image.ImageBytes),
		})
	}
	return &image.GenerationResponse{Images: results, Model: g.options.model.APIModel}, nil
}

// GenerateImageStreaming is not supported by the Imagen backend.
func (g *imagenClient) GenerateImageStreaming(
	context.Context, string, image.StreamCallback, ...image.GenerationOption,
) error {
	return image.ErrStreamingNotSupported
}

// EditImage is not supported by the Imagen backend.
func (g *imagenClient) EditImage(
	context.Context, string, ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	return nil, image.ErrEditNotSupported
}

// EditImageStreaming is not supported by the Imagen backend.
func (g *imagenClient) EditImageStreaming(
	context.Context, string, image.StreamCallback, ...image.GenerationOption,
) error {
	return image.ErrEditNotSupported
}
