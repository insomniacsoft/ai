// Package gemini provides a Google Gemini implementation of the optioned
// [image.ImageGeneration] interface. It exposes two backends behind a single
// NewGeneration constructor:
//
//   - The Imagen path ([imagenClient], via Models.GenerateImages) for the
//     Imagen-family models — generation only.
//   - The native path ([nativeClient], via Models.GenerateContent with IMAGE
//     response modality) for the Gemini-image models (gemini-2.5-flash-image,
//     gemini-3-pro-image, gemini-3.1-flash-image) — generation AND conversational
//     editing via inline image data.
//
// This is the overtura optioned surface re-homed from the monolith fork. Like
// image/openai it implements [image.ImageGeneration] (per-call options + edit),
// not upstream's prompt-only [image.Generation].
package gemini

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/joakimcarlsson/ai/image"
	"github.com/joakimcarlsson/ai/model"
	"google.golang.org/genai"
)

// Options configures the Gemini image generation client.
type Options struct {
	apiKey  string
	model   model.ImageGenerationModel
	timeout *time.Duration
	backend genai.Backend
}

// Option configures Options.
type Option func(*Options)

// WithAPIKey sets the API key used to authenticate with the Gemini API.
func WithAPIKey(apiKey string) Option {
	return func(o *Options) { o.apiKey = apiKey }
}

// WithModel selects the image generation model.
func WithModel(m model.ImageGenerationModel) Option {
	return func(o *Options) { o.model = m }
}

// WithTimeout sets the maximum duration to wait for a single request.
func WithTimeout(d time.Duration) Option {
	return func(o *Options) { o.timeout = &d }
}

// WithBackend selects the Gemini backend (GeminiAPI or VertexAI). Defaults to
// genai.BackendGeminiAPI.
func WithBackend(backend genai.Backend) Option {
	return func(o *Options) { o.backend = backend }
}

// NewGeneration constructs a Gemini image generation client. Native-image models
// get the conversational-edit-capable backend; the rest get the Imagen backend.
// The returned [image.ImageGeneration] is wrapped with [image.WithEditingTracing].
func NewGeneration(opts ...Option) image.ImageGeneration {
	options := Options{backend: genai.BackendGeminiAPI}
	for _, o := range opts {
		o(&options)
	}

	// A nil client is tolerated here (matches the monolith): construction never
	// returned an error, and a failed client surfaces as a request-time error.
	client, _ := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  options.apiKey,
		Backend: options.backend,
	})

	if isNativeModel(options.model) {
		return image.WithEditingTracing(&nativeClient{client: client, options: options}, image.TracingAttrs{})
	}
	return image.WithEditingTracing(&imagenClient{client: client, options: options}, image.TracingAttrs{})
}

// nativeClient implements generation and editing via GenerateContent with the
// IMAGE response modality.
type nativeClient struct {
	client  *genai.Client
	options Options
}

// Model returns the configured image generation model.
func (g *nativeClient) Model() model.ImageGenerationModel { return g.options.model }

// SupportsEditing reports that the native Gemini image client supports editing.
func (g *nativeClient) SupportsEditing() bool { return true }

func (g *nativeClient) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if g.options.timeout != nil {
		return context.WithTimeout(ctx, *g.options.timeout)
	}
	return ctx, func() {}
}

// GenerateImage creates an image from a text prompt.
func (g *nativeClient) GenerateImage(
	ctx context.Context,
	prompt string,
	options ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	opts := image.ApplyGenerationOptionsWithModelDefaults(g.options.model, "b64_json", options...)

	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"IMAGE", "TEXT"},
		ImageConfig:        geminiImageConfig(opts),
	}

	ctx, cancel := g.withTimeout(ctx)
	defer cancel()

	resp, err := g.client.Models.GenerateContent(ctx, g.options.model.APIModel, genai.Text(prompt), config)
	if err != nil {
		return nil, fmt.Errorf("failed to generate image: %w", err)
	}
	return g.mapResponse(resp)
}

// EditImage edits an existing image. The source image is provided through
// [image.WithInputImage].
func (g *nativeClient) EditImage(
	ctx context.Context,
	prompt string,
	options ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	opts := image.ApplyGenerationOptionsWithModelDefaults(g.options.model, "b64_json", options...)
	if len(opts.InputImage) == 0 {
		return nil, fmt.Errorf("input image required for editing: use WithInputImage(data)")
	}

	mimeType := detectImageMIME(opts.InputImage)
	if !isAllowedImageMIME(mimeType) {
		return nil, fmt.Errorf("unsupported image type: %s", mimeType)
	}

	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"IMAGE", "TEXT"},
		ImageConfig:        geminiImageConfig(opts),
	}
	contents := []*genai.Content{{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{InlineData: &genai.Blob{MIMEType: mimeType, Data: opts.InputImage}},
			{Text: prompt},
		},
	}}

	ctx, cancel := g.withTimeout(ctx)
	defer cancel()

	resp, err := g.client.Models.GenerateContent(ctx, g.options.model.APIModel, contents, config)
	if err != nil {
		return nil, fmt.Errorf("failed to edit image: %w", err)
	}
	return g.mapResponse(resp)
}

// GenerateImageStreaming is not supported by the native Gemini image backend.
func (g *nativeClient) GenerateImageStreaming(
	context.Context, string, image.StreamCallback, ...image.GenerationOption,
) error {
	return image.ErrStreamingNotSupported
}

// EditImageStreaming is not supported by the native Gemini image backend.
func (g *nativeClient) EditImageStreaming(
	context.Context, string, image.StreamCallback, ...image.GenerationOption,
) error {
	return image.ErrStreamingNotSupported
}

func (g *nativeClient) mapResponse(resp *genai.GenerateContentResponse) (*image.GenerationResponse, error) {
	if err := checkFinishReason(resp); err != nil {
		return nil, err
	}

	var results []image.GenerationResult
	for _, candidate := range resp.Candidates {
		if candidate.Content == nil {
			continue
		}
		for _, part := range candidate.Content.Parts {
			if part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "image/") {
				results = append(results, image.GenerationResult{
					ImageBase64: base64.StdEncoding.EncodeToString(part.InlineData.Data),
				})
			}
		}
	}
	if len(results) == 0 {
		return nil, errors.New("no image generated in response")
	}
	return &image.GenerationResponse{Images: results, Model: g.options.model.APIModel}, nil
}

func isNativeModel(m model.ImageGenerationModel) bool {
	switch m.ID {
	case model.Gemini25FlashImage, model.Gemini3ProImage, model.Gemini31FlashImagePreview:
		return true
	default:
		return false
	}
}

func detectImageMIME(data []byte) string {
	if len(data) == 0 {
		return "application/octet-stream"
	}
	return http.DetectContentType(data)
}

func isAllowedImageMIME(mime string) bool {
	switch mime {
	case "image/png", "image/jpeg", "image/webp", "image/gif":
		return true
	default:
		return false
	}
}
