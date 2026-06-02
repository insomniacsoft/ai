// Package openai provides an OpenAI implementation of the optioned
// [image.ImageGeneration] interface, supporting per-call generation options and
// image editing (gpt-image-1 / gpt-image-1.5 via /v1/images/edits).
//
// Unlike upstream's prompt-only image/openai (which implements the construct-once
// [image.Generation] interface), this client takes all image knobs per call via
// [image.GenerationOption] and supports EditImage. It is the overtura surface
// re-homed from the monolith fork; the two interfaces cannot share one type
// because GenerateImage has different signatures, so this module implements
// [image.ImageGeneration] exclusively. image/xai remains on [image.Generation].
package openai

import (
	"context"
	"fmt"
	"time"

	"github.com/joakimcarlsson/ai/image"
	"github.com/joakimcarlsson/ai/model"
	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// StreamingOptions contains OpenAI-specific options for streaming image generation.
type StreamingOptions struct {
	// PartialImages specifies the number of partial images to receive during streaming (0-3).
	// If set to 0, only the final image will be received. You may receive fewer
	// partial images than requested if the full image is generated quickly.
	PartialImages int
}

// Options configures the OpenAI image generation client.
type Options struct {
	apiKey           string
	model            model.ImageGenerationModel
	timeout          *time.Duration
	baseURL          string
	extraHeaders     map[string]string
	streamingOptions StreamingOptions
}

// Option configures Options.
type Option func(*Options)

// WithAPIKey sets the API key used to authenticate with OpenAI.
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

// WithBaseURL points the client at a custom OpenAI-compatible endpoint
// (used for xAI's OpenAI-compatible image API).
func WithBaseURL(baseURL string) Option {
	return func(o *Options) { o.baseURL = baseURL }
}

// WithExtraHeaders adds custom HTTP headers to every request.
func WithExtraHeaders(headers map[string]string) Option {
	return func(o *Options) { o.extraHeaders = headers }
}

// WithStreamingOptions configures streaming behaviour (partial-image count).
func WithStreamingOptions(opts StreamingOptions) Option {
	return func(o *Options) { o.streamingOptions = opts }
}

// Client implements [image.ImageGeneration] against the OpenAI image API.
type Client struct {
	options Options
	client  openaisdk.Client
}

// NewGeneration constructs an OpenAI image generation client. The returned
// [image.ImageGeneration] is wrapped with [image.WithEditingTracing], so callers
// always get tracing spans and metrics.
func NewGeneration(opts ...Option) image.ImageGeneration {
	options := Options{
		extraHeaders:     map[string]string{},
		streamingOptions: StreamingOptions{PartialImages: 2},
	}
	for _, o := range opts {
		o(&options)
	}

	clientOpts := []option.RequestOption{option.WithAPIKey(options.apiKey)}
	if options.baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(options.baseURL))
	}
	for k, v := range options.extraHeaders {
		clientOpts = append(clientOpts, option.WithHeader(k, v))
	}

	return image.WithEditingTracing(&Client{
		options: options,
		client:  openaisdk.NewClient(clientOpts...),
	}, image.TracingAttrs{})
}

// Model returns the configured image generation model.
func (c *Client) Model() model.ImageGenerationModel { return c.options.model }

// withTimeout derives a context bounded by the configured per-request timeout.
// When no timeout is set it returns ctx and a no-op cancel.
func (c *Client) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.options.timeout != nil {
		return context.WithTimeout(ctx, *c.options.timeout)
	}
	return ctx, func() {}
}

// SupportsEditing reports that the OpenAI image client supports editing.
func (c *Client) SupportsEditing() bool { return true }

// GenerateImage performs a non-streaming image generation request.
func (c *Client) GenerateImage(
	ctx context.Context,
	prompt string,
	options ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	genOpts := image.ApplyGenerationOptionsWithModelDefaults(c.options.model, "url", options...)

	params := openaisdk.ImageGenerateParams{
		Prompt: prompt,
		Model:  openaisdk.ImageModel(c.options.model.APIModel),
		N:      openaisdk.Int(int64(genOpts.N)),
	}

	if genOpts.ResponseFormat != "" && !isGPTImage(c.options.model.APIModel) {
		params.ResponseFormat = openaisdk.ImageGenerateParamsResponseFormat(genOpts.ResponseFormat)
	}
	if genOpts.Size != "" && len(c.options.model.SupportedSizes) > 0 {
		params.Size = openaisdk.ImageGenerateParamsSize(genOpts.Size)
	}
	if genOpts.Quality != "" && genOpts.Quality != "default" && len(c.options.model.SupportedQualities) > 1 {
		params.Quality = openaisdk.ImageGenerateParamsQuality(genOpts.Quality)
	}

	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	response, err := c.client.Images.Generate(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to generate image: %w", err)
	}
	return mapResponse(response, c.options.model), nil
}

// GenerateImageStreaming performs a streaming image generation request. Returns
// [image.ErrStreamingNotSupported] if the configured model does not stream.
func (c *Client) GenerateImageStreaming(
	ctx context.Context,
	prompt string,
	callback image.StreamCallback,
	options ...image.GenerationOption,
) error {
	if !c.options.model.SupportsStreaming {
		return image.ErrStreamingNotSupported
	}

	genOpts := image.ApplyGenerationOptionsWithModelDefaults(c.options.model, "", options...)

	params := openaisdk.ImageGenerateParams{
		Prompt:        prompt,
		Model:         openaisdk.ImageModel(c.options.model.APIModel),
		N:             openaisdk.Int(int64(genOpts.N)),
		PartialImages: openaisdk.Int(int64(c.options.streamingOptions.PartialImages)),
	}
	if genOpts.Size != "" && len(c.options.model.SupportedSizes) > 0 {
		params.Size = openaisdk.ImageGenerateParamsSize(genOpts.Size)
	}
	if genOpts.Quality != "" && genOpts.Quality != "default" && len(c.options.model.SupportedQualities) > 1 {
		params.Quality = openaisdk.ImageGenerateParamsQuality(genOpts.Quality)
	}

	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	stream := c.client.Images.GenerateStreaming(ctx, params)
	defer stream.Close()

	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "image_generation.partial_image":
			if err := callback(image.StreamEvent{
				Type:              image.EventPartialImage,
				ImageBase64:       event.B64JSON,
				PartialImageIndex: int(event.PartialImageIndex),
				Size:              event.Size,
				Quality:           event.Quality,
			}); err != nil {
				return fmt.Errorf("callback error on partial image: %w", err)
			}
		case "image_generation.completed":
			if err := callback(image.StreamEvent{
				Type:        image.EventCompleted,
				ImageBase64: event.B64JSON,
				Size:        event.Size,
				Quality:     event.Quality,
			}); err != nil {
				return fmt.Errorf("callback error on completed image: %w", err)
			}
		}
	}
	if err := stream.Err(); err != nil {
		return fmt.Errorf("streaming error: %w", err)
	}
	return nil
}

// isGPTImage reports whether the API model is a gpt-image-* model, which rejects
// the response_format parameter (it always returns b64_json).
func isGPTImage(apiModel string) bool {
	switch apiModel {
	case "gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini":
		return true
	default:
		return false
	}
}

func mapResponse(resp *openaisdk.ImagesResponse, m model.ImageGenerationModel) *image.GenerationResponse {
	results := make([]image.GenerationResult, 0, len(resp.Data))
	for _, img := range resp.Data {
		result := image.GenerationResult{RevisedPrompt: img.RevisedPrompt}
		if img.URL != "" {
			result.ImageURL = img.URL
		}
		if img.B64JSON != "" {
			result.ImageBase64 = img.B64JSON
		}
		results = append(results, result)
	}
	return &image.GenerationResponse{Images: results, Model: m.APIModel}
}
