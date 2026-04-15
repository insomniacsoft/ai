// Package image_generation provides a unified interface for generating and editing
// images using various AI providers (OpenAI, Gemini, OpenRouter, xAI).
package image_generation

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/joakimcarlsson/ai/model"
	"github.com/joakimcarlsson/ai/tracing"
)

// ImageGenerationUsage tracks the resource consumption for image generation operations.
type ImageGenerationUsage struct {
	// PromptTokens is the number of tokens in the input prompt.
	PromptTokens int64
}

// ImageGenerationResult represents a single generated image with its metadata.
type ImageGenerationResult struct {
	// ImageURL contains the URL to the generated image if ResponseFormat was "url".
	ImageURL string
	// ImageBase64 contains the base64-encoded image data if ResponseFormat was "b64_json".
	ImageBase64 string
	// RevisedPrompt contains the prompt that was actually used to generate the image.
	RevisedPrompt string
}

// ImageGenerationResponse contains the generated images and metadata from an image generation request.
type ImageGenerationResponse struct {
	// Images contains the generated image results.
	Images []ImageGenerationResult
	// Usage tracks resource consumption for this request.
	Usage ImageGenerationUsage
	// Model identifies which image generation model was used.
	Model string
}

// ImageStreamEventType identifies the type of streaming event during image generation.
type ImageStreamEventType string

const (
	// EventPartialImage is emitted when a partial image is available during streaming.
	EventPartialImage ImageStreamEventType = "partial_image"
	// EventCompleted is emitted when image generation is complete and the final image is available.
	EventCompleted ImageStreamEventType = "completed"
)

// ImageStreamEvent represents a streaming event during image generation.
type ImageStreamEvent struct {
	// Type identifies the kind of streaming event.
	Type ImageStreamEventType `json:"type"`
	// ImageBase64 contains the base64-encoded image data.
	ImageBase64 string `json:"image_base64"`
	// PartialImageIndex is the 0-based index of the partial image (only for partial_image events).
	PartialImageIndex int `json:"partial_image_index,omitempty"`
	// Size is the dimensions of the image.
	Size string `json:"size,omitempty"`
	// Quality is the quality setting of the image.
	Quality string `json:"quality,omitempty"`
}

// StreamCallback is called for each streaming event during image generation.
type StreamCallback func(ImageStreamEvent) error

// ErrStreamingNotSupported is returned when streaming is requested but the model doesn't support it.
var ErrStreamingNotSupported = errors.New(
	"streaming not supported by this model",
)

// ErrEditNotSupported is returned when editing is requested but the provider/model doesn't support it.
var ErrEditNotSupported = errors.New(
	"image editing is not supported by this provider/model",
)

// ImageGeneration defines the interface for generating and editing images.
type ImageGeneration interface {
	GenerateImage(
		ctx context.Context,
		prompt string,
		options ...GenerationOption,
	) (*ImageGenerationResponse, error)

	GenerateImageStreaming(
		ctx context.Context,
		prompt string,
		callback StreamCallback,
		options ...GenerationOption,
	) error

	// EditImage modifies an existing image. Source image via WithInputImage().
	EditImage(
		ctx context.Context,
		prompt string,
		options ...GenerationOption,
	) (*ImageGenerationResponse, error)

	// EditImageStreaming streams partial previews during image editing.
	EditImageStreaming(
		ctx context.Context,
		prompt string,
		callback StreamCallback,
		options ...GenerationOption,
	) error

	// SupportsEditing returns true if this provider supports image editing.
	SupportsEditing() bool

	// Model returns the image generation model configuration being used.
	Model() model.ImageGenerationModel
}

type imageGenerationClientOptions struct {
	apiKey     string
	model      model.ImageGenerationModel
	timeout    *time.Duration
	httpClient *http.Client

	openaiOptions []OpenAIOption
	geminiOptions []GeminiOption
}

// ImageGenerationClientOption configures an image generation client.
type ImageGenerationClientOption func(*imageGenerationClientOptions)

// ImageGenerationClient is the internal interface implemented by provider-specific clients.
type ImageGenerationClient interface {
	generate(
		ctx context.Context,
		prompt string,
		options ...GenerationOption,
	) (*ImageGenerationResponse, error)
}

// StreamingImageGenerationClient is an optional interface for clients that support streaming.
type StreamingImageGenerationClient interface {
	generateStreaming(
		ctx context.Context,
		prompt string,
		callback StreamCallback,
		options ...GenerationOption,
	) error
}

// EditingImageGenerationClient is an optional interface for clients that support editing.
type EditingImageGenerationClient interface {
	edit(
		ctx context.Context,
		prompt string,
		options ...GenerationOption,
	) (*ImageGenerationResponse, error)
}

// StreamingEditingImageGenerationClient is optional for clients that support streaming edits.
type StreamingEditingImageGenerationClient interface {
	editStreaming(
		ctx context.Context,
		prompt string,
		callback StreamCallback,
		options ...GenerationOption,
	) error
}

type baseImageGeneration[C ImageGenerationClient] struct {
	options imageGenerationClientOptions
	client  C
}

// NewImageGeneration creates a new image generation client for the specified provider.
// Supported providers include OpenAI, xAI, and Gemini. Use WithModel() to specify the image generation model
// and WithAPIKey() for authentication.
func NewImageGeneration(
	provider model.Provider,
	opts ...ImageGenerationClientOption,
) (ImageGeneration, error) {
	clientOptions := imageGenerationClientOptions{}
	for _, o := range opts {
		o(&clientOptions)
	}

	switch provider {
	case model.ProviderOpenAI:
		return &baseImageGeneration[OpenAIClient]{
			options: clientOptions,
			client:  newOpenAIClient(clientOptions),
		}, nil
	case model.ProviderXAI:
		clientOptions.openaiOptions = append(clientOptions.openaiOptions,
			WithOpenAIBaseURL("https://api.x.ai/v1"),
		)
		return &baseImageGeneration[OpenAIClient]{
			options: clientOptions,
			client:  newOpenAIClient(clientOptions),
		}, nil
	case model.ProviderGemini:
		if isNativeModel(clientOptions.model) {
			return &baseImageGeneration[GeminiNativeClient]{
				options: clientOptions,
				client:  newGeminiNativeClient(clientOptions),
			}, nil
		}
		return &baseImageGeneration[GeminiClient]{
			options: clientOptions,
			client:  newGeminiClient(clientOptions),
		}, nil
	}

	return nil, errors.New("image generation provider not supported: " + string(provider))
}

func (i *baseImageGeneration[C]) GenerateImage(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*ImageGenerationResponse, error) {
	start := time.Now()
	ctx, span := tracing.StartImageSpan(
		ctx,
		i.options.model.APIModel,
		string(i.options.model.Provider),
	)
	defer span.End()

	resp, err := i.client.generate(ctx, prompt, options...)
	if err != nil {
		tracing.SetError(span, err)
		tracing.RecordMetrics(
			ctx,
			"generate_image",
			i.options.model.APIModel,
			string(i.options.model.Provider),
			time.Since(start),
			0,
			0,
			err,
		)
		return nil, err
	}

	tracing.SetResponseAttrs(span,
		tracing.AttrUsageInputTokens.Int64(int64(resp.Usage.PromptTokens)),
		tracing.AttrResultCount.Int(len(resp.Images)),
	)
	tracing.RecordMetrics(
		ctx,
		"generate_image",
		i.options.model.APIModel,
		string(i.options.model.Provider),
		time.Since(start),
		int64(resp.Usage.PromptTokens),
		0,
		nil,
	)
	return resp, nil
}

func (i *baseImageGeneration[C]) GenerateImageStreaming(
	ctx context.Context,
	prompt string,
	callback StreamCallback,
	options ...GenerationOption,
) error {
	start := time.Now()
	ctx, span := tracing.StartImageSpan(
		ctx,
		i.options.model.APIModel,
		string(i.options.model.Provider),
	)
	defer span.End()

	if streamingClient, ok := any(i.client).(StreamingImageGenerationClient); ok {
		err := streamingClient.generateStreaming(
			ctx,
			prompt,
			callback,
			options...)
		tracing.RecordMetrics(
			ctx,
			"generate_image",
			i.options.model.APIModel,
			string(i.options.model.Provider),
			time.Since(start),
			0,
			0,
			err,
		)
		if err != nil {
			tracing.SetError(span, err)
		}
		return err
	}
	return ErrStreamingNotSupported
}

func (i *baseImageGeneration[C]) EditImage(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*ImageGenerationResponse, error) {
	start := time.Now()
	ctx, span := tracing.StartImageSpan(
		ctx,
		i.options.model.APIModel,
		string(i.options.model.Provider),
	)
	defer span.End()

	editClient, ok := any(i.client).(EditingImageGenerationClient)
	if !ok {
		return nil, ErrEditNotSupported
	}

	resp, err := editClient.edit(ctx, prompt, options...)
	if err != nil {
		tracing.SetError(span, err)
		tracing.RecordMetrics(
			ctx,
			"edit_image",
			i.options.model.APIModel,
			string(i.options.model.Provider),
			time.Since(start),
			0,
			0,
			err,
		)
		return nil, err
	}

	tracing.SetResponseAttrs(span,
		tracing.AttrUsageInputTokens.Int64(int64(resp.Usage.PromptTokens)),
		tracing.AttrResultCount.Int(len(resp.Images)),
	)
	tracing.RecordMetrics(
		ctx,
		"edit_image",
		i.options.model.APIModel,
		string(i.options.model.Provider),
		time.Since(start),
		int64(resp.Usage.PromptTokens),
		0,
		nil,
	)
	return resp, nil
}

func (i *baseImageGeneration[C]) EditImageStreaming(
	ctx context.Context,
	prompt string,
	callback StreamCallback,
	options ...GenerationOption,
) error {
	start := time.Now()
	ctx, span := tracing.StartImageSpan(
		ctx,
		i.options.model.APIModel,
		string(i.options.model.Provider),
	)
	defer span.End()

	editClient, ok := any(i.client).(StreamingEditingImageGenerationClient)
	if !ok {
		return ErrEditNotSupported
	}

	err := editClient.editStreaming(ctx, prompt, callback, options...)
	tracing.RecordMetrics(
		ctx,
		"edit_image",
		i.options.model.APIModel,
		string(i.options.model.Provider),
		time.Since(start),
		0,
		0,
		err,
	)
	if err != nil {
		tracing.SetError(span, err)
	}
	return err
}

func (i *baseImageGeneration[C]) SupportsEditing() bool {
	_, ok := any(i.client).(EditingImageGenerationClient)
	return ok
}

func (i *baseImageGeneration[C]) Model() model.ImageGenerationModel {
	return i.options.model
}

func isNativeModel(m model.ImageGenerationModel) bool {
	switch m.ID {
	case model.Gemini25FlashImage, model.Gemini3ProImage:
		return true
	default:
		return false
	}
}


