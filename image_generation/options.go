package image_generation

import (
	"net/http"
	"time"

	"github.com/joakimcarlsson/ai/model"
)

// GenerationOptions contains parameters for customizing image generation and editing requests.
type GenerationOptions struct {
	// Size specifies the dimensions of the generated image (e.g., "1024x1024").
	// Not supported by all providers.
	Size string
	// Quality controls the quality level of the generated image (e.g., "standard", "hd").
	Quality string
	// ResponseFormat specifies the format of the response ("url" or "b64_json").
	ResponseFormat string
	// N is the number of images to generate from the prompt.
	N int

	// InputImage is the source image bytes for editing operations.
	// Callers must provide raw bytes — URL download is the caller's responsibility
	// (avoids SSRF risk in the library).
	InputImage []byte
	// Mask is the mask image bytes (PNG, transparent area = edit region).
	Mask []byte
	// InputFidelity controls how closely the edit follows the source image.
	// Values: "low" or "high" (OpenAI GPT-Image-1 specific).
	InputFidelity string
	// Background controls the background of the generated image.
	// Values: "transparent", "opaque", "auto" (OpenAI GPT-Image-1 specific).
	Background string

	// Modalities is the list of output modalities for providers that require it
	// (currently OpenRouter image generation). Examples: []string{"image"} for
	// image-only models, []string{"image", "text"} for text+image models.
	// When empty, the provider falls back to its historical default.
	Modalities []string

	// ImageSize is a provider-specific output-size tier (currently OpenRouter
	// image_config.image_size). Examples: "0.5K", "1K", "2K", "4K". This is
	// independent from Quality (Quality controls rendering effort, ImageSize
	// controls pixel resolution). When empty, the provider falls back to a
	// quality-derived default.
	ImageSize string
}

// GenerationOption is a function that configures GenerationOptions.
type GenerationOption func(*GenerationOptions)

// WithSize sets the dimensions of the generated image.
// Not all providers support this option. Format is typically "WIDTHxHEIGHT" (e.g., "1024x1024").
func WithSize(size string) GenerationOption {
	return func(options *GenerationOptions) {
		options.Size = size
	}
}

// WithQuality sets the quality level of the generated image.
// Common values are "standard" and "hd" (high definition).
func WithQuality(quality string) GenerationOption {
	return func(options *GenerationOptions) {
		options.Quality = quality
	}
}

// WithResponseFormat specifies how the generated image should be returned.
// Valid values are "url" (returns a URL to the image) or "b64_json" (returns base64-encoded image data).
func WithResponseFormat(format string) GenerationOption {
	return func(options *GenerationOptions) {
		options.ResponseFormat = format
	}
}

// WithN sets the number of images to generate from the prompt.
// Most providers charge per image generated.
func WithN(n int) GenerationOption {
	return func(options *GenerationOptions) {
		options.N = n
	}
}

// WithInputImage sets the source image bytes for editing operations.
func WithInputImage(data []byte) GenerationOption {
	return func(options *GenerationOptions) {
		options.InputImage = data
	}
}

// WithMask sets the mask image bytes for editing operations.
// The mask should be a PNG where transparent areas indicate the region to edit.
func WithMask(data []byte) GenerationOption {
	return func(options *GenerationOptions) {
		options.Mask = data
	}
}

// WithInputFidelity controls how closely the edit follows the source image.
// Values: "low" or "high". Only supported by OpenAI GPT-Image-1.
func WithInputFidelity(fidelity string) GenerationOption {
	return func(options *GenerationOptions) {
		options.InputFidelity = fidelity
	}
}

// WithBackground controls the background of the generated image.
// Values: "transparent", "opaque", "auto". Only supported by OpenAI GPT-Image-1.
func WithBackground(bg string) GenerationOption {
	return func(options *GenerationOptions) {
		options.Background = bg
	}
}

// WithModalities sets the output modalities for providers that need them
// (currently OpenRouter image generation). Pass []string{"image"} for
// image-only models, []string{"image", "text"} for text+image models.
// When unset, OpenRouter falls back to its historical default of
// []string{"image", "text"}.
func WithModalities(modalities ...string) GenerationOption {
	return func(options *GenerationOptions) {
		options.Modalities = modalities
	}
}

// WithImageSize sets the provider-specific output-size tier (currently
// OpenRouter image_config.image_size). Examples: "0.5K", "1K", "2K", "4K".
// This is independent from quality. When unset, OpenRouter derives the
// size from quality via mapToImageSize.
func WithImageSize(size string) GenerationOption {
	return func(options *GenerationOptions) {
		options.ImageSize = size
	}
}

// WithAPIKey sets the API key for authentication with the image generation provider.
func WithAPIKey(apiKey string) ImageGenerationClientOption {
	return func(options *imageGenerationClientOptions) {
		options.apiKey = apiKey
	}
}

// WithModel specifies which image generation model to use for creating images.
func WithModel(m model.ImageGenerationModel) ImageGenerationClientOption {
	return func(options *imageGenerationClientOptions) {
		options.model = m
	}
}

// WithTimeout sets the maximum duration to wait for image generation requests to complete.
func WithTimeout(timeout time.Duration) ImageGenerationClientOption {
	return func(options *imageGenerationClientOptions) {
		options.timeout = &timeout
	}
}

// WithOpenAIOptions applies OpenAI-specific configuration options.
// Also used for xAI since it uses OpenAI-compatible API.
func WithOpenAIOptions(openaiOptions ...OpenAIOption) ImageGenerationClientOption {
	return func(options *imageGenerationClientOptions) {
		options.openaiOptions = openaiOptions
	}
}

// WithGeminiOptions applies Gemini-specific configuration options.
func WithGeminiOptions(geminiOptions ...GeminiOption) ImageGenerationClientOption {
	return func(options *imageGenerationClientOptions) {
		options.geminiOptions = geminiOptions
	}
}

// WithHTTPClient sets a custom HTTP client for providers that use raw HTTP
// (currently OpenRouter). Allows callers to inject custom transports for
// testing, proxies, TLS, or observability.
func WithHTTPClient(client *http.Client) ImageGenerationClientOption {
	return func(options *imageGenerationClientOptions) {
		options.httpClient = client
	}
}

// applyGenerationOptions creates a GenerationOptions with model defaults and overlays
// the provided functional options. Handles provider-specific ResponseFormat defaults.
func applyGenerationOptions(m model.ImageGenerationModel, defaultResponseFormat string, opts ...GenerationOption) GenerationOptions {
	o := GenerationOptions{
		Size:           m.DefaultSize,
		Quality:        m.DefaultQuality,
		ResponseFormat: defaultResponseFormat,
		N:              1,
	}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// isAllowedImageMIME validates MIME type against supported image formats.
func isAllowedImageMIME(mime string) bool {
	switch mime {
	case "image/png", "image/jpeg", "image/webp", "image/gif":
		return true
	default:
		return false
	}
}
