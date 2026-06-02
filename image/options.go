package image

import "errors"

// ErrEditNotSupported is returned when image editing is requested but the model
// (or vendor client) does not support it. Vendor clients that implement only
// generation return this from EditImage/EditImageStreaming.
var ErrEditNotSupported = errors.New("image editing not supported by this model")

// GenerationOptions contains per-call parameters for image generation and editing
// requests. Not every field is honored by every provider; unsupported fields are
// ignored by the vendor client.
//
// This is the overtura per-call surface (re-homed from the monolith fork). It
// coexists with the upstream "configure once, prompt many" Generation interface:
// vendor clients that support it accept these options on the optioned
// ImageGeneration interface (see editing.go), while the base GenerateImage(prompt)
// path remains for vendors that don't.
type GenerationOptions struct {
	// Size specifies the dimensions of the generated image (e.g., "1024x1024").
	Size string
	// Quality controls the quality level (e.g., "standard", "hd").
	Quality string
	// ResponseFormat specifies the response format ("url" or "b64_json").
	ResponseFormat string
	// N is the number of images to generate from the prompt.
	N int
	// InputImage is the source image bytes for editing. Callers provide raw bytes;
	// URL download is the caller's responsibility (avoids SSRF in the library).
	InputImage []byte
	// InputImages carries multiple reference images for multi-reference edits.
	// When non-empty, providers that support it use this instead of InputImage.
	InputImages [][]byte
	// Mask is the mask image bytes (PNG, transparent area = edit region). With
	// InputImages, the mask applies to the first image (OpenAI semantics).
	Mask []byte
	// InputFidelity controls how closely the edit follows the source ("low"/"high",
	// OpenAI GPT-Image-1).
	InputFidelity string
	// Background controls the generated background ("transparent"/"opaque"/"auto",
	// OpenAI GPT-Image-1).
	Background string
	// OutputFormat requests a provider-side format ("png"/"jpeg"/"webp").
	OutputFormat string
	// OutputCompression is a 1-100 quality knob for lossy formats; 0 means unset.
	OutputCompression int
	// Modalities lists output modalities for providers that require them (OpenRouter
	// image generation): []string{"image"} or []string{"image","text"}.
	Modalities []string
	// ImageSize is a provider-specific output-size tier (OpenRouter
	// image_config.image_size), e.g. "0.5K"/"1K"/"2K"/"4K". Independent from Quality.
	ImageSize string
}

// GenerationOption configures GenerationOptions.
type GenerationOption func(*GenerationOptions)

// WithSize sets the dimensions of the generated image ("WIDTHxHEIGHT").
func WithSize(size string) GenerationOption {
	return func(o *GenerationOptions) { o.Size = size }
}

// WithQuality sets the quality level ("standard"/"hd").
func WithQuality(quality string) GenerationOption {
	return func(o *GenerationOptions) { o.Quality = quality }
}

// WithResponseFormat sets how the image is returned ("url"/"b64_json").
func WithResponseFormat(format string) GenerationOption {
	return func(o *GenerationOptions) { o.ResponseFormat = format }
}

// WithN sets the number of images to generate.
func WithN(n int) GenerationOption {
	return func(o *GenerationOptions) { o.N = n }
}

// WithInputImage sets the source image bytes for editing.
func WithInputImage(data []byte) GenerationOption {
	return func(o *GenerationOptions) { o.InputImage = data }
}

// WithInputImages sets multiple reference images for a multi-reference edit.
// Empty entries are skipped.
func WithInputImages(images [][]byte) GenerationOption {
	return func(o *GenerationOptions) {
		filtered := make([][]byte, 0, len(images))
		for _, img := range images {
			if len(img) > 0 {
				filtered = append(filtered, img)
			}
		}
		o.InputImages = filtered
	}
}

// WithOutputFormat requests a provider-side output format ("png"/"jpeg"/"webp").
func WithOutputFormat(format string) GenerationOption {
	return func(o *GenerationOptions) { o.OutputFormat = format }
}

// WithOutputCompression sets quality 1-100 for lossy formats (0 = unset).
func WithOutputCompression(quality int) GenerationOption {
	return func(o *GenerationOptions) { o.OutputCompression = quality }
}

// WithMask sets the mask image bytes (PNG; transparent = edit region).
func WithMask(data []byte) GenerationOption {
	return func(o *GenerationOptions) { o.Mask = data }
}

// WithInputFidelity controls edit fidelity ("low"/"high", OpenAI GPT-Image-1).
func WithInputFidelity(fidelity string) GenerationOption {
	return func(o *GenerationOptions) { o.InputFidelity = fidelity }
}

// WithBackground controls the background ("transparent"/"opaque"/"auto").
func WithBackground(bg string) GenerationOption {
	return func(o *GenerationOptions) { o.Background = bg }
}

// WithModalities sets output modalities (OpenRouter image generation).
func WithModalities(modalities ...string) GenerationOption {
	return func(o *GenerationOptions) { o.Modalities = modalities }
}

// WithImageSize sets the provider-specific output-size tier (OpenRouter).
func WithImageSize(size string) GenerationOption {
	return func(o *GenerationOptions) { o.ImageSize = size }
}

// ApplyGenerationOptions builds a GenerationOptions from the given options. Vendor
// clients call this to resolve the per-call surface.
func ApplyGenerationOptions(opts ...GenerationOption) GenerationOptions {
	var o GenerationOptions
	for _, opt := range opts {
		opt(&o)
	}
	return o
}
