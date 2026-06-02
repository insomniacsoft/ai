package gemini

import (
	"errors"

	"github.com/joakimcarlsson/ai/image"
	"google.golang.org/genai"
)

// geminiImageConfig builds the genai.ImageConfig for native-image requests from
// the per-call options. Returns nil when neither Size nor ImageSize is set so
// the request omits image_config entirely.
func geminiImageConfig(opts image.GenerationOptions) *genai.ImageConfig {
	if opts.Size == "" && opts.ImageSize == "" {
		return nil
	}
	cfg := &genai.ImageConfig{}
	if opts.Size != "" {
		cfg.AspectRatio = mapToAspectRatio(opts.Size)
	}
	if opts.ImageSize != "" {
		cfg.ImageSize = opts.ImageSize
	}
	return cfg
}

// mapToAspectRatio normalizes a WIDTHxHEIGHT size (or an already-ratio string)
// to a Gemini aspect-ratio value. Ratio-format inputs (e.g. "4:3") pass through.
func mapToAspectRatio(size string) string {
	switch size {
	case "1024x1024", "512x512", "256x256":
		return "1:1"
	case "1024x1792", "1024x1536":
		return "9:16"
	case "1792x1024", "1536x1024":
		return "16:9"
	default:
		if size != "" {
			return size // passthrough for ratio format like "4:3"
		}
		return "1:1"
	}
}

// checkFinishReason maps a blocked/empty GenerateContent response to an error.
func checkFinishReason(resp *genai.GenerateContentResponse) error {
	if len(resp.Candidates) == 0 {
		return errors.New("no candidates in response")
	}
	switch resp.Candidates[0].FinishReason {
	case genai.FinishReasonSafety, genai.FinishReasonImageSafety:
		return errors.New("content blocked by safety filters")
	case genai.FinishReasonRecitation, genai.FinishReasonImageRecitation:
		return errors.New("content blocked: recitation detected")
	case genai.FinishReasonBlocklist:
		return errors.New("content blocked: blocklist match")
	case genai.FinishReasonProhibitedContent, genai.FinishReasonImageProhibitedContent:
		return errors.New("content blocked: prohibited content")
	default:
		return nil
	}
}
