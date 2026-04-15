package image_generation

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"google.golang.org/genai"
)

// GeminiNativeClient implements image generation and editing using the Gemini
// GenerateContent API with IMAGE response modality. Unlike the Imagen-based
// GeminiClient, this supports conversational image editing via inline image data.
type GeminiNativeClient struct {
	client  *genai.Client
	options imageGenerationClientOptions
}

func newGeminiNativeClient(opts imageGenerationClientOptions) GeminiNativeClient {
	geminiOpts := geminiOptions{
		backend: genai.BackendGeminiAPI,
	}

	for _, o := range opts.geminiOptions {
		o(&geminiOpts)
	}

	client, err := genai.NewClient(
		context.Background(),
		&genai.ClientConfig{
			APIKey:  opts.apiKey,
			Backend: geminiOpts.backend,
		},
	)
	if err != nil {
		return GeminiNativeClient{}
	}

	return GeminiNativeClient{client: client, options: opts}
}

func (g GeminiNativeClient) generate(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*ImageGenerationResponse, error) {
	_ = applyGenerationOptions(g.options.model, "b64_json", options...)

	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"IMAGE", "TEXT"},
	}

	if g.options.timeout != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *g.options.timeout)
		defer cancel()
	}

	contents := genai.Text(prompt)

	resp, err := g.client.Models.GenerateContent(
		ctx, g.options.model.APIModel, contents, config,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate image: %w", err)
	}

	return g.mapResponse(resp)
}

func (g GeminiNativeClient) edit(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*ImageGenerationResponse, error) {
	opts := applyGenerationOptions(g.options.model, "b64_json", options...)

	if len(opts.InputImage) == 0 {
		return nil, fmt.Errorf("input image required for editing: use WithInputImage(data)")
	}

	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"IMAGE", "TEXT"},
	}

	if g.options.timeout != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *g.options.timeout)
		defer cancel()
	}

	mimeType := detectImageMIME(opts.InputImage)
	if !isAllowedImageMIME(mimeType) {
		return nil, fmt.Errorf("unsupported image type: %s", mimeType)
	}
	parts := []*genai.Part{
		{InlineData: &genai.Blob{
			MIMEType: mimeType,
			Data:     opts.InputImage,
		}},
		{Text: prompt},
	}

	contents := []*genai.Content{{
		Role:  genai.RoleUser,
		Parts: parts,
	}}

	resp, err := g.client.Models.GenerateContent(
		ctx, g.options.model.APIModel, contents, config,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to edit image: %w", err)
	}

	return g.mapResponse(resp)
}

func (g GeminiNativeClient) mapResponse(
	resp *genai.GenerateContentResponse,
) (*ImageGenerationResponse, error) {
	if err := checkFinishReason(resp); err != nil {
		return nil, err
	}

	var results []ImageGenerationResult

	for _, candidate := range resp.Candidates {
		if candidate.Content == nil {
			continue
		}
		for _, part := range candidate.Content.Parts {
			if part.InlineData != nil &&
				strings.HasPrefix(part.InlineData.MIMEType, "image/") {
				b64 := base64.StdEncoding.EncodeToString(part.InlineData.Data)
				results = append(results, ImageGenerationResult{
					ImageBase64: b64,
				})
			}
		}
	}

	if len(results) == 0 {
		return nil, errors.New("no image generated in response")
	}

	return &ImageGenerationResponse{
		Images: results,
		Model:  g.options.model.APIModel,
	}, nil
}

func checkFinishReason(resp *genai.GenerateContentResponse) error {
	if len(resp.Candidates) == 0 {
		return errors.New("no candidates in response")
	}
	reason := resp.Candidates[0].FinishReason
	switch reason {
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

func detectImageMIME(data []byte) string {
	if len(data) == 0 {
		return "application/octet-stream"
	}
	return http.DetectContentType(data)
}
