package image_generation

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenRouterClient implements image generation and editing using the OpenRouter API.
// Uses raw HTTP because OpenRouter's image extensions (modalities, image_config)
// are not in the standard OpenAI SDK types.
type OpenRouterClient struct {
	httpClient *http.Client
	baseURL    string
	options    imageGenerationClientOptions
}

func newOpenRouterClient(opts imageGenerationClientOptions) OpenRouterClient {
	httpClient := opts.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}
	return OpenRouterClient{
		httpClient: httpClient,
		baseURL:    "https://openrouter.ai/api/v1",
		options:    opts,
	}
}

func (o OpenRouterClient) generate(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*ImageGenerationResponse, error) {
	opts := applyGenerationOptions(o.options.model, "b64_json", options...)

	body := map[string]any{
		"model":      o.options.model.APIModel,
		"messages":   []map[string]any{{"role": "user", "content": prompt}},
		"modalities": []string{"image", "text"},
		"image_config": map[string]any{
			"aspect_ratio": mapToAspectRatio(opts.Size),
			"image_size":   mapToImageSize(opts.Quality),
		},
	}

	return o.doImageRequest(ctx, body)
}

func (o OpenRouterClient) edit(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*ImageGenerationResponse, error) {
	opts := applyGenerationOptions(o.options.model, "b64_json", options...)

	if len(opts.InputImage) == 0 {
		return nil, fmt.Errorf("input image required for editing: use WithInputImage(data)")
	}

	mimeType := http.DetectContentType(opts.InputImage)
	if !isAllowedImageMIME(mimeType) {
		return nil, fmt.Errorf("unsupported image type: %s", mimeType)
	}

	b64 := base64.StdEncoding.EncodeToString(opts.InputImage)
	content := []map[string]any{
		{"type": "text", "text": prompt},
		{"type": "image_url", "image_url": map[string]any{
			"url": "data:" + mimeType + ";base64," + b64,
		}},
	}

	body := map[string]any{
		"model":      o.options.model.APIModel,
		"messages":   []map[string]any{{"role": "user", "content": content}},
		"modalities": []string{"image", "text"},
	}

	return o.doImageRequest(ctx, body)
}

func (o OpenRouterClient) doImageRequest(
	ctx context.Context,
	body map[string]any,
) (*ImageGenerationResponse, error) {
	if o.options.timeout != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *o.options.timeout)
		defer cancel()
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		o.baseURL+"/chat/completions",
		bytes.NewReader(jsonBody),
	)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.options.apiKey)

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openrouter request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openrouter error (status %d): %s",
			resp.StatusCode, truncate(string(respBody), 200))
	}

	return o.mapResponse(respBody)
}

// openRouterResponse matches the OpenRouter chat completions response shape.
type openRouterResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			Images  []struct {
				Type     string `json:"type"`
				ImageURL struct {
					URL string `json:"url"`
				} `json:"image_url"`
			} `json:"images"`
		} `json:"message"`
	} `json:"choices"`
}

func (o OpenRouterClient) mapResponse(body []byte) (*ImageGenerationResponse, error) {
	var resp openRouterResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices in openrouter response")
	}

	var results []ImageGenerationResult
	for _, img := range resp.Choices[0].Message.Images {
		b64 := extractBase64FromDataURI(img.ImageURL.URL)
		if b64 != "" {
			results = append(results, ImageGenerationResult{ImageBase64: b64})
		}
	}

	if len(results) == 0 {
		return nil, errors.New("no images in openrouter response")
	}

	return &ImageGenerationResponse{
		Images: results,
		Model:  o.options.model.APIModel,
	}, nil
}

func extractBase64FromDataURI(uri string) string {
	if idx := strings.Index(uri, ";base64,"); idx != -1 {
		return uri[idx+8:]
	}
	return uri
}

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

func mapToImageSize(quality string) string {
	switch quality {
	case "low":
		return "0.5K"
	case "standard", "medium", "default", "":
		return "1K"
	case "high", "hd":
		return "2K"
	default:
		return "1K"
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
