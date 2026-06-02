// Package openrouter provides an OpenRouter implementation of the optioned
// [image.ImageGeneration] interface (generation + editing).
//
// This is a FORK-ONLY module — upstream joakimcarlsson/ai has no OpenRouter image
// vendor. It uses raw HTTP because OpenRouter's image extensions (modalities,
// image_config) are not in the standard OpenAI SDK types. Re-homed from the
// monolith fork's image_generation/openrouter.go.
package openrouter

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

	"github.com/joakimcarlsson/ai/image"
	"github.com/joakimcarlsson/ai/model"
)

const (
	defaultBaseURL  = "https://openrouter.ai/api/v1"
	maxResponseSize = 50 * 1024 * 1024 // 50 MB
)

// Options configures the OpenRouter image generation client.
type Options struct {
	apiKey     string
	model      model.ImageGenerationModel
	timeout    *time.Duration
	httpClient *http.Client
}

// Option configures Options.
type Option func(*Options)

// WithAPIKey sets the API key used to authenticate with OpenRouter.
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

// WithHTTPClient injects a custom HTTP client (transports, proxies, TLS,
// observability, testing). Defaults to a client with a 120s timeout.
func WithHTTPClient(client *http.Client) Option {
	return func(o *Options) { o.httpClient = client }
}

// Client implements [image.ImageGeneration] against the OpenRouter chat
// completions endpoint with image output modalities.
type Client struct {
	options    Options
	httpClient *http.Client
	baseURL    string
}

// NewGeneration constructs an OpenRouter image generation client. The returned
// [image.ImageGeneration] is wrapped with [image.WithEditingTracing].
func NewGeneration(opts ...Option) image.ImageGeneration {
	var options Options
	for _, o := range opts {
		o(&options)
	}
	httpClient := options.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}
	return image.WithEditingTracing(&Client{
		options:    options,
		httpClient: httpClient,
		baseURL:    defaultBaseURL,
	}, image.TracingAttrs{})
}

// Model returns the configured image generation model.
func (c *Client) Model() model.ImageGenerationModel { return c.options.model }

// SupportsEditing reports that the OpenRouter image client supports editing
// (image_url content blocks for multi-turn editing).
func (c *Client) SupportsEditing() bool { return true }

// GenerateImage creates an image from a text prompt.
func (c *Client) GenerateImage(
	ctx context.Context,
	prompt string,
	options ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	opts := image.ApplyGenerationOptionsWithModelDefaults(c.options.model, "b64_json", options...)

	body := map[string]any{
		"model":      c.options.model.APIModel,
		"messages":   []map[string]any{{"role": "user", "content": prompt}},
		"modalities": resolveModalities(c.options.model.OutputModalities, opts.Modalities),
		"image_config": map[string]any{
			"aspect_ratio": mapToAspectRatio(opts.Size),
			"image_size":   resolveImageSize(opts.ImageSize, opts.Quality),
		},
	}
	return c.doImageRequest(ctx, body)
}

// EditImage edits one or more reference images. Provide them via
// [image.WithInputImage] (single) or [image.WithInputImages] (multi-reference).
func (c *Client) EditImage(
	ctx context.Context,
	prompt string,
	options ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	opts := image.ApplyGenerationOptionsWithModelDefaults(c.options.model, "b64_json", options...)
	if len(opts.InputImage) == 0 && len(opts.InputImages) == 0 {
		return nil, fmt.Errorf("input image required for editing: use WithInputImage(data) or WithInputImages([][]byte)")
	}

	images := opts.InputImages
	if len(images) == 0 {
		images = [][]byte{opts.InputImage}
	}

	content := []map[string]any{{"type": "text", "text": prompt}}
	for i, raw := range images {
		mimeType := http.DetectContentType(raw)
		if !isAllowedImageMIME(mimeType) {
			return nil, fmt.Errorf("unsupported image type at reference %d: %s", i, mimeType)
		}
		b64 := base64.StdEncoding.EncodeToString(raw)
		content = append(content, map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": "data:" + mimeType + ";base64," + b64},
		})
	}

	body := map[string]any{
		"model":      c.options.model.APIModel,
		"messages":   []map[string]any{{"role": "user", "content": content}},
		"modalities": resolveModalities(c.options.model.OutputModalities, opts.Modalities),
	}
	if opts.ImageSize != "" || opts.Size != "" {
		body["image_config"] = map[string]any{
			"aspect_ratio": mapToAspectRatio(opts.Size),
			"image_size":   resolveImageSize(opts.ImageSize, opts.Quality),
		}
	}
	return c.doImageRequest(ctx, body)
}

// GenerateImageStreaming is not supported by the OpenRouter image client.
func (c *Client) GenerateImageStreaming(
	context.Context, string, image.StreamCallback, ...image.GenerationOption,
) error {
	return image.ErrStreamingNotSupported
}

// EditImageStreaming is not supported by the OpenRouter image client (editing is
// supported, but only non-streamed).
func (c *Client) EditImageStreaming(
	context.Context, string, image.StreamCallback, ...image.GenerationOption,
) error {
	return image.ErrStreamingNotSupported
}

func (c *Client) doImageRequest(ctx context.Context, body map[string]any) (*image.GenerationResponse, error) {
	if c.options.timeout != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *c.options.timeout)
		defer cancel()
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.options.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openrouter request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openrouter error (status %d): %s", resp.StatusCode, truncate(string(respBody), 200))
	}
	return c.mapResponse(respBody)
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

func (c *Client) mapResponse(body []byte) (*image.GenerationResponse, error) {
	var resp openRouterResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices in openrouter response")
	}

	var results []image.GenerationResult
	for _, img := range resp.Choices[0].Message.Images {
		if b64 := extractBase64FromDataURI(img.ImageURL.URL); b64 != "" {
			results = append(results, image.GenerationResult{ImageBase64: b64})
		}
	}
	if len(results) == 0 {
		return nil, errors.New("no images in openrouter response")
	}
	return &image.GenerationResponse{Images: results, Model: c.options.model.APIModel}, nil
}

func extractBase64FromDataURI(uri string) string {
	if idx := strings.Index(uri, ";base64,"); idx != -1 {
		return uri[idx+8:]
	}
	return uri
}

// mapToAspectRatio normalizes a WIDTHxHEIGHT size (or ratio string) to an
// OpenRouter aspect_ratio value. Ratio-format inputs pass through.
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
			return size
		}
		return "1:1"
	}
}

// mapToImageSize derives the OpenRouter image_config.image_size tier from quality.
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

// resolveModalities returns the output modalities to send. Caller-supplied
// modalities (WithModalities) take precedence — required for image-only models
// that must NOT send "text". When the caller is silent, the model's configured
// OutputModalities is used, then the historical default ["image","text"].
func resolveModalities(modelOutputModalities, callerModalities []string) []string {
	if len(callerModalities) > 0 {
		return callerModalities
	}
	if len(modelOutputModalities) > 0 {
		return modelOutputModalities
	}
	return []string{"image", "text"}
}

// resolveImageSize returns the OpenRouter image_config.image_size. An explicit
// caller ImageSize (WithImageSize) takes precedence (1K/2K/4K independent of
// Quality); otherwise it is derived from Quality.
func resolveImageSize(callerSize, quality string) string {
	if callerSize != "" {
		return callerSize
	}
	return mapToImageSize(quality)
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// isAllowedImageMIME validates a MIME type against the supported image formats.
func isAllowedImageMIME(mime string) bool {
	switch mime {
	case "image/png", "image/jpeg", "image/webp", "image/gif":
		return true
	default:
		return false
	}
}
