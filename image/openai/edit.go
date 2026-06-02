package openai

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/joakimcarlsson/ai/image"
	"github.com/joakimcarlsson/ai/model"
	openaisdk "github.com/openai/openai-go"
)

// EditImage modifies an existing image via /v1/images/edits. The source image is
// provided through [image.WithInputImage] (single) or [image.WithInputImages]
// (multi-reference). Quality is intentionally not forwarded — the edits endpoint
// rejects it with a 400 (it is valid only on Images.Generate).
func (c *Client) EditImage(
	ctx context.Context,
	prompt string,
	options ...image.GenerationOption,
) (*image.GenerationResponse, error) {
	opts := image.ApplyGenerationOptionsWithModelDefaults(c.options.model, "url", options...)
	if len(opts.InputImage) == 0 && len(opts.InputImages) == 0 {
		return nil, fmt.Errorf("input image required for editing: use WithInputImage(data) or WithInputImages([][]byte)")
	}

	imageUnion, err := buildImageEditUnion(opts)
	if err != nil {
		return nil, err
	}
	params := openaisdk.ImageEditParams{
		Prompt: prompt,
		Model:  openaisdk.ImageModel(c.options.model.APIModel),
		Image:  imageUnion,
	}
	applyEditParams(&params, c.options.model, opts)

	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	response, err := c.client.Images.Edit(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to edit image: %w", err)
	}
	return mapResponse(response, c.options.model), nil
}

// EditImageStreaming streams partial previews during image editing. The
// /v1/images/edits endpoint emits "image_edit.*" event types (the generate
// endpoint emits "image_generation.*"); mixing them silently drops every event,
// leaving an empty result after the full latency.
func (c *Client) EditImageStreaming(
	ctx context.Context,
	prompt string,
	callback image.StreamCallback,
	options ...image.GenerationOption,
) error {
	opts := image.ApplyGenerationOptionsWithModelDefaults(c.options.model, "url", options...)
	if len(opts.InputImage) == 0 && len(opts.InputImages) == 0 {
		return fmt.Errorf("input image required for editing: use WithInputImage(data) or WithInputImages([][]byte)")
	}

	imageUnion, err := buildImageEditUnion(opts)
	if err != nil {
		return err
	}
	params := openaisdk.ImageEditParams{
		Prompt:        prompt,
		Model:         openaisdk.ImageModel(c.options.model.APIModel),
		Image:         imageUnion,
		PartialImages: openaisdk.Int(int64(c.options.streamingOptions.PartialImages)),
	}
	applyEditParams(&params, c.options.model, opts)

	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	stream := c.client.Images.EditStreaming(ctx, params)
	defer stream.Close()

	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "image_edit.partial_image":
			if err := callback(image.StreamEvent{
				Type:              image.EventPartialImage,
				ImageBase64:       event.B64JSON,
				PartialImageIndex: int(event.PartialImageIndex),
			}); err != nil {
				return fmt.Errorf("callback error on partial image: %w", err)
			}
		case "image_edit.completed":
			if err := callback(image.StreamEvent{
				Type:        image.EventCompleted,
				ImageBase64: event.B64JSON,
			}); err != nil {
				return fmt.Errorf("callback error on completed image: %w", err)
			}
		}
	}
	return stream.Err()
}

// applyEditParams forwards the size + edit-only knobs (mask, input fidelity,
// background, output format/compression) honored by the edits endpoint.
func applyEditParams(params *openaisdk.ImageEditParams, m model.ImageGenerationModel, opts image.GenerationOptions) {
	if opts.Size != "" && len(m.SupportedSizes) > 0 {
		params.Size = openaisdk.ImageEditParamsSize(opts.Size)
	}
	if opts.Mask != nil {
		params.Mask = newNamedImageReader(opts.Mask)
	}
	if opts.InputFidelity != "" {
		params.InputFidelity = openaisdk.ImageEditParamsInputFidelity(opts.InputFidelity)
	}
	if opts.Background != "" {
		params.Background = openaisdk.ImageEditParamsBackground(opts.Background)
	}
	if opts.OutputFormat != "" {
		params.OutputFormat = openaisdk.ImageEditParamsOutputFormat(opts.OutputFormat)
	}
	if opts.OutputCompression > 0 && opts.OutputFormat != "png" && opts.OutputFormat != "" {
		params.OutputCompression = openaisdk.Int(int64(opts.OutputCompression))
	}
}

// buildImageEditUnion picks the right openai.ImageEditParamsImageUnion variant
// (single OfFile vs multi-reference OfFileArray) based on whether the caller
// provided InputImages. Each entry is wrapped in a namedImageReader so the
// openai-go multipart encoder picks the right Filename/ContentType per file.
// Pre-flight MIME validation runs on every reference; one bad blob fails the
// whole call rather than silently uploading garbage.
func buildImageEditUnion(opts image.GenerationOptions) (openaisdk.ImageEditParamsImageUnion, error) {
	if len(opts.InputImages) > 0 {
		readers := make([]io.Reader, 0, len(opts.InputImages))
		for i, raw := range opts.InputImages {
			r := newNamedImageReader(raw)
			if !isAllowedImageMIME(r.contentType) {
				return openaisdk.ImageEditParamsImageUnion{},
					fmt.Errorf("unsupported image type at reference %d: %s", i, r.contentType)
			}
			readers = append(readers, r)
		}
		return openaisdk.ImageEditParamsImageUnion{OfFileArray: readers}, nil
	}
	r := newNamedImageReader(opts.InputImage)
	if !isAllowedImageMIME(r.contentType) {
		return openaisdk.ImageEditParamsImageUnion{},
			fmt.Errorf("unsupported image type: %s", r.contentType)
	}
	return openaisdk.ImageEditParamsImageUnion{OfFile: r}, nil
}

// namedImageReader carries a Filename + ContentType alongside the bytes so the
// openai-go multipart encoder picks them up via its
// `interface{ Filename() string }` / `interface{ ContentType() string }`
// type-assertions (internal/apiform/encoder.go). Without this the SDK falls back
// to filename="anonymous_file" + Content-Type="application/octet-stream", which
// gpt-image-1's images/edits endpoint rejects with a 400 "invalid_request" — the
// upload bytes never reach the model. http.DetectContentType reads the magic
// bytes, so the caller doesn't need to track origin format.
type namedImageReader struct {
	*bytes.Reader
	filename    string
	contentType string
}

func newNamedImageReader(data []byte) *namedImageReader {
	mime := http.DetectContentType(data)
	return &namedImageReader{
		Reader:      bytes.NewReader(data),
		filename:    "image" + extForMIME(mime),
		contentType: mime,
	}
}

// Close is a no-op; satisfies io.ReadCloser for callers that close.
func (r *namedImageReader) Close() error { return nil }

// Filename is read by openai-go's multipart encoder.
func (r *namedImageReader) Filename() string { return r.filename }

// ContentType is read by openai-go's multipart encoder.
func (r *namedImageReader) ContentType() string { return r.contentType }

func extForMIME(mime string) string {
	switch mime {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "image/gif":
		return ".gif"
	default:
		return ".bin"
	}
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
