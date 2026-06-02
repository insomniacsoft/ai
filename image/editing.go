package image

import (
	"context"
	"time"

	"github.com/joakimcarlsson/ai/model"
	"github.com/joakimcarlsson/ai/tracing"
)

// ImageGeneration is the per-call optioned image interface. It extends the
// construct-once [Generation] surface (prompt-only, used by image/xai) with a
// per-call [GenerationOption] surface and image editing. Every method accepts
// ...GenerationOption so callers can vary size/quality/edit inputs per request.
//
// Vendors that do not support a capability return the matching sentinel:
// GenerateImageStreaming/EditImageStreaming return [ErrStreamingNotSupported];
// EditImage/EditImageStreaming return [ErrEditNotSupported]. Call
// SupportsEditing to probe edit capability before issuing an edit.
//
// This is the overtura optioned surface re-homed from the monolith fork. The
// monolith expressed optional capabilities (edit, streaming) as separate
// interfaces with UNEXPORTED methods (generate/edit/...) that a central
// baseImageGeneration[C] dispatched via type assertion. That pattern cannot
// survive the multi-module split: unexported interface methods are
// implementable only inside the declaring package, so a vendor module
// (image/openai, image/gemini) could never satisfy them. The exported-method
// interface below is the faithful multi-module equivalent — it mirrors how
// upstream's own [Generation] uses exported methods so vendor packages can
// implement it.
type ImageGeneration interface {
	// GenerateImage creates one or more images from a text prompt.
	GenerateImage(
		ctx context.Context,
		prompt string,
		options ...GenerationOption,
	) (*GenerationResponse, error)

	// GenerateImageStreaming streams partial images during generation. Returns
	// [ErrStreamingNotSupported] if the model doesn't support streaming.
	GenerateImageStreaming(
		ctx context.Context,
		prompt string,
		callback StreamCallback,
		options ...GenerationOption,
	) error

	// EditImage modifies an existing image. Provide the source image via
	// [WithInputImage] (or [WithInputImages] for multi-reference edits).
	// Returns [ErrEditNotSupported] if the model doesn't support editing.
	EditImage(
		ctx context.Context,
		prompt string,
		options ...GenerationOption,
	) (*GenerationResponse, error)

	// EditImageStreaming streams partial previews during image editing.
	// Returns [ErrEditNotSupported] if editing is unsupported, or
	// [ErrStreamingNotSupported] if editing is supported but not streamed.
	EditImageStreaming(
		ctx context.Context,
		prompt string,
		callback StreamCallback,
		options ...GenerationOption,
	) error

	// SupportsEditing reports whether the configured model can edit images.
	SupportsEditing() bool

	// Model returns the image generation model configuration being used.
	Model() model.ImageGenerationModel
}

// WithEditingTracing wraps an optioned [ImageGeneration] client so every call
// records OpenTelemetry spans and metrics. It mirrors [WithTracing] for the
// [Generation] surface; vendor packages (image/openai, image/gemini) return
// their concrete client wrapped in this so callers always get tracing.
func WithEditingTracing(inner ImageGeneration, attrs TracingAttrs) ImageGeneration {
	return &editingTracingClient{inner: inner, attrs: attrs}
}

type editingTracingClient struct {
	inner ImageGeneration
	attrs TracingAttrs
}

func (t *editingTracingClient) Model() model.ImageGenerationModel { return t.inner.Model() }

func (t *editingTracingClient) SupportsEditing() bool { return t.inner.SupportsEditing() }

func (t *editingTracingClient) GenerateImage(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*GenerationResponse, error) {
	return t.traced(ctx, "generate_image", func(ctx context.Context) (*GenerationResponse, error) {
		return t.inner.GenerateImage(ctx, prompt, options...)
	})
}

func (t *editingTracingClient) EditImage(
	ctx context.Context,
	prompt string,
	options ...GenerationOption,
) (*GenerationResponse, error) {
	return t.traced(ctx, "edit_image", func(ctx context.Context) (*GenerationResponse, error) {
		return t.inner.EditImage(ctx, prompt, options...)
	})
}

func (t *editingTracingClient) GenerateImageStreaming(
	ctx context.Context,
	prompt string,
	callback StreamCallback,
	options ...GenerationOption,
) error {
	return t.tracedStream(ctx, "generate_image", func(ctx context.Context) error {
		return t.inner.GenerateImageStreaming(ctx, prompt, callback, options...)
	})
}

func (t *editingTracingClient) EditImageStreaming(
	ctx context.Context,
	prompt string,
	callback StreamCallback,
	options ...GenerationOption,
) error {
	return t.tracedStream(ctx, "edit_image", func(ctx context.Context) error {
		return t.inner.EditImageStreaming(ctx, prompt, callback, options...)
	})
}

// traced wraps a non-streaming call with a span + metrics recording the prompt
// token usage and result count on success.
func (t *editingTracingClient) traced(
	ctx context.Context,
	op string,
	call func(context.Context) (*GenerationResponse, error),
) (*GenerationResponse, error) {
	m := t.inner.Model()
	start := time.Now()
	ctx, span := tracing.StartImageSpan(ctx, m.APIModel, string(m.Provider))
	defer span.End()

	resp, err := call(ctx)
	if err != nil {
		tracing.SetError(span, err)
		tracing.RecordMetrics(ctx, op, m.APIModel, string(m.Provider), time.Since(start), 0, 0, err)
		return nil, err
	}

	tracing.SetResponseAttrs(span,
		tracing.AttrUsageInputTokens.Int64(resp.Usage.PromptTokens),
		tracing.AttrResultCount.Int(len(resp.Images)),
	)
	tracing.RecordMetrics(ctx, op, m.APIModel, string(m.Provider), time.Since(start), resp.Usage.PromptTokens, 0, nil)
	return resp, nil
}

// tracedStream wraps a streaming call with a span + metrics.
func (t *editingTracingClient) tracedStream(
	ctx context.Context,
	op string,
	call func(context.Context) error,
) error {
	m := t.inner.Model()
	start := time.Now()
	ctx, span := tracing.StartImageSpan(ctx, m.APIModel, string(m.Provider))
	defer span.End()

	err := call(ctx)
	tracing.RecordMetrics(ctx, op, m.APIModel, string(m.Provider), time.Since(start), 0, 0, err)
	if err != nil {
		tracing.SetError(span, err)
	}
	return err
}
