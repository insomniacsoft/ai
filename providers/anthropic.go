package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/model"
	"github.com/joakimcarlsson/ai/schema"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/types"
)

// AnthropicReasoningEffort controls thinking depth for Anthropic models.
type AnthropicReasoningEffort string

// AnthropicReasoningEffort values.
const (
	AnthropicReasoningEffortLow    AnthropicReasoningEffort = "low"
	AnthropicReasoningEffortMedium AnthropicReasoningEffort = "medium"
	AnthropicReasoningEffortHigh   AnthropicReasoningEffort = "high"
	AnthropicReasoningEffortMax    AnthropicReasoningEffort = "max"
)

// AnthropicCacheTTL controls how long Anthropic retains a cache breakpoint.
// "5m" is the API default; "1h" pays a 2.0× write multiplier but lasts 12×
// longer. For multi-turn agent sessions, "1h" pays for itself by turn 2.
type AnthropicCacheTTL string

const (
	AnthropicCacheTTL5m AnthropicCacheTTL = "5m"
	AnthropicCacheTTL1h AnthropicCacheTTL = "1h"
)

type anthropicOptions struct {
	useBedrock      bool
	disableCache    bool
	reasoningEffort *AnthropicReasoningEffort
	// cacheTTL pins the ephemeral cache TTL on every CacheControl breakpoint
	// emitted by this client. Empty means the API default (5m).
	cacheTTL AnthropicCacheTTL
	// metadataUserID is sent as anthropic.MessageNewParams.Metadata.UserID,
	// a privacy-preserving abuse-detection signal. Caller MUST hash with a
	// per-tenant salt before passing it in to prevent cross-tenant linkage.
	metadataUserID string
}

// AnthropicOption configures optional settings for Anthropic clients.
type AnthropicOption func(*anthropicOptions)

type anthropicClient struct {
	llmOptions llmClientOptions
	options    anthropicOptions
	client     anthropic.Client
}

// AnthropicClient is the Anthropic Client implementation type.
type AnthropicClient Client

func newAnthropicClient(opts llmClientOptions) AnthropicClient {
	anthropicOpts := anthropicOptions{}
	for _, o := range opts.anthropicOptions {
		o(&anthropicOpts)
	}

	anthropicClientOptions := []option.RequestOption{}
	if opts.apiKey != "" {
		anthropicClientOptions = append(
			anthropicClientOptions,
			option.WithAPIKey(opts.apiKey),
		)
	}
	if anthropicOpts.useBedrock {
		anthropicClientOptions = append(
			anthropicClientOptions,
			bedrock.WithLoadDefaultConfig(context.Background()),
		)
	}

	client := anthropic.NewClient(anthropicClientOptions...)
	return &anthropicClient{
		llmOptions: opts,
		options:    anthropicOpts,
		client:     client,
	}
}

func (a *anthropicClient) convertMessages(
	messages []message.Message,
) (anthropicMessages []anthropic.MessageParam, systemMessages []string, systemCacheBreakpoints []bool) {
	for i, msg := range messages {
		cache := false
		if i == len(messages)-1 && !a.options.disableCache {
			cache = true
		}
		switch msg.Role {
		case message.System:
			systemMessages = append(systemMessages, msg.Content().String())
			// Track explicit per-block cache hints so preparedMessages can
			// emit cache_control on the precise blocks the assembler
			// chose, instead of only the auto last-block default.
			breakpoint := false
			for _, part := range msg.Parts {
				if tc, ok := part.(message.TextContent); ok && tc.CacheBreakpoint {
					breakpoint = true
					break
				}
			}
			systemCacheBreakpoints = append(systemCacheBreakpoints, breakpoint)
		case message.User:
			content := anthropic.NewTextBlock(msg.Content().String())
			if cache {
				content.OfText.CacheControl = anthropic.CacheControlEphemeralParam{
					Type: "ephemeral",
					TTL:  a.cacheTTLValue(),
				}
			}
			var contentBlocks []anthropic.ContentBlockParamUnion
			contentBlocks = append(contentBlocks, content)

			for _, binaryContent := range msg.BinaryContent() {
				base64Image := binaryContent.String(model.ProviderAnthropic)
				imageBlock := anthropic.NewImageBlockBase64(
					binaryContent.MIMEType,
					base64Image,
				)
				contentBlocks = append(contentBlocks, imageBlock)
			}

			for _, imageURLContent := range msg.ImageURLContent() {
				imageBlock := anthropic.NewImageBlock(
					anthropic.URLImageSourceParam{
						Type: "url",
						URL:  imageURLContent.URL,
					},
				)
				contentBlocks = append(contentBlocks, imageBlock)
			}

			anthropicMessages = append(
				anthropicMessages,
				anthropic.NewUserMessage(contentBlocks...),
			)

		case message.Assistant:
			blocks := []anthropic.ContentBlockParamUnion{}
			if msg.Content().String() != "" {
				content := anthropic.NewTextBlock(msg.Content().String())
				blocks = append(blocks, content)
			}

			for _, toolCall := range msg.ToolCalls() {
				var inputMap map[string]any
				err := json.Unmarshal([]byte(toolCall.Input), &inputMap)
				if err != nil {
					continue
				}
				blocks = append(
					blocks,
					anthropic.NewToolUseBlock(
						toolCall.ID,
						inputMap,
						toolCall.Name,
					),
				)
			}

			if len(blocks) == 0 {
				slog.Warn(
					"There is a message without content, investigate, this should not happen",
				)
				continue
			}
			anthropicMessages = append(
				anthropicMessages,
				anthropic.NewAssistantMessage(blocks...),
			)

		case message.Tool:
			results := make(
				[]anthropic.ContentBlockParamUnion,
				len(msg.ToolResults()),
			)
			for i, toolResult := range msg.ToolResults() {
				results[i] = anthropic.NewToolResultBlock(
					toolResult.ToolCallID,
					toolResult.Content,
					toolResult.IsError,
				)
			}
			anthropicMessages = append(
				anthropicMessages,
				anthropic.NewUserMessage(results...),
			)
		}
	}
	return
}

func (a *anthropicClient) convertTools(
	tools []tool.BaseTool,
) []anthropic.ToolUnionParam {
	anthropicTools := make([]anthropic.ToolUnionParam, len(tools))

	for i, tool := range tools {
		info := tool.Info()
		toolParam := anthropic.ToolParam{
			Name:        info.Name,
			Description: anthropic.String(info.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: info.Parameters,
			},
		}

		if i == len(tools)-1 && !a.options.disableCache {
			toolParam.CacheControl = anthropic.CacheControlEphemeralParam{
				Type: "ephemeral",
				TTL:  a.cacheTTLValue(),
			}
		}

		anthropicTools[i] = anthropic.ToolUnionParam{OfTool: &toolParam}
	}

	return anthropicTools
}

func (a *anthropicClient) finishReason(reason string) message.FinishReason {
	switch reason {
	case "end_turn":
		return message.FinishReasonEndTurn
	case "max_tokens":
		return message.FinishReasonMaxTokens
	case "tool_use":
		return message.FinishReasonToolUse
	case "stop_sequence":
		return message.FinishReasonEndTurn
	default:
		return message.FinishReasonUnknown
	}
}

func (a *anthropicClient) preparedMessages(
	messages []anthropic.MessageParam,
	tools []anthropic.ToolUnionParam,
	systemMessages []string,
	systemCacheBreakpoints []bool,
) anthropic.MessageNewParams {
	var thinkingParam anthropic.ThinkingConfigParamUnion
	var outputConfig anthropic.OutputConfigParam
	temperature := anthropic.Float(0)
	paramBuilder := newParameterBuilder(a.llmOptions)
	paramBuilder.applyFloat64Temperature(
		func(t *float64) { temperature = anthropic.Float(*t) },
	)

	if a.options.reasoningEffort != nil && a.llmOptions.model.CanReason {
		temperature = anthropic.Float(1)
		apiModel := a.llmOptions.model.APIModel
		if strings.Contains(apiModel, "4-6") || strings.Contains(apiModel, "4.6") {
			thinkingParam = anthropic.ThinkingConfigParamUnion{
				OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
			}
			switch *a.options.reasoningEffort {
			case AnthropicReasoningEffortLow:
				outputConfig.Effort = anthropic.OutputConfigEffortLow
			case AnthropicReasoningEffortMedium:
				outputConfig.Effort = anthropic.OutputConfigEffortMedium
			case AnthropicReasoningEffortHigh:
				outputConfig.Effort = anthropic.OutputConfigEffortHigh
			case AnthropicReasoningEffortMax:
				outputConfig.Effort = anthropic.OutputConfigEffortMax
			}
		} else {
			thinkingParam = anthropic.ThinkingConfigParamUnion{
				OfEnabled: &anthropic.ThinkingConfigEnabledParam{
					BudgetTokens: int64(float64(a.llmOptions.maxTokens) * 0.8),
				},
			}
		}
	}

	if a.llmOptions.maxTokens == 0 {
		a.llmOptions.maxTokens = a.llmOptions.model.DefaultMaxTokens
	} else {
		a.llmOptions.maxTokens = int64(a.llmOptions.maxTokens)
	}

	params := anthropic.MessageNewParams{
		Model:        anthropic.Model(a.llmOptions.model.APIModel),
		MaxTokens:    a.llmOptions.maxTokens,
		Temperature:  temperature,
		Messages:     messages,
		Tools:        tools,
		Thinking:     thinkingParam,
		OutputConfig: outputConfig,
	}

	paramBuilder.applyFloat64TopP(
		func(p *float64) { params.TopP = anthropic.Float(*p) },
	)
	paramBuilder.applyInt64TopK(
		func(k *int64) { params.TopK = anthropic.Int(*k) },
	)

	if len(a.llmOptions.stopSequences) > 0 {
		params.StopSequences = a.llmOptions.stopSequences
	}

	if a.options.metadataUserID != "" {
		params.Metadata = anthropic.MetadataParam{
			UserID: anthropic.String(a.options.metadataUserID),
		}
	}

	if len(systemMessages) > 0 {
		// Two cache-control modes:
		//
		// 1. Implicit (default, backwards-compatible): no caller marked
		//    a system block with CacheBreakpoint=true → emit one
		//    breakpoint on the very last system block. Keeps existing
		//    callers working unchanged.
		//
		// 2. Explicit: at least one system block carries
		//    CacheBreakpoint=true (set via
		//    message.NewSystemMessageWithCacheBreakpoint) → emit
		//    breakpoints exactly on those blocks. The auto last-block
		//    breakpoint is suppressed so the caller controls the full
		//    cache budget. Anthropic permits up to 4 breakpoints per
		//    request total (system + tools + messages combined); the
		//    caller is responsible for staying within the budget.
		hasExplicitBreakpoints := false
		for _, bp := range systemCacheBreakpoints {
			if bp {
				hasExplicitBreakpoints = true
				break
			}
		}

		systemBlocks := make([]anthropic.TextBlockParam, len(systemMessages))
		for i, sysMsg := range systemMessages {
			block := anthropic.TextBlockParam{
				Text: sysMsg,
			}
			cacheThis := false
			if hasExplicitBreakpoints {
				cacheThis = i < len(systemCacheBreakpoints) && systemCacheBreakpoints[i]
			} else if i == len(systemMessages)-1 {
				cacheThis = true
			}
			if cacheThis && !a.options.disableCache {
				block.CacheControl = anthropic.CacheControlEphemeralParam{
					Type: "ephemeral",
					TTL:  a.cacheTTLValue(),
				}
			}
			systemBlocks[i] = block
		}
		params.System = systemBlocks
	}

	return params
}

func (a *anthropicClient) send(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
) (resposne *Response, err error) {
	anthropicMessages, systemMessages, systemCacheBreakpoints := a.convertMessages(messages)
	preparedMessages := a.preparedMessages(
		anthropicMessages,
		a.convertTools(tools),
		systemMessages,
		systemCacheBreakpoints,
	)

	ctx, cancel := withTimeout(ctx, a.llmOptions.timeout)
	defer cancel()

	return ExecuteWithRetry(
		ctx,
		AnthropicRetryConfig(),
		func() (*Response, error) {
			anthropicResponse, err := a.client.Messages.New(
				ctx,
				preparedMessages,
			)
			if err != nil {
				return nil, err
			}

			content := ""
			for _, block := range anthropicResponse.Content {
				if text, ok := block.AsAny().(anthropic.TextBlock); ok {
					content += text.Text
				}
			}

			return &Response{
				Content:   content,
				ToolCalls: a.toolCalls(*anthropicResponse),
				Usage:     a.usage(*anthropicResponse),
				FinishReason: a.finishReason(
					string(anthropicResponse.StopReason),
				),
			}, nil
		},
	)
}

func (a *anthropicClient) stream(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
) <-chan Event {
	anthropicMessages, systemMessages, systemCacheBreakpoints := a.convertMessages(messages)
	preparedMessages := a.preparedMessages(
		anthropicMessages,
		a.convertTools(tools),
		systemMessages,
		systemCacheBreakpoints,
	)
	eventChan := make(chan Event)

	ctx, cancel := withTimeout(ctx, a.llmOptions.timeout)
	defer cancel()

	go func() {
		defer close(eventChan)

		ExecuteStreamWithRetry(ctx, AnthropicRetryConfig(), func() error {
			anthropicStream := a.client.Messages.NewStreaming(
				ctx,
				preparedMessages,
			)
			accumulatedMessage := anthropic.Message{}

			currentToolCallID := ""
			for anthropicStream.Next() {
				event := anthropicStream.Current()
				err := accumulatedMessage.Accumulate(event)
				if err != nil {
					slog.Warn("Error accumulating message", "error", err)
					continue
				}

				switch event := event.AsAny().(type) {
				case anthropic.ContentBlockStartEvent:
					switch event.ContentBlock.Type {
					case "text":
						eventChan <- Event{Type: types.EventContentStart}
					case "tool_use":
						currentToolCallID = event.ContentBlock.ID
						eventChan <- Event{
							Type: types.EventToolUseStart,
							ToolCall: &message.ToolCall{
								ID:       event.ContentBlock.ID,
								Name:     event.ContentBlock.Name,
								Finished: false,
							},
						}
					}

				case anthropic.ContentBlockDeltaEvent:
					switch event.Delta.Type {
					case "thinking_delta":
						if event.Delta.Thinking != "" {
							eventChan <- Event{
								Type:     types.EventThinkingDelta,
								Thinking: event.Delta.Thinking,
							}
						}
					case "text_delta":
						if event.Delta.Text != "" {
							eventChan <- Event{
								Type:    types.EventContentDelta,
								Content: event.Delta.Text,
							}
						}
					case "input_json_delta":
						if currentToolCallID != "" {
							eventChan <- Event{
								Type: types.EventToolUseDelta,
								ToolCall: &message.ToolCall{
									ID:       currentToolCallID,
									Finished: false,
									Input:    event.Delta.JSON.PartialJSON.Raw(),
								},
							}
						}
					}
				case anthropic.ContentBlockStopEvent:
					if currentToolCallID != "" {
						eventChan <- Event{
							Type: types.EventToolUseStop,
							ToolCall: &message.ToolCall{
								ID: currentToolCallID,
							},
						}
						currentToolCallID = ""
					} else {
						eventChan <- Event{Type: types.EventContentStop}
					}

				case anthropic.MessageStopEvent:
					content := ""
					for _, block := range accumulatedMessage.Content {
						if text, ok := block.AsAny().(anthropic.TextBlock); ok {
							content += text.Text
						}
					}

					eventChan <- Event{
						Type: types.EventComplete,
						Response: &Response{
							Content:      content,
							ToolCalls:    a.toolCalls(accumulatedMessage),
							Usage:        a.usage(accumulatedMessage),
							FinishReason: a.finishReason(string(accumulatedMessage.StopReason)),
						},
					}
				}
			}

			err := anthropicStream.Err()
			if err == nil || errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}, eventChan)
	}()
	return eventChan
}

func (a *anthropicClient) toolCalls(msg anthropic.Message) []message.ToolCall {
	var toolCalls []message.ToolCall

	for _, block := range msg.Content {
		if variant, ok := block.AsAny().(anthropic.ToolUseBlock); ok {
			toolCall := message.ToolCall{
				ID:       variant.ID,
				Name:     variant.Name,
				Input:    string(variant.Input),
				Type:     string(variant.Type),
				Finished: true,
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}

	return toolCalls
}

func (a *anthropicClient) usage(msg anthropic.Message) TokenUsage {
	return TokenUsage{
		InputTokens:           msg.Usage.InputTokens,
		OutputTokens:          msg.Usage.OutputTokens,
		CacheCreationTokens:   msg.Usage.CacheCreationInputTokens,
		CacheReadTokens:       msg.Usage.CacheReadInputTokens,
		CacheCreation5mTokens: msg.Usage.CacheCreation.Ephemeral5mInputTokens,
		CacheCreation1hTokens: msg.Usage.CacheCreation.Ephemeral1hInputTokens,
	}
}

// cacheTTLValue returns the SDK TTL constant matching the configured TTL option,
// or empty string when unset (which the SDK treats as the 5m API default).
func (a *anthropicClient) cacheTTLValue() anthropic.CacheControlEphemeralTTL {
	switch a.options.cacheTTL {
	case AnthropicCacheTTL1h:
		return anthropic.CacheControlEphemeralTTLTTL1h
	case AnthropicCacheTTL5m:
		return anthropic.CacheControlEphemeralTTLTTL5m
	default:
		return ""
	}
}

// WithAnthropicBedrock configures whether to use AWS Bedrock for Anthropic models
func WithAnthropicBedrock(useBedrock bool) AnthropicOption {
	return func(options *anthropicOptions) {
		options.useBedrock = useBedrock
	}
}

// WithAnthropicDisableCache disables response caching for Anthropic requests
func WithAnthropicDisableCache() AnthropicOption {
	return func(options *anthropicOptions) {
		options.disableCache = true
	}
}

// WithAnthropicReasoningEffort sets the reasoning effort level for Anthropic models.
func WithAnthropicReasoningEffort(effort AnthropicReasoningEffort) AnthropicOption {
	return func(options *anthropicOptions) {
		options.reasoningEffort = &effort
	}
}

// WithAnthropicCacheTTL pins the ephemeral cache TTL ("5m" or "1h") on every
// CacheControl breakpoint emitted by this client. Unset uses the API default.
// Pair with the per-tier write rates exposed via the Anthropic billing API
// (5m=1.25× / 1h=2.0× of base input cost).
func WithAnthropicCacheTTL(ttl AnthropicCacheTTL) AnthropicOption {
	return func(options *anthropicOptions) {
		options.cacheTTL = ttl
	}
}

// WithAnthropicMetadataUserID sets the per-request `metadata.user_id` field.
// Anthropic uses this for abuse signals only (no PII contract). Callers MUST
// hash with a per-tenant salt before passing it in to prevent cross-tenant
// linkage of the same user across multiple Anthropic API customers.
func WithAnthropicMetadataUserID(uid string) AnthropicOption {
	return func(options *anthropicOptions) {
		options.metadataUserID = uid
	}
}

func (a *anthropicClient) supportsStructuredOutput() bool {
	return a.llmOptions.model.SupportsStructuredOut
}

func (a *anthropicClient) buildOutputConfig(
	outputSchema *schema.StructuredOutputInfo,
) anthropic.OutputConfigParam {
	schemaMap := map[string]any{
		"type":                 "object",
		"properties":           outputSchema.Parameters,
		"additionalProperties": false,
	}
	if len(outputSchema.Required) > 0 {
		schemaMap["required"] = outputSchema.Required
	}
	return anthropic.OutputConfigParam{
		Format: anthropic.JSONOutputFormatParam{
			Schema: schemaMap,
		},
	}
}

func (a *anthropicClient) sendWithStructuredOutput(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
	outputSchema *schema.StructuredOutputInfo,
) (*Response, error) {
	anthropicMessages, systemMessages, systemCacheBreakpoints := a.convertMessages(messages)
	preparedMessages := a.preparedMessages(
		anthropicMessages,
		a.convertTools(tools),
		systemMessages,
		systemCacheBreakpoints,
	)
	preparedMessages.OutputConfig = a.buildOutputConfig(outputSchema)

	ctx, cancel := withTimeout(ctx, a.llmOptions.timeout)
	defer cancel()

	return ExecuteWithRetry(
		ctx,
		AnthropicRetryConfig(),
		func() (*Response, error) {
			anthropicResponse, err := a.client.Messages.New(
				ctx,
				preparedMessages,
			)
			if err != nil {
				return nil, err
			}

			content := ""
			for _, block := range anthropicResponse.Content {
				if text, ok := block.AsAny().(anthropic.TextBlock); ok {
					content += text.Text
				}
			}

			return &Response{
				Content:   content,
				ToolCalls: a.toolCalls(*anthropicResponse),
				Usage:     a.usage(*anthropicResponse),
				FinishReason: a.finishReason(
					string(anthropicResponse.StopReason),
				),
				StructuredOutput:           &content,
				UsedNativeStructuredOutput: true,
			}, nil
		},
	)
}

func (a *anthropicClient) streamWithStructuredOutput(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
	outputSchema *schema.StructuredOutputInfo,
) <-chan Event {
	anthropicMessages, systemMessages, systemCacheBreakpoints := a.convertMessages(messages)
	preparedMessages := a.preparedMessages(
		anthropicMessages,
		a.convertTools(tools),
		systemMessages,
		systemCacheBreakpoints,
	)
	preparedMessages.OutputConfig = a.buildOutputConfig(outputSchema)

	eventChan := make(chan Event)

	ctx, cancel := withTimeout(ctx, a.llmOptions.timeout)
	defer cancel()

	go func() {
		defer close(eventChan)

		ExecuteStreamWithRetry(ctx, AnthropicRetryConfig(), func() error {
			anthropicStream := a.client.Messages.NewStreaming(
				ctx,
				preparedMessages,
			)
			accumulatedMessage := anthropic.Message{}

			currentToolCallID := ""
			for anthropicStream.Next() {
				event := anthropicStream.Current()
				err := accumulatedMessage.Accumulate(event)
				if err != nil {
					slog.Warn("Error accumulating message", "error", err)
					continue
				}

				switch event := event.AsAny().(type) {
				case anthropic.ContentBlockStartEvent:
					switch event.ContentBlock.Type {
					case "text":
						eventChan <- Event{Type: types.EventContentStart}
					case "tool_use":
						currentToolCallID = event.ContentBlock.ID
						eventChan <- Event{
							Type: types.EventToolUseStart,
							ToolCall: &message.ToolCall{
								ID:       event.ContentBlock.ID,
								Name:     event.ContentBlock.Name,
								Finished: false,
							},
						}
					}

				case anthropic.ContentBlockDeltaEvent:
					switch event.Delta.Type {
					case "thinking_delta":
						if event.Delta.Thinking != "" {
							eventChan <- Event{
								Type:     types.EventThinkingDelta,
								Thinking: event.Delta.Thinking,
							}
						}
					case "text_delta":
						if event.Delta.Text != "" {
							eventChan <- Event{
								Type:    types.EventContentDelta,
								Content: event.Delta.Text,
							}
						}
					case "input_json_delta":
						if currentToolCallID != "" {
							eventChan <- Event{
								Type: types.EventToolUseDelta,
								ToolCall: &message.ToolCall{
									ID:       currentToolCallID,
									Finished: false,
									Input:    event.Delta.JSON.PartialJSON.Raw(),
								},
							}
						}
					}
				case anthropic.ContentBlockStopEvent:
					if currentToolCallID != "" {
						eventChan <- Event{
							Type: types.EventToolUseStop,
							ToolCall: &message.ToolCall{
								ID: currentToolCallID,
							},
						}
						currentToolCallID = ""
					} else {
						eventChan <- Event{Type: types.EventContentStop}
					}

				case anthropic.MessageStopEvent:
					content := ""
					for _, block := range accumulatedMessage.Content {
						if text, ok := block.AsAny().(anthropic.TextBlock); ok {
							content += text.Text
						}
					}

					eventChan <- Event{
						Type: types.EventComplete,
						Response: &Response{
							Content:                    content,
							ToolCalls:                  a.toolCalls(accumulatedMessage),
							Usage:                      a.usage(accumulatedMessage),
							FinishReason:               a.finishReason(string(accumulatedMessage.StopReason)),
							StructuredOutput:           &content,
							UsedNativeStructuredOutput: true,
						},
					}
				}
			}

			err := anthropicStream.Err()
			if err == nil || errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}, eventChan)
	}()
	return eventChan
}
