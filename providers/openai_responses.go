package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/joakimcarlsson/ai/message"
	"github.com/joakimcarlsson/ai/schema"
	"github.com/joakimcarlsson/ai/tool"
	"github.com/joakimcarlsson/ai/types"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// newResponseParams returns a ResponseNewParams with security defaults.
// SECURITY: Store is always false — never store tenant data at OpenAI.
// All Responses API calls MUST use this constructor.
func (o *openaiClient) newResponseParams() responses.ResponseNewParams {
	return responses.ResponseNewParams{
		Store: param.NewOpt(false),
	}
}

// convertResponseMessages converts library messages to Responses API input items.
// System messages are extracted to the instructions string (not as input items).
func (o *openaiClient) convertResponseMessages(
	messages []message.Message,
) (responses.ResponseNewParamsInputUnion, string) {
	items := make([]responses.ResponseInputItemUnionParam, 0, len(messages)*2)
	var systemParts []string

	for _, msg := range messages {
		switch msg.Role {
		case message.System:
			systemParts = append(systemParts, msg.Content().String())

		case message.User:
			items = append(items,
				responses.ResponseInputItemParamOfMessage(msg.Content().String(), "user"),
			)

		case message.Assistant:
			// Text content as a separate message item
			if text := msg.Content().String(); text != "" {
				items = append(items,
					responses.ResponseInputItemParamOfMessage(text, "assistant"),
				)
			}
			// Each tool call as a separate function_call input item
			for _, tc := range msg.ToolCalls() {
				items = append(items,
					responses.ResponseInputItemParamOfFunctionCall(tc.Input, tc.ID, tc.Name),
				)
			}

		case message.Tool:
			for _, tr := range msg.ToolResults() {
				items = append(items,
					responses.ResponseInputItemParamOfFunctionCallOutput(tr.ToolCallID, tr.Content),
				)
			}

		case message.Summary:
			// Summaries are stored as User role when sending to LLM
			items = append(items,
				responses.ResponseInputItemParamOfMessage(msg.Content().String(), "user"),
			)
		}
	}

	input := responses.ResponseNewParamsInputUnion{
		OfInputItemList: responses.ResponseInputParam(items),
	}
	instructions := strings.Join(systemParts, "\n\n")
	return input, instructions
}

// convertResponseTools converts library tools to Responses API flat format.
func (o *openaiClient) convertResponseTools(
	tools []tool.BaseTool,
) []responses.ToolUnionParam {
	result := make([]responses.ToolUnionParam, len(tools))
	for i, t := range tools {
		info := t.Info()
		params := map[string]any{
			"type":       "object",
			"properties": info.Parameters,
		}
		if len(info.Required) > 0 {
			params["required"] = info.Required
		}
		result[i] = responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        info.Name,
				Description: param.NewOpt(info.Description),
				Parameters:  params,
			},
		}
	}
	return result
}

// parseResponseOutput extracts content and tool calls from Responses API output.
// Uses flat struct fields directly (not AsMessage/AsFunctionCall which require JSON raw).
// Does NOT retain references to the Response object — copies values for GC.
func parseResponseOutput(resp *responses.Response) (string, []message.ToolCall) {
	var contentParts []string
	var toolCalls []message.ToolCall

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					contentParts = append(contentParts, part.Text)
				}
			}
		case "function_call":
			toolCalls = append(toolCalls, message.ToolCall{
				ID:       item.CallID,
				Name:     item.Name,
				Input:    item.Arguments,
				Type:     "function",
				Finished: true,
			})
		}
	}

	return strings.Join(contentParts, ""), toolCalls
}

// responseUsage extracts token usage from Responses API format.
func responseUsage(resp *responses.Response) TokenUsage {
	if resp == nil {
		return TokenUsage{}
	}
	cachedTokens := resp.Usage.InputTokensDetails.CachedTokens
	return TokenUsage{
		InputTokens:         resp.Usage.InputTokens - cachedTokens,
		OutputTokens:        resp.Usage.OutputTokens,
		CacheCreationTokens: 0,
		CacheReadTokens:     cachedTokens,
	}
}

// responsesFinishReason maps Responses API status to the library's FinishReason.
func responsesFinishReason(status responses.ResponseStatus, toolCalls []message.ToolCall) message.FinishReason {
	reason := message.FinishReasonEndTurn
	if status == "incomplete" {
		reason = message.FinishReasonMaxTokens
	} else if status == "failed" {
		reason = message.FinishReasonError
	}
	if len(toolCalls) > 0 {
		reason = message.FinishReasonToolUse
	}
	return reason
}

// prepareResponseParams builds the common ResponseNewParams for all Responses API calls.
func (o *openaiClient) prepareResponseParams(
	messages []message.Message,
	tools []tool.BaseTool,
) responses.ResponseNewParams {
	input, instructions := o.convertResponseMessages(messages)
	params := o.newResponseParams()
	params.Model = shared.ResponsesModel(o.providerOptions.model.APIModel)
	params.Input = input
	params.MaxOutputTokens = param.NewOpt(o.providerOptions.maxTokens)

	if instructions != "" {
		params.Instructions = param.NewOpt(instructions)
	}

	if len(tools) > 0 {
		params.Tools = o.convertResponseTools(tools)
	}

	if o.options.parallelToolCalls != nil {
		params.ParallelToolCalls = param.NewOpt(*o.options.parallelToolCalls)
	}

	if o.options.reasoningEffort != nil {
		params.Reasoning = shared.ReasoningParam{
			Effort: shared.ReasoningEffort(*o.options.reasoningEffort),
		}
	}

	paramBuilder := newParameterBuilder(o.providerOptions)
	paramBuilder.applyFloat64Temperature(
		func(t *float64) { params.Temperature = param.NewOpt(*t) },
	)
	paramBuilder.applyFloat64TopP(
		func(p *float64) { params.TopP = param.NewOpt(*p) },
	)

	return params
}

// sendResponses sends a non-streaming request via the Responses API.
func (o *openaiClient) sendResponses(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
) (*Response, error) {
	params := o.prepareResponseParams(messages, tools)

	ctx, cancel := withTimeout(ctx, o.providerOptions.timeout)
	defer cancel()

	return ExecuteWithRetry(
		ctx,
		OpenAIRetryConfig(),
		func() (*Response, error) {
			resp, err := o.client.Responses.New(ctx, params)
			if err != nil {
				return nil, err
			}

			content, toolCalls := parseResponseOutput(resp)

			return &Response{
				Content:      content,
				ToolCalls:    toolCalls,
				Usage:        responseUsage(resp),
				FinishReason: responsesFinishReason(resp.Status, toolCalls),
			}, nil
		},
	)
}

// streamResponses streams a response via the Responses API.
func (o *openaiClient) streamResponses(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
) <-chan Event {
	params := o.prepareResponseParams(messages, tools)
	eventChan := make(chan Event)

	go func() {
		ctx, cancel := withTimeout(ctx, o.providerOptions.timeout)
		defer cancel()
		defer close(eventChan)

		ExecuteStreamWithRetry(ctx, OpenAIRetryConfig(), func() error {
			stream := o.client.Responses.NewStreaming(ctx, params)
			defer stream.Close()

			currentContent := ""
			toolCalls := []message.ToolCall{}
			var finalResponse *responses.Response

			for stream.Next() {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}

				event := stream.Current()
				switch v := event.AsAny().(type) {
				case responses.ResponseTextDeltaEvent:
					currentContent += v.Delta
					eventChan <- Event{
						Type:    types.EventContentDelta,
						Content: v.Delta,
					}

				case responses.ResponseOutputItemAddedEvent:
					if v.Item.Type == "function_call" {
						fc := v.Item.AsFunctionCall()
						eventChan <- Event{
							Type: types.EventToolUseStart,
							ToolCall: &message.ToolCall{
								ID:   fc.CallID,
								Name: fc.Name,
								Type: "function",
							},
						}
					}

				case responses.ResponseFunctionCallArgumentsDeltaEvent:
					eventChan <- Event{
						Type:    types.EventToolUseDelta,
						Content: v.Delta,
					}

				case responses.ResponseOutputItemDoneEvent:
					if v.Item.Type == "function_call" {
						fc := v.Item.AsFunctionCall()
						toolCalls = append(toolCalls, message.ToolCall{
							ID:       fc.CallID,
							Name:     fc.Name,
							Input:    fc.Arguments,
							Type:     "function",
							Finished: true,
						})
						eventChan <- Event{
							Type: types.EventToolUseStop,
							ToolCall: &message.ToolCall{
								ID:       fc.CallID,
								Name:     fc.Name,
								Input:    fc.Arguments,
								Type:     "function",
								Finished: true,
							},
						}
					}

				case responses.ResponseCompletedEvent:
					finalResponse = &v.Response

				case responses.ResponseFailedEvent:
					errMsg := "response failed"
					if v.Response.Error.Message != "" {
						errMsg = v.Response.Error.Message
					}
					eventChan <- Event{
						Type:  types.EventError,
						Error: errors.New(errMsg),
					}
					finalResponse = &v.Response

				case responses.ResponseIncompleteEvent:
					finalResponse = &v.Response

				case responses.ResponseRefusalDeltaEvent:
					currentContent += v.Delta
					eventChan <- Event{
						Type:    types.EventContentDelta,
						Content: v.Delta,
					}

				case responses.ResponseReasoningSummaryTextDeltaEvent:
					eventChan <- Event{
						Type:     types.EventThinkingDelta,
						Thinking: v.Delta,
					}

				default:
					// Explicitly ignore lifecycle, audio, and other events
				}
			}

			if err := stream.Err(); err != nil {
				eventChan <- Event{Type: types.EventError, Error: err}
				return err
			}

			var status responses.ResponseStatus
			var usage TokenUsage
			if finalResponse != nil {
				status = finalResponse.Status
				usage = responseUsage(finalResponse)
			}

			eventChan <- Event{
				Type: types.EventComplete,
				Response: &Response{
					Content:      currentContent,
					ToolCalls:    toolCalls,
					Usage:        usage,
					FinishReason: responsesFinishReason(status, toolCalls),
				},
			}
			return nil
		}, eventChan)
	}()

	return eventChan
}

// sendResponsesWithStructuredOutput sends a non-streaming structured output request.
func (o *openaiClient) sendResponsesWithStructuredOutput(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
	outputSchema *schema.StructuredOutputInfo,
) (*Response, error) {
	params := o.prepareResponseParams(messages, tools)

	schemaMap := map[string]any{
		"type":                 "object",
		"properties":           outputSchema.Parameters,
		"additionalProperties": false,
	}
	if len(outputSchema.Required) > 0 {
		schemaMap["required"] = outputSchema.Required
	}
	params.Text = responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   outputSchema.Name,
				Schema: schemaMap,
				Strict: param.NewOpt(true),
			},
		},
	}

	ctx, cancel := withTimeout(ctx, o.providerOptions.timeout)
	defer cancel()

	return ExecuteWithRetry(
		ctx,
		OpenAIRetryConfig(),
		func() (*Response, error) {
			resp, err := o.client.Responses.New(ctx, params)
			if err != nil {
				return nil, err
			}

			content, toolCalls := parseResponseOutput(resp)

			return &Response{
				Content:                    content,
				ToolCalls:                  toolCalls,
				Usage:                      responseUsage(resp),
				FinishReason:               responsesFinishReason(resp.Status, toolCalls),
				StructuredOutput:           &content,
				UsedNativeStructuredOutput: true,
			}, nil
		},
	)
}

// streamResponsesWithStructuredOutput streams a structured output response.
func (o *openaiClient) streamResponsesWithStructuredOutput(
	ctx context.Context,
	messages []message.Message,
	tools []tool.BaseTool,
	outputSchema *schema.StructuredOutputInfo,
) <-chan Event {
	params := o.prepareResponseParams(messages, tools)

	schemaMap := map[string]any{
		"type":                 "object",
		"properties":           outputSchema.Parameters,
		"additionalProperties": false,
	}
	if len(outputSchema.Required) > 0 {
		schemaMap["required"] = outputSchema.Required
	}
	params.Text = responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   outputSchema.Name,
				Schema: schemaMap,
				Strict: param.NewOpt(true),
			},
		},
	}

	eventChan := make(chan Event)

	go func() {
		ctx, cancel := withTimeout(ctx, o.providerOptions.timeout)
		defer cancel()
		defer close(eventChan)

		ExecuteStreamWithRetry(ctx, OpenAIRetryConfig(), func() error {
			stream := o.client.Responses.NewStreaming(ctx, params)
			defer stream.Close()

			currentContent := ""
			toolCalls := []message.ToolCall{}
			var finalResponse *responses.Response

			for stream.Next() {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}

				event := stream.Current()
				switch v := event.AsAny().(type) {
				case responses.ResponseTextDeltaEvent:
					currentContent += v.Delta
					eventChan <- Event{
						Type:    types.EventContentDelta,
						Content: v.Delta,
					}
				case responses.ResponseOutputItemDoneEvent:
					if v.Item.Type == "function_call" {
						fc := v.Item.AsFunctionCall()
						toolCalls = append(toolCalls, message.ToolCall{
							ID:       fc.CallID,
							Name:     fc.Name,
							Input:    fc.Arguments,
							Type:     "function",
							Finished: true,
						})
					}
				case responses.ResponseCompletedEvent:
					finalResponse = &v.Response
				case responses.ResponseFailedEvent:
					errMsg := fmt.Sprintf("response failed: %s", v.Response.Error.Message)
					eventChan <- Event{Type: types.EventError, Error: errors.New(errMsg)}
					finalResponse = &v.Response
				case responses.ResponseIncompleteEvent:
					finalResponse = &v.Response
				default:
				}
			}

			if err := stream.Err(); err != nil {
				eventChan <- Event{Type: types.EventError, Error: err}
				return err
			}

			var status responses.ResponseStatus
			var usage TokenUsage
			if finalResponse != nil {
				status = finalResponse.Status
				usage = responseUsage(finalResponse)
			}

			eventChan <- Event{
				Type: types.EventComplete,
				Response: &Response{
					Content:                    currentContent,
					ToolCalls:                  toolCalls,
					Usage:                      usage,
					FinishReason:               responsesFinishReason(status, toolCalls),
					StructuredOutput:           &currentContent,
					UsedNativeStructuredOutput: true,
				},
			}
			return nil
		}, eventChan)
	}()

	return eventChan
}
