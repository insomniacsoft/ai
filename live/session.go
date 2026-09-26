package live

import (
	"fmt"
	"net/http"
	"time"

	"github.com/joakimcarlsson/ai/tool"
)

// DefaultBaseURL is the primary WebSocket endpoint for a GPT-Live session.
const DefaultBaseURL = "wss://api.openai.com/v1/live/sessions"

// SampleRateHz is the audio sample rate this client uses on both directions
// of the socket. The Live WebSocket accepts 16000 or 24000 Hz PCM16; this
// package always requests 24000, the value the docs call the WebSocket
// default, and sends it explicitly in session.start rather than relying on
// a server-side default that could change.
const SampleRateHz = 24000

// dialTimeout bounds the WebSocket handshake.
const dialTimeout = 15 * time.Second

// eventBufferSize sizes the Events channel. Past this, a stalled consumer is
// a fact worth the sender blocking on rather than a reason to grow the queue
// unboundedly or drop audio on the floor.
const eventBufferSize = 64

// Config configures a Client for one Live session.
type Config struct {
	// APIKey is the bearer token sent as "Authorization: Bearer <key>".
	APIKey string

	// BaseURL is the WebSocket endpoint to dial. Defaults to DefaultBaseURL.
	// Callers only need to set this in tests, to point at a fake server.
	BaseURL string

	// HTTPClient, when set, is used for the WebSocket dial instead of this
	// package's own default *http.Client.
	HTTPClient *http.Client

	// SafetyIdentifier, when non-empty, is sent as the OpenAI-Safety-Identifier
	// header: a stable, privacy-preserving identifier for the end user.
	SafetyIdentifier string

	// Session is the session configuration sent in the first message,
	// session.start.
	Session SessionConfig
}

// Validate checks the minimum shape a Client can use. New calls this.
func (c Config) Validate() error {
	if c.APIKey == "" {
		return fmt.Errorf("live: Config.APIKey is required")
	}
	return c.Session.Validate()
}

// InputMessage is one piece of prior conversation history seeded at startup,
// via SessionConfig.Input. The docs accept at most 128 messages and 8,192
// combined tokens across the whole list; this package does not enforce that
// count locally, since it has no tokenizer to check the second half of it.
type InputMessage struct {
	// Role is "user", "developer", or "assistant". A message from any other
	// role is rendered as if it were "user".
	Role string
	// Text is the message's single text part.
	Text string
}

// ResponsesDelegation configures the Responses backend GPT-Live delegates
// tasks to. A nil *ResponsesDelegation on SessionConfig selects client
// delegation instead, sent as delegation type "client": the application
// receives each delegated task as EventDelegation and answers it with
// AppendCommentary or AppendThinking.
type ResponsesDelegation struct {
	// Model is the backend Responses model, e.g. "gpt-6-luna". Required.
	Model string
	// Instructions is the backend's own prompt, separate from the Live
	// session's conversational Instructions.
	Instructions string
	// Tools are the function tools the APPLICATION runs. GPT-Live never
	// executes these itself; it delegates a call and waits for
	// SendFunctionResult.
	Tools []tool.BaseTool
	// ToolChoice is "auto", "none", or "required". Empty omits the field,
	// which the API treats as "auto".
	ToolChoice string
	// ParallelToolCalls, when non-nil, is sent explicitly. The docs recommend
	// false to start.
	ParallelToolCalls *bool
	// ReasoningEffort is the backend model's reasoning effort, e.g. "low" or
	// "high". Empty omits the field.
	ReasoningEffort string
}

// SessionConfig is the session GPT-Live opens on session.start.
type SessionConfig struct {
	// Model is the Live model id, e.g. "gpt-live-1". Required.
	Model string
	// Instructions is the conversational prompt: voice, persona, and when to
	// delegate. Backend business rules belong in Delegation.Instructions
	// instead. Empty leaves the provider's own default instructions in
	// place, which the docs say is not a neutral persona.
	Instructions string
	// Voice selects the output voice. Empty leaves the provider's default
	// ("marin"). Immutable after startup.
	Voice string
	// Input seeds prior conversation history. Defaults to none.
	Input []InputMessage
	// Delegation selects the Responses backend. Nil selects client
	// delegation.
	Delegation *ResponsesDelegation
	// Store, when true, makes the session available for forking and
	// recording download after it ends.
	Store bool
}

// Validate rejects a config that cannot open a usable session.
func (s SessionConfig) Validate() error {
	if s.Model == "" {
		return fmt.Errorf("live: SessionConfig.Model is required")
	}
	if s.Delegation != nil && s.Delegation.Model == "" {
		return fmt.Errorf(
			"live: SessionConfig.Delegation.Model is required when Delegation is set",
		)
	}
	return nil
}

// wireAudioFormat is the first of the wire* types below, which together are
// the JSON shape of the session.start client event, hand-rolled from the
// API reference rather than taken from an SDK: this package deliberately
// does not depend on one, so that it stays proposable as a standalone
// module. Field names and nesting are copied from the reference
// documentation's session.started echo, which mirrors what session.start
// accepts.
type wireAudioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate"`
}

type wireAudioOutput struct {
	Voice string `json:"voice,omitempty"`
}

type wireAudio struct {
	Format wireAudioFormat  `json:"format"`
	Output *wireAudioOutput `json:"output,omitempty"`
}

type wireFunctionTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type wireReasoning struct {
	Effort string `json:"effort,omitempty"`
}

type wireResponsesDelegation struct {
	Model             string             `json:"model"`
	Instructions      string             `json:"instructions,omitempty"`
	Tools             []wireFunctionTool `json:"tools,omitempty"`
	ToolChoice        string             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool              `json:"parallel_tool_calls,omitempty"`
	Reasoning         *wireReasoning     `json:"reasoning,omitempty"`
}

type wireDelegation struct {
	Type      string                   `json:"type"`
	Responses *wireResponsesDelegation `json:"responses,omitempty"`
}

type wireContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type wireInputMessage struct {
	Type    string            `json:"type"`
	Role    string            `json:"role"`
	Content []wireContentPart `json:"content"`
}

type wireSession struct {
	Model        string             `json:"model"`
	Instructions string             `json:"instructions,omitempty"`
	Input        []wireInputMessage `json:"input,omitempty"`
	Audio        wireAudio          `json:"audio"`
	Delegation   *wireDelegation    `json:"delegation,omitempty"`
	Store        bool               `json:"store,omitempty"`
}

type sessionStartEvent struct {
	Type    string      `json:"type"`
	EventID string      `json:"event_id"`
	Session wireSession `json:"session"`
}

// wireInputMessages renders seed history. Developer and user messages use
// "input_text"; assistant messages use "output_text" — the docs' own
// example uses this pairing, and using "input_text" for an assistant message
// describes a different, and wrong, content type.
func wireInputMessages(msgs []InputMessage) []wireInputMessage {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]wireInputMessage, 0, len(msgs))
	for _, m := range msgs {
		role := m.Role
		contentType := "input_text"
		switch role {
		case "assistant":
			contentType = "output_text"
		case "developer", "user":
		default:
			role = "user"
		}
		out = append(out, wireInputMessage{
			Type: "message",
			Role: role,
			Content: []wireContentPart{
				{Type: contentType, Text: m.Text},
			},
		})
	}
	return out
}

// liveTools converts the application's function tools into the delegated
// backend's tool schema.
//
// tool.Info splits a JSON Schema in two: Parameters holds the PROPERTIES map
// alone, and Required is a sibling slice. The backend wants one whole schema
// object, so this reassembles it. A tool with no properties still gets an
// explicit empty object — omitting "properties" entirely describes a
// different (unconstrained) schema, inviting the model to invent arguments
// for a tool that takes none.
func liveTools(tools []tool.BaseTool) ([]wireFunctionTool, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	out := make([]wireFunctionTool, 0, len(tools))
	seen := make(map[string]bool, len(tools))
	for _, t := range tools {
		info := t.Info()
		if info.Name == "" {
			return nil, fmt.Errorf(
				"live: a tool has no name; the backend cannot address it",
			)
		}
		if seen[info.Name] {
			return nil, fmt.Errorf(
				"live: duplicate tool name %q; a call to it could not be routed",
				info.Name,
			)
		}
		seen[info.Name] = true

		props := info.Parameters
		if props == nil {
			props = map[string]any{}
		}
		schema := map[string]any{"type": "object", "properties": props}
		if len(info.Required) > 0 {
			schema["required"] = info.Required
		}
		out = append(out, wireFunctionTool{
			Type:        "function",
			Name:        info.Name,
			Description: info.Description,
			Parameters:  schema,
		})
	}
	return out, nil
}

// buildSessionStart renders cfg.Session into the session.start client event.
func (c *Client) buildSessionStart() (sessionStartEvent, error) {
	sess := c.cfg.Session
	wire := wireSession{
		Model:        sess.Model,
		Instructions: sess.Instructions,
		Input:        wireInputMessages(sess.Input),
		Audio: wireAudio{
			Format: wireAudioFormat{Type: "audio/pcm", Rate: SampleRateHz},
		},
		Store: sess.Store,
	}
	if sess.Voice != "" {
		wire.Audio.Output = &wireAudioOutput{Voice: sess.Voice}
	}
	if sess.Delegation == nil {
		wire.Delegation = &wireDelegation{Type: "client"}
	} else {
		tools, err := liveTools(sess.Delegation.Tools)
		if err != nil {
			return sessionStartEvent{}, err
		}
		responses := &wireResponsesDelegation{
			Model:             sess.Delegation.Model,
			Instructions:      sess.Delegation.Instructions,
			Tools:             tools,
			ToolChoice:        sess.Delegation.ToolChoice,
			ParallelToolCalls: sess.Delegation.ParallelToolCalls,
		}
		if sess.Delegation.ReasoningEffort != "" {
			responses.Reasoning = &wireReasoning{
				Effort: sess.Delegation.ReasoningEffort,
			}
		}
		wire.Delegation = &wireDelegation{
			Type:      "responses",
			Responses: responses,
		}
	}
	return sessionStartEvent{
		Type:    "session.start",
		EventID: c.nextEventID(),
		Session: wire,
	}, nil
}
