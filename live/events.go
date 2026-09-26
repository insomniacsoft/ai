package live

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Sentinel errors a caller can match with errors.Is.
var (
	// ErrUnauthorized means the API key was refused — a 401/403 on the
	// initial dial, or an authentication error event on an open socket.
	ErrUnauthorized = errors.New("live: authentication rejected")

	// ErrOutOfCredit means the provider refused because the account cannot
	// pay for the session (an insufficient_quota error).
	ErrOutOfCredit = errors.New("live: the account is out of credit")

	// ErrNotReady is returned by a send method called before the socket has
	// been dialed — Start has not been called, or the dial has not finished
	// yet. It does not require session.started: like realtime's sendLive,
	// writes are allowed as soon as a connection exists, because the whole
	// point of a duplex channel is not discarding the caller's first words
	// while the handshake completes.
	ErrNotReady = errors.New("live: not connected yet")

	// ErrClosed is returned by a send method, or by Close, called after the
	// session has permanently ended — closed by the caller, closed by the
	// provider, or lost. A Live session cannot be resumed, so this is
	// always terminal.
	ErrClosed = errors.New("live: the client is closed")
)

// EventKind discriminates Event.
type EventKind int

// The kinds of things worth telling a caller about.
const (
	// EventSessionStarted fires when session.started arrives: the provider
	// accepted the session configuration and assigned a session ID.
	EventSessionStarted EventKind = iota
	// EventAudio carries decoded PCM from session.output_audio.delta:
	// 24 kHz mono s16 little-endian, in Event.Audio.
	EventAudio
	// EventInputTranscript carries a fragment of what the human said, from
	// session.input_transcript.delta.
	EventInputTranscript
	// EventOutputTranscript carries a fragment of what the model said, from
	// session.output_transcript.delta.
	EventOutputTranscript
	// EventDelegation fires when the Live model delegates work, from
	// session.delegation.created.
	EventDelegation
	// EventFunctionCall fires when a delegated Responses backend calls a
	// function tool, from a response.event wrapping
	// response.output_item.done with a function_call item.
	EventFunctionCall
	// EventBackendResponseDone fires when a delegated Responses backend
	// finishes a response (completed, failed, cancelled, or incomplete),
	// from a response.event wrapping that Responses completion event.
	EventBackendResponseDone
	// EventUsage carries cumulative voice usage, from session.usage.updated.
	EventUsage
	// EventMuted acknowledges session.input_audio.mute.
	EventMuted
	// EventUnmuted acknowledges session.input_audio.unmute.
	EventUnmuted
	// EventClosed fires once, when session.closed arrives, whether or not
	// the caller is waiting on Close.
	EventClosed
	// EventError reports a provider error event, an undecodable message, or
	// the connection ending. Err is always set; Code is set when the
	// provider's error event carried a machine-readable code.
	EventError
	// EventOther is any other server event, including a response.event
	// whose nested type this package does not otherwise act on. Type names
	// which one, so a caller can log it.
	EventOther
)

// String names an EventKind, for logging.
func (k EventKind) String() string {
	switch k {
	case EventSessionStarted:
		return "session_started"
	case EventAudio:
		return "audio"
	case EventInputTranscript:
		return "input_transcript"
	case EventOutputTranscript:
		return "output_transcript"
	case EventDelegation:
		return "delegation"
	case EventFunctionCall:
		return "function_call"
	case EventBackendResponseDone:
		return "backend_response_done"
	case EventUsage:
		return "usage"
	case EventMuted:
		return "muted"
	case EventUnmuted:
		return "unmuted"
	case EventClosed:
		return "closed"
	case EventError:
		return "error"
	case EventOther:
		return "other"
	}
	return fmt.Sprintf("kind(%d)", int(k))
}

// Delegation is the metadata of one delegated task, from
// session.delegation.created. It carries no task text — only where the work
// went and how to correlate later events with it.
type Delegation struct {
	// ID correlates later events to this delegation: as delegation_id on a
	// response.event, or as the delegationID argument to AppendInstructions,
	// AppendThinking, and AppendCommentary.
	ID string
	// Target is "client" or "responses".
	Target string
	// ResponseID is the Responses API response ID, set only when
	// Target == "responses".
	ResponseID string
	// OffsetMs is where on the session timeline the delegation was created.
	OffsetMs int64
}

// FunctionCall is one function call a delegated Responses backend made.
type FunctionCall struct {
	// DelegationID is the outer response.event's delegation_id.
	DelegationID string
	// ResponseID is the Responses API response this call belongs to, known
	// from the delegation that started it.
	ResponseID string
	// CallID is what SendFunctionResult must address its answer to.
	CallID string
	// Name is the function name.
	Name string
	// Arguments is the model's JSON, byte for byte, unparsed.
	Arguments string
}

// BackendUsage is one delegated Responses backend call's token accounting.
//
// InputTokens is the UNCACHED remainder: the backend's own input_tokens is
// the whole prompt, and its cached_tokens is a SUBSET of it, not an
// addition — so InputTokens here is input_tokens minus cached_tokens,
// clamped at zero, matching this fork's llm.TokenUsage contract (see
// llm/openai/responses.go's usage()). Reporting the raw input_tokens instead
// would double-count every cached token.
type BackendUsage struct {
	InputTokens     int64
	CacheReadTokens int64
	OutputTokens    int64
	// ReasoningTokens is a SUBSET of OutputTokens, not an addition to it.
	ReasoningTokens int64
}

// BackendResponse is one delegated Responses backend response's outcome.
type BackendResponse struct {
	// DelegationID is the outer response.event's delegation_id.
	DelegationID string
	// ResponseID is the Responses API response ID.
	ResponseID string
	// Status is the backend's own verdict: "completed", "failed",
	// "cancelled", or "incomplete".
	Status string
	// Usage is the token accounting, when the backend reported one. Its
	// zero value means none was reported (not that usage was zero).
	Usage BackendUsage
}

// Usage is cumulative voice usage, from session.usage.updated.
type Usage struct {
	// Seconds is the cumulative Live audio duration so far. It is a running
	// total, not a delta: do not sum it across events.
	Seconds float64
	// ContextWindowRatio is the latest active context token count divided
	// by the model's context limit, when the provider reported one.
	ContextWindowRatio float64
	// HasContextWindow reports whether ContextWindowRatio is meaningful:
	// the provider omits context_window when the limit is unknown.
	HasContextWindow bool
}

// Closed is the outcome of a session, from session.closed.
type Closed struct {
	// Seconds is the final cumulative voice duration.
	Seconds float64
	// Reason is why the session ended: "close_requested", "expired",
	// "content", "remote_hangup", or "connection_lost".
	Reason string
}

// Event is one thing worth telling a caller about. Only the fields
// documented for Event.Kind are meaningful; the rest are zero.
type Event struct {
	Kind EventKind

	// Audio is decoded PCM, set on EventAudio.
	Audio []byte

	// Text is a transcript fragment, set on EventInputTranscript and
	// EventOutputTranscript.
	Text string
	// StartMs and EndMs place Text on the session timeline, set alongside
	// Text.
	StartMs int64
	EndMs   int64

	// Delegation is set on EventDelegation.
	Delegation Delegation
	// FunctionCall is set on EventFunctionCall.
	FunctionCall FunctionCall
	// BackendResponse is set on EventBackendResponseDone.
	BackendResponse BackendResponse
	// Usage is set on EventUsage.
	Usage Usage
	// Closed is set on EventClosed.
	Closed Closed

	// Err is set on EventError.
	Err error
	// Code is the provider's machine-readable error code, set on
	// EventError when the provider sent one.
	Code string

	// Type names the server event, set on EventOther: either the outer
	// event's own type, or — for an unrecognised response.event — the
	// nested Responses event's type.
	Type string

	// Raw is the server event exactly as received, for a caller that wants
	// more than this package extracts.
	Raw json.RawMessage
}

// ── the wire shape of server events ────────────────────────────────────

// serverEnvelope is the envelope every inbound message shares. Fields
// absent from a given event simply stay zero; dispatch's switch on Type
// decides which of them mean anything.
type serverEnvelope struct {
	Type          string          `json:"type"`
	EventID       string          `json:"event_id"`
	ClientEventID string          `json:"client_event_id"`
	Session       json.RawMessage `json:"session"`
	Reason        string          `json:"reason"`
	Usage         *wireUsage      `json:"usage"`
	ContextWindow *struct {
		UsageRatio float64 `json:"usage_ratio"`
	} `json:"context_window"`
	Delegation *wireDelegationInfo `json:"delegation"`
	OffsetMs   int64               `json:"offset_ms"`
	Delta      string              `json:"delta"`
	StartMs    int64               `json:"start_ms"`
	EndMs      int64               `json:"end_ms"`
	Error      *wireError          `json:"error"`
	// DelegationID is a pointer because the field's own presence (as
	// opposed to its value) tells response.event's null case apart from a
	// server event of some other type that never carried this field at all
	// — both decode to "", but only the former is a delegation_id the docs
	// promise can be null.
	DelegationID *string         `json:"delegation_id"`
	Event        json.RawMessage `json:"event"`
}

// wireUsage is SessionUsage: cumulative voice duration in seconds.
type wireUsage struct {
	Seconds float64 `json:"seconds"`
}

// wireDelegationInfo is the delegation object on session.delegation.created.
type wireDelegationInfo struct {
	ID         string `json:"id"`
	Target     string `json:"target"`
	ResponseID string `json:"response_id"`
}

// wireError is the payload of an error event.
type wireError struct {
	Type          string `json:"type"`
	Code          string `json:"code"`
	Message       string `json:"message"`
	Param         string `json:"param"`
	ClientEventID string `json:"client_event_id"`
}

// wireResponsesEvent is the part of a response.event's nested Responses
// event this client reads: enough to recognise a function call and a
// finished response's status and usage.
type wireResponsesEvent struct {
	Type     string `json:"type"`
	Response *struct {
		ID     string            `json:"id"`
		Status string            `json:"status"`
		Usage  *wireBackendUsage `json:"usage"`
	} `json:"response"`
	Item *struct {
		Type      string `json:"type"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`
}

// wireBackendUsage is a Responses API usage block.
type wireBackendUsage struct {
	InputTokens        int64 `json:"input_tokens"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokens        int64 `json:"output_tokens"`
	OutputTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}
