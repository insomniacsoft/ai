package realtime

import "encoding/json"

// Event names, as MEASURED against gpt-realtime on 2026-08-23 rather than
// read from documentation. The Realtime API renamed several of these between
// the preview and GA surfaces (`response.audio.delta` became
// `response.output_audio.delta`, and so on), and a client that listens for a
// preview name against a GA session hears nothing at all — no error, no
// warning, just a session that never seems to say anything. That failure is
// indistinguishable from a quiet model, which is why these are pinned here as
// one named set of constants instead of being spelled inline at each switch
// arm.
const (
	// Session lifecycle.
	evSessionCreated = "session.created"
	evSessionUpdated = "session.updated"

	// Conversation items. `.truncated` is the acknowledgement of our own
	// truncate; `.retrieved` answers conversation.item.retrieve.
	//
	//lint:ignore U1000 these four complete the named set this block exists
	// to be (see the comment above the const block); no switch arm needs
	// them today, but the point is that none may ever spell these event
	// strings inline.
	evItemAdded     = "conversation.item.added"
	evItemDone      = "conversation.item.done"
	evItemTruncated = "conversation.item.truncated"
	evItemRetrieved = "conversation.item.retrieved"

	// Responses.
	evResponseCreated  = "response.created"
	evResponseDone     = "response.done"
	evOutputItemAdded  = "response.output_item.added"
	evOutputItemDone   = "response.output_item.done"
	evContentPartAdded = "response.content_part.added"
	evContentPartDone  = "response.content_part.done"

	// Audio out, and the model's own transcript of it.
	evAudioDelta          = "response.output_audio.delta"
	evAudioDone           = "response.output_audio.done"
	evAudioTranscriptDlta = "response.output_audio_transcript.delta"
	evAudioTranscriptDone = "response.output_audio_transcript.done"

	// What the human said, when input transcription is enabled.
	evInputTranscriptDone   = "conversation.item.input_audio_transcription.completed"
	evInputTranscriptFailed = "conversation.item.input_audio_transcription.failed"

	// Tool calls. The complete argument string arrives on `.done`, so nothing
	// needs to accumulate the deltas.
	evFuncArgsDone = "response.function_call_arguments.done"

	// Speech boundaries from the provider's own VAD. speech_started during
	// playback is the barge-in signal.
	evSpeechStarted = "input_audio_buffer.speech_started"
	evSpeechStopped = "input_audio_buffer.speech_stopped"

	evRateLimits = "rate_limits.updated"
	evError      = "error"
)

// Client event names — what this client sends.
const (
	ceSessionUpdate = "session.update"
	ceAudioAppend   = "input_audio_buffer.append"
	ceItemCreate    = "conversation.item.create"
	ceItemTruncate  = "conversation.item.truncate"
	ceItemRetrieve  = "conversation.item.retrieve"
	ceResponseCreN  = "response.create"
	ceResponseCancl = "response.cancel"
)

// serverEvent is the envelope every inbound message shares. Fields absent
// from a given event simply stay zero; the type switch in dispatch decides
// which of them mean anything.
type serverEvent struct {
	Type    string `json:"type"`
	EventID string `json:"event_id"`

	// Correlation. ItemID and CallID are DIFFERENT identifiers that travel
	// together on function-call events: ItemID names the conversation item,
	// CallID is what a tool result must be addressed to. Using one where the
	// other belongs correlates a result to the wrong call, and the model is
	// given an answer to a question it did not ask.
	ItemID     string `json:"item_id"`
	CallID     string `json:"call_id"`
	ResponseID string `json:"response_id"`

	Name       string `json:"name"`
	Arguments  string `json:"arguments"`
	Delta      string `json:"delta"`
	Transcript string `json:"transcript"`
	AudioEndMs int64  `json:"audio_end_ms"`

	Session  json.RawMessage `json:"session"`
	Item     json.RawMessage `json:"item"`
	Response json.RawMessage `json:"response"`
	Error    *serverError    `json:"error"`
}

// serverError is the payload of an `error` event, and also the `error` field
// some responses carry.
type serverError struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param"`
	EventID string `json:"event_id"`
}

func (e *serverError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Code != "" {
		return "realtime: " + e.Code + ": " + e.Message
	}
	return "realtime: " + e.Message
}

// responseEnvelope is the part of a response.done payload this client reads.
type responseEnvelope struct {
	ID            string `json:"id"`
	Status        string `json:"status"`
	StatusDetails struct {
		Type   string       `json:"type"`
		Reason string       `json:"reason"`
		Error  *serverError `json:"error"`
	} `json:"status_details"`
	Output []responseItem `json:"output"`
	Usage  *Usage         `json:"usage"`
}

// responseItem is one output item inside a response.
type responseItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Status    string `json:"status"`
	Role      string `json:"role"`
	Name      string `json:"name"`
	CallID    string `json:"call_id"`
	Arguments string `json:"arguments"`
	Content   []struct {
		Type       string `json:"type"`
		Text       string `json:"text"`
		Transcript string `json:"transcript"`
	} `json:"content"`
}

// Usage is the token accounting a response reports. Voice sessions are billed
// per token like everything else, and this is the raw material cost.go's
// Billable turns into an itemized bill.
type Usage struct {
	TotalTokens  int64 `json:"total_tokens"`
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`

	InputDetails  InputTokenDetails  `json:"input_token_details"`
	OutputDetails OutputTokenDetails `json:"output_token_details"`
}

// InputTokenDetails splits input tokens by modality and by cache hit.
//
// # The subset rule
//
// Measured against the live API on 2026-08-23:
//
//	input_tokens: 123
//	input_token_details: {text_tokens: 123, audio_tokens: 0, cached_tokens: 64,
//	                      cached_tokens_details: {text_tokens: 64}}
//
// CachedTokens is a SUBSET of InputTokens, and TextTokens already INCLUDES the
// cached ones. So the number billed at the uncached rate is the difference, not
// the sum — adding them instead would bill 187 tokens where 123 were used, and
// at an 10x rate gap between cached and uncached text that error is not small.
// Billable() below is the only place that arithmetic is written down.
type InputTokenDetails struct {
	TextTokens  int64 `json:"text_tokens"`
	AudioTokens int64 `json:"audio_tokens"`
	ImageTokens int64 `json:"image_tokens"`

	CachedTokens  int64 `json:"cached_tokens"`
	CachedDetails struct {
		TextTokens  int64 `json:"text_tokens"`
		AudioTokens int64 `json:"audio_tokens"`
		ImageTokens int64 `json:"image_tokens"`
	} `json:"cached_tokens_details"`
}

// OutputTokenDetails splits output tokens by modality. There is no cache
// dimension: output is generated, never replayed.
type OutputTokenDetails struct {
	TextTokens  int64 `json:"text_tokens"`
	AudioTokens int64 `json:"audio_tokens"`

	// ReasoningTokens is a SUBSET of TextTokens, not an addition to it —
	// measured 2026-08-23 on gpt-realtime-2.1, where text 176 + audio 343
	// equalled the reported output_tokens of 519 while reasoning was 42.
	// Billing it separately would double-charge it.
	//
	// Recorded anyway because it is the one class the caller pays for and
	// never hears: on the same utterance gpt-realtime-2 spent 148 reasoning
	// tokens against gpt-realtime-2.1's 42, which is a real cost difference
	// invisible in any other number here.
	ReasoningTokens int64 `json:"reasoning_tokens"`
}

// sessionEnvelope is the part of session.created / session.updated this
// client reads back.
type sessionEnvelope struct {
	ID        string `json:"id"`
	Model     string `json:"model"`
	ExpiresAt int64  `json:"expires_at"`
}
