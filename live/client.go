// Package live is a client for one OpenAI GPT-Live session over a
// server-side WebSocket: a single speech-to-speech connection that carries
// audio and JSON events both ways, and — when the session delegates to a
// Responses backend — the delegated task's own streamed events wrapped
// inside response.event.
//
// # What this is not
//
// A Live session cannot be resumed. A dropped connection is terminal and
// reported as an error; this package never reconnects. There is no WebRTC,
// no SIP, no sideband connection, and no fork endpoint here — one primary
// WebSocket, one session, for its whole life.
//
// # What it copies, and what it deliberately does not
//
// The realtime package in this repo (github.com/joakimcarlsson/ai/realtime)
// is the pattern this one follows: the same WebSocket library, the same
// shape of Config/Client/Events, the same way of turning a tool.BaseTool
// into a provider function schema. This package does not import realtime —
// it copies the small pieces it needs — because realtime does not exist
// upstream and this module must be proposable on its own. What it does NOT
// copy is realtime's reconnect machinery: a Live session has nothing to
// reconnect to, so there is no connectLoop, no backoff, and no replay.
package live

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/coder/websocket"
)

// Client holds one Live session. It is safe for concurrent use: every send
// method, Close, WaitReady, and SessionID may be called from any goroutine.
type Client struct {
	cfg    Config
	dialer wsDialer

	started atomic.Bool
	cancel  context.CancelFunc
	stop    <-chan struct{} // the run context's Done: closed once Close or the caller's ctx ends the session
	doneCh  chan struct{}

	events chan Event

	eventSeq atomic.Uint64

	mu            sync.Mutex
	conn          wsConn
	connCtx       context.Context
	ready         bool
	ended         bool
	readySignal   chan struct{}
	fatalErr      error
	sessionID     string
	sawClosed     bool
	closedPayload Closed

	writeMu sync.Mutex

	delegMu             sync.Mutex
	delegationResponses map[string]string

	closeOnce           sync.Once
	closeDone           chan struct{}
	closeResult         Closed
	closeErr            error
	sessionClosedSignal chan struct{}
	closedSignalOnce    sync.Once
}

// New builds a Client for one session. It does not connect — call Start.
func New(cfg Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	c := &Client{
		cfg:                 cfg,
		events:              make(chan Event, eventBufferSize),
		doneCh:              make(chan struct{}),
		readySignal:         make(chan struct{}),
		delegationResponses: make(map[string]string),
		closeDone:           make(chan struct{}),
		sessionClosedSignal: make(chan struct{}),
	}
	c.dialer = realDialer{httpClient: cfg.HTTPClient, readLimit: 8 << 20}
	return c, nil
}

// Events is the stream of things worth acting on. It is closed exactly
// once, after the read loop ends for any reason. Keep reading it while Close
// runs: session.closed, and the final usage it carries, arrives on it, and a
// consumer that stopped reading makes Close wait out its deadline instead.
func (c *Client) Events() <-chan Event { return c.events }

// SessionID returns the session ID from session.started, or "" before it
// arrives.
func (c *Client) SessionID() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sessionID
}

// Start dials the session, sends session.start, and starts the read loop in
// the background. Calling it twice is a programming error and panics.
func (c *Client) Start(ctx context.Context) {
	if !c.started.CompareAndSwap(false, true) {
		panic("live: Client.Start called more than once")
	}
	runCtx, cancel := context.WithCancel(ctx)
	c.cancel = cancel
	c.stop = runCtx.Done()
	go c.run(runCtx)
}

// run is the whole lifetime of the connection: dial, configure, read until
// something ends it. It always leaves WaitReady unblocked, always closes
// doneCh, and always closes the Events channel — in that order — no matter
// which of those paths it takes. A read error is emitted as EventError only
// when the connection ended on its own: ctx.Err() != nil means WE ended it
// (Close cancelled the run context, or the caller's own ctx passed to Start
// did), which is not new information for a caller, and neither is a
// socket-level error arriving immediately after session.closed has already
// been dispatched — the caller already has the definitive outcome.
func (c *Client) run(ctx context.Context) {
	defer close(c.doneCh)
	defer close(c.events)
	defer c.markEnded()

	conn, err := c.dial(ctx)
	if err != nil {
		c.fail(err)
		return
	}
	c.mu.Lock()
	c.conn = conn
	c.connCtx = ctx
	c.mu.Unlock()

	start, err := c.buildSessionStart()
	if err != nil {
		c.fail(err)
		_ = conn.Close(websocket.StatusInternalError, "building session.start")
		return
	}
	if err := c.send(start); err != nil {
		c.fail(fmt.Errorf("live: sending session.start: %w", err))
		_ = conn.Close(websocket.StatusInternalError, "session.start failed")
		return
	}

	err = c.readLoop(ctx, conn)

	c.mu.Lock()
	c.conn = nil
	sawClosed := c.sawClosed
	c.mu.Unlock()

	if err != nil && ctx.Err() == nil && !sawClosed {
		c.emit(
			Event{
				Kind: EventError,
				Err:  fmt.Errorf("live: connection ended: %w", err),
			},
		)
	}
}

// markEnded flips the client into its terminal state and unblocks any
// WaitReady callers left waiting — the connection is not coming back
// regardless of why run is returning.
func (c *Client) markEnded() {
	c.mu.Lock()
	c.ended = true
	if !c.ready && c.fatalErr == nil {
		c.fatalErr = ErrClosed
	}
	c.signalReadyLocked()
	c.mu.Unlock()
}

// WaitReady blocks until session.started has arrived, ctx is done, or the
// connection has ended without ever becoming ready.
func (c *Client) WaitReady(ctx context.Context) error {
	for {
		c.mu.Lock()
		ready, fatal, sig := c.ready, c.fatalErr, c.readySignal
		c.mu.Unlock()
		if fatal != nil {
			return fatal
		}
		if ready {
			return nil
		}
		select {
		case <-sig:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// signalReadyLocked wakes WaitReady callers. c.mu must be held.
func (c *Client) signalReadyLocked() {
	close(c.readySignal)
	c.readySignal = make(chan struct{})
}

// fail records a fatal error, wakes any WaitReady callers, and emits it as
// an EventError. Called only from within run's own goroutine, always before
// the deferred channel closes, so emitting here can never race a close.
func (c *Client) fail(err error) {
	c.mu.Lock()
	if c.fatalErr == nil {
		c.fatalErr = err
	}
	c.signalReadyLocked()
	c.mu.Unlock()
	c.emit(Event{Kind: EventError, Err: err})
}

// emit delivers an event, blocking rather than dropping it. Audio is never
// silently discarded: a stalled consumer should feel that as backpressure,
// not lose a chunk it can never get back.
//
// Until the session is being torn down. A caller that has what it wanted
// stops reading and calls Close; blocking then would keep the read loop, and
// Close's wait for it, alive forever. Once the run context is done nobody is
// owed another event, so the send gives up.
func (c *Client) emit(ev Event) {
	select {
	case c.events <- ev:
	case <-c.stop:
	}
}

// dial opens one WebSocket connection. The handshake's own HTTP status is
// the only thing that distinguishes a bad key from a bad network, and
// callers act on that difference: one is worth reporting as
// ErrUnauthorized and the other is not.
func (c *Client) dial(ctx context.Context) (wsConn, error) {
	dialCtx, cancel := context.WithTimeout(ctx, dialTimeout)
	defer cancel()

	if _, err := url.Parse(c.cfg.BaseURL); err != nil {
		return nil, fmt.Errorf("live: parsing BaseURL: %w", err)
	}
	header := http.Header{"Authorization": {"Bearer " + c.cfg.APIKey}}
	if c.cfg.SafetyIdentifier != "" {
		header.Set("OpenAI-Safety-Identifier", c.cfg.SafetyIdentifier)
	}
	conn, err := c.dialer.Dial(dialCtx, c.cfg.BaseURL, header)
	if err != nil {
		if s := err.Error(); strings.Contains(s, "http 401") ||
			strings.Contains(s, "http 403") {
			return nil, fmt.Errorf("%w: %v", ErrUnauthorized, err)
		}
		return nil, err
	}
	return conn, nil
}

// readLoop reads server events until the connection fails or ctx — the
// CONNECTION's own context, never a per-read deadline — is done. A shorter
// deadline would not abandon just one frame: coder/websocket closes the
// whole connection when a Read context expires; see transport.go's wsConn
// doc.
func (c *Client) readLoop(ctx context.Context, conn wsConn) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		c.dispatch(data)
	}
}

// dispatch turns one server event into zero or more Events. Unknown event
// types become EventOther rather than being treated as errors: the provider
// ships its own versions and may send events this client never asked for.
func (c *Client) dispatch(data []byte) {
	var env serverEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		c.emit(
			Event{
				Kind: EventError,
				Err:  fmt.Errorf("live: undecodable server event: %w", err),
				Raw:  data,
			},
		)
		return
	}

	switch env.Type {
	case "session.started":
		var s struct {
			ID string `json:"id"`
		}
		if len(env.Session) > 0 {
			_ = json.Unmarshal(env.Session, &s)
		}
		c.mu.Lock()
		c.sessionID = s.ID
		if !c.ready {
			c.ready = true
			c.signalReadyLocked()
		}
		c.mu.Unlock()
		c.emit(Event{Kind: EventSessionStarted, Raw: data})

	case "session.output_audio.delta":
		pcm, err := base64.StdEncoding.DecodeString(env.Delta)
		if err != nil {
			c.emit(
				Event{
					Kind: EventError,
					Err:  fmt.Errorf("live: undecodable audio delta: %w", err),
					Raw:  data,
				},
			)
			return
		}
		c.emit(Event{Kind: EventAudio, Audio: pcm, Raw: data})

	case "session.input_transcript.delta":
		c.emit(
			Event{
				Kind:    EventInputTranscript,
				Text:    env.Delta,
				StartMs: env.StartMs,
				EndMs:   env.EndMs,
				Raw:     data,
			},
		)

	case "session.output_transcript.delta":
		c.emit(
			Event{
				Kind:    EventOutputTranscript,
				Text:    env.Delta,
				StartMs: env.StartMs,
				EndMs:   env.EndMs,
				Raw:     data,
			},
		)

	case "session.delegation.created":
		if env.Delegation == nil {
			c.emit(Event{Kind: EventOther, Type: env.Type, Raw: data})
			return
		}
		if env.Delegation.ResponseID != "" {
			c.rememberResponseID(env.Delegation.ID, env.Delegation.ResponseID)
		}
		c.emit(Event{Kind: EventDelegation, Delegation: Delegation{
			ID:         env.Delegation.ID,
			Target:     env.Delegation.Target,
			ResponseID: env.Delegation.ResponseID,
			OffsetMs:   env.OffsetMs,
		}, Raw: data})

	case "response.event":
		c.dispatchResponseEvent(env, data)

	case "session.usage.updated":
		u := Usage{}
		if env.Usage != nil {
			u.Seconds = env.Usage.Seconds
		}
		if env.ContextWindow != nil {
			u.ContextWindowRatio = env.ContextWindow.UsageRatio
			u.HasContextWindow = true
		}
		c.emit(Event{Kind: EventUsage, Usage: u, Raw: data})

	case "session.input_audio.muted":
		c.emit(Event{Kind: EventMuted, Raw: data})

	case "session.input_audio.unmuted":
		c.emit(Event{Kind: EventUnmuted, Raw: data})

	case "session.closed":
		closed := Closed{Reason: env.Reason}
		if env.Usage != nil {
			closed.Seconds = env.Usage.Seconds
		}
		c.mu.Lock()
		c.closedPayload = closed
		c.sawClosed = true
		c.mu.Unlock()
		c.closedSignalOnce.Do(func() { close(c.sessionClosedSignal) })
		c.emit(Event{Kind: EventClosed, Closed: closed, Raw: data})

	case "error":
		c.dispatchError(env, data)

	default:
		c.emit(Event{Kind: EventOther, Type: env.Type, Raw: data})
	}
}

// dispatchResponseEvent unwraps a response.event envelope and dispatches on
// the nested Responses event's own type.
func (c *Client) dispatchResponseEvent(env serverEnvelope, raw []byte) {
	delegationID := ""
	if env.DelegationID != nil {
		delegationID = *env.DelegationID
	}

	var nested wireResponsesEvent
	if len(env.Event) == 0 || json.Unmarshal(env.Event, &nested) != nil {
		c.emit(
			Event{
				Kind: EventError,
				Err:  fmt.Errorf("live: undecodable response.event payload"),
				Raw:  raw,
			},
		)
		return
	}

	switch nested.Type {
	case "response.output_item.done":
		if nested.Item == nil || nested.Item.Type != "function_call" {
			c.emit(Event{Kind: EventOther, Type: nested.Type, Raw: raw})
			return
		}
		c.emit(Event{Kind: EventFunctionCall, FunctionCall: FunctionCall{
			DelegationID: delegationID,
			ResponseID:   c.responseIDFor(delegationID),
			CallID:       nested.Item.CallID,
			Name:         nested.Item.Name,
			Arguments:    nested.Item.Arguments,
		}, Raw: raw})

	case "response.completed",
		"response.failed",
		"response.cancelled",
		"response.incomplete":
		status := strings.TrimPrefix(nested.Type, "response.")
		responseID := c.responseIDFor(delegationID)
		var usage BackendUsage
		if nested.Response != nil {
			if nested.Response.ID != "" {
				responseID = nested.Response.ID
				c.rememberResponseID(delegationID, responseID)
			}
			if nested.Response.Status != "" {
				status = nested.Response.Status
			}
			if nested.Response.Usage != nil {
				cached := nested.Response.Usage.InputTokensDetails.CachedTokens
				usage = BackendUsage{
					InputTokens: max(
						nested.Response.Usage.InputTokens-cached,
						0,
					),
					CacheReadTokens: cached,
					OutputTokens:    nested.Response.Usage.OutputTokens,
					ReasoningTokens: nested.Response.Usage.OutputTokensDetails.ReasoningTokens,
				}
			}
		}
		c.emit(
			Event{
				Kind: EventBackendResponseDone,
				BackendResponse: BackendResponse{
					DelegationID: delegationID,
					ResponseID:   responseID,
					Status:       status,
					Usage:        usage,
				},
				Raw: raw,
			},
		)

	default:
		c.emit(Event{Kind: EventOther, Type: nested.Type, Raw: raw})
	}
}

// dispatchError turns an error event into an EventError, classifying it as
// ErrOutOfCredit or ErrUnauthorized where the provider's own code or type
// says so, so a caller can react to the specific condition with errors.Is
// rather than parsing prose.
func (c *Client) dispatchError(env serverEnvelope, raw []byte) {
	if env.Error == nil {
		c.emit(
			Event{
				Kind: EventError,
				Err:  fmt.Errorf("live: error event with no error detail"),
				Raw:  raw,
			},
		)
		return
	}
	e := env.Error
	var err error
	switch {
	case isOutOfCredit(e):
		err = fmt.Errorf("live: %s: %w", e.Message, ErrOutOfCredit)
	case isAuthError(e):
		err = fmt.Errorf("live: %s: %w", e.Message, ErrUnauthorized)
	case e.Code != "":
		err = fmt.Errorf("live: %s: %s", e.Code, e.Message)
	default:
		err = fmt.Errorf("live: %s", e.Message)
	}
	c.emit(Event{Kind: EventError, Err: err, Code: e.Code, Raw: raw})
}

// isOutOfCredit recognises the provider's own machine-readable signal for
// an account that cannot pay. Not documented for Live specifically at the
// time this was written — matched the same way realtime's isOutOfCredit
// matches it for the Realtime API, which does document it.
func isOutOfCredit(e *wireError) bool {
	return e != nil &&
		(strings.Contains(e.Code, "insufficient_quota") || strings.Contains(e.Type, "insufficient_quota"))
}

// isAuthError recognises a rejected key on an event arriving over an
// already-open socket, as opposed to a rejected dial. Also not documented
// for Live specifically at the time this was written; matched on the
// vocabulary OpenAI APIs generally use for this condition.
func isAuthError(e *wireError) bool {
	return e != nil && (strings.Contains(e.Type, "authentication") ||
		strings.Contains(e.Code, "invalid_api_key") || strings.Contains(e.Code, "unauthorized"))
}

// rememberResponseID records which backend response id a delegation is
// currently answered by, so a later function-call result can be addressed
// to it. A call with no delegation or response id is a no-op.
func (c *Client) rememberResponseID(delegationID, responseID string) {
	if delegationID == "" || responseID == "" {
		return
	}
	c.delegMu.Lock()
	c.delegationResponses[delegationID] = responseID
	c.delegMu.Unlock()
}

// responseIDFor returns the backend response id last recorded for
// delegationID, or empty if none has been.
func (c *Client) responseIDFor(delegationID string) string {
	if delegationID == "" {
		return ""
	}
	c.delegMu.Lock()
	defer c.delegMu.Unlock()
	return c.delegationResponses[delegationID]
}

// nextEventID generates a unique event_id for one client event.
func (c *Client) nextEventID() string {
	return fmt.Sprintf("live_%d", c.eventSeq.Add(1))
}

// writable returns the current connection and its context, or the sentinel
// error explaining why there is none: ErrNotReady before the socket exists,
// ErrClosed once it never will again.
func (c *Client) writable() (wsConn, context.Context, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ended {
		return nil, nil, ErrClosed
	}
	if c.conn == nil {
		return nil, nil, ErrNotReady
	}
	return c.conn, c.connCtx, nil
}

// send serialises and writes one client event. Like realtime's sendLive, it
// gates on a connection existing, not on session.started having arrived:
// audio and other sends made in the window between Start and session.started
// are delivered rather than dropped. Writes are serialised through writeMu
// because coder/websocket permits one writer at a time, and an audio append
// races every other send by nature.
func (c *Client) send(v any) error {
	conn, ctx, err := c.writable()
	if err != nil {
		return err
	}
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("live: encoding client event: %w", err)
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if err := conn.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("live: writing client event: %w", err)
	}
	return nil
}

// maxAppendBytes bounds session.instructions.append, session.thinking.append,
// and session.commentary.append content. The API limits these to 500
// tokens; this package has no tokenizer to check that directly, so it
// enforces a conservative byte approximation instead: 500 tokens * 4
// bytes/token, the commonly cited average for English text. A caller
// writing a denser script (CJK, heavy punctuation) may need to send less
// than this cap allows to stay under the real limit; there is no way to
// check that locally without a tokenizer.
const maxAppendBytes = 2000

// checkAppendSize rejects append content over maxAppendBytes, before it is
// sent.
func checkAppendSize(text string) error {
	if len(text) > maxAppendBytes {
		return fmt.Errorf(
			"live: append content is %d bytes, over the %d-byte conservative cap for the 500-token limit",
			len(text),
			maxAppendBytes,
		)
	}
	return nil
}

// AppendAudio streams captured audio to the provider: 24 kHz mono s16
// little-endian, matching SampleRateHz.
func (c *Client) AppendAudio(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	return c.send(map[string]any{
		"type":  "session.input_audio.append",
		"audio": base64.StdEncoding.EncodeToString(pcm),
	})
}

// Mute sends session.input_audio.mute. It is fire and forget: the
// acknowledgment arrives asynchronously as EventMuted.
func (c *Client) Mute() error {
	return c.send(
		map[string]any{
			"type":     "session.input_audio.mute",
			"event_id": c.nextEventID(),
		},
	)
}

// Unmute sends session.input_audio.unmute. It is fire and forget: the
// acknowledgment arrives asynchronously as EventUnmuted.
func (c *Client) Unmute() error {
	return c.send(
		map[string]any{
			"type":     "session.input_audio.unmute",
			"event_id": c.nextEventID(),
		},
	)
}

// appendContext sends one of session.instructions.append,
// session.thinking.append, or session.commentary.append.
func (c *Client) appendContext(eventType, text, delegationID string) error {
	if err := checkAppendSize(text); err != nil {
		return err
	}
	payload := map[string]any{
		"type":     eventType,
		"event_id": c.nextEventID(),
		"content":  text,
	}
	if delegationID == "" {
		payload["delegation_id"] = nil
	} else {
		payload["delegation_id"] = delegationID
	}
	return c.send(payload)
}

// AppendInstructions sends session.instructions.append: trusted application
// context that influences behavior and speech. delegationID selects an
// existing client delegation, or "" for session-wide context.
func (c *Client) AppendInstructions(text, delegationID string) error {
	return c.appendContext("session.instructions.append", text, delegationID)
}

// AppendThinking sends session.thinking.append: factual context that does
// not directly request speech. delegationID selects an existing client
// delegation, or "" for session-wide context.
func (c *Client) AppendThinking(text, delegationID string) error {
	return c.appendContext("session.thinking.append", text, delegationID)
}

// AppendCommentary sends session.commentary.append: information the model
// may say aloud, possibly paraphrased. delegationID selects an existing
// client delegation, or "" for session-wide context.
func (c *Client) AppendCommentary(text, delegationID string) error {
	return c.appendContext("session.commentary.append", text, delegationID)
}

// SendUserText sends response.item.create with a user message whose content
// is a single input_text part. Requires Responses delegation.
func (c *Client) SendUserText(text string) error {
	return c.send(map[string]any{
		"type":     "response.item.create",
		"event_id": c.nextEventID(),
		"item": map[string]any{
			"type": "message",
			"role": "user",
			"content": []map[string]any{
				{"type": "input_text", "text": text},
			},
		},
	})
}

// SendFunctionResult sends response.item.create with a function_call_output
// item, addressed to callID. It does not itself continue the response —
// call CreateResponse once every pending call has an answer. Requires
// Responses delegation.
func (c *Client) SendFunctionResult(callID, output string) error {
	if callID == "" {
		return fmt.Errorf(
			"live: SendFunctionResult needs a call id; an unaddressed result answers the wrong call",
		)
	}
	return c.send(map[string]any{
		"type":     "response.item.create",
		"event_id": c.nextEventID(),
		"item": map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		},
	})
}

// CreateResponse sends response.create: ask the delegated backend to
// continue, after every pending function call has a result. Requires
// Responses delegation.
func (c *Client) CreateResponse() error {
	return c.send(
		map[string]any{"type": "response.create", "event_id": c.nextEventID()},
	)
}

// Close asks the session to end gracefully: it sends session.close, waits
// (bounded by ctx) for session.closed, and then closes the socket either
// way. It is idempotent — every call, concurrent or sequential, returns the
// same result — and safe to call even if Start was never called or the
// connection already ended on its own.
func (c *Client) Close(ctx context.Context) (Closed, error) {
	c.closeOnce.Do(func() { c.doClose(ctx) })
	<-c.closeDone
	return c.closeResult, c.closeErr
}

// doClose runs Close's work exactly once, guarded by closeOnce. If nothing
// was ever opened there is nothing to close and nothing to wait for, and it
// returns immediately. A close that never received session.closed drops the
// socket with CloseNow rather than attempting the normal-closure handshake,
// which would add up to five more seconds waiting on a peer that is not
// answering.
func (c *Client) doClose(ctx context.Context) {
	defer close(c.closeDone)

	if !c.started.Load() {
		return
	}

	c.mu.Lock()
	conn := c.conn
	c.mu.Unlock()
	if conn != nil {
		_ = c.send(
			map[string]any{
				"type":     "session.close",
				"event_id": c.nextEventID(),
			},
		)
	}

	select {
	case <-c.sessionClosedSignal:
		c.mu.Lock()
		c.closeResult = c.closedPayload
		c.mu.Unlock()
	case <-c.doneCh:
		c.mu.Lock()
		fatal := c.fatalErr
		c.mu.Unlock()
		if fatal == nil {
			fatal = ErrClosed
		}
		c.closeErr = fmt.Errorf(
			"live: connection ended before session.closed: %w",
			fatal,
		)
	case <-ctx.Done():
		c.closeErr = fmt.Errorf(
			"live: waiting for session.closed: %w",
			ctx.Err(),
		)
	}

	if c.cancel != nil {
		c.cancel()
	}
	c.mu.Lock()
	conn = c.conn
	c.conn = nil
	c.mu.Unlock()
	if conn != nil {
		if c.closeErr == nil {
			_ = conn.Close(websocket.StatusNormalClosure, "client closing")
		} else {
			_ = conn.CloseNow()
		}
	}
	<-c.doneCh
}
