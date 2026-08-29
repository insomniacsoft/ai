// Package realtime is a client for the OpenAI Realtime API: one
// speech-to-speech session per voice conversation, carrying whatever toolset
// the caller hands it.
//
// # What this is not
//
// It is not a text-message transport and not an llm.LLM. Both seams are the
// wrong shape — a text transport pays for two model turns to carry voice
// through it, and llm.LLM is stateless-per-call with no audio, so reaching it
// requires transcribing first and throws away everything voice is for.
//
// # What it copies, and what it deliberately does not
//
// The lifecycle skeleton is the conventional one for a reconnecting client:
// New / Start / Close / WaitConnected, a connectLoop with full-jitter
// backoff, in-flight callers drained on disconnect. What it does NOT carry is
// the machinery such a client usually grows — an id-correlated pending map, a
// subscription replay. This is a bidirectional event stream with almost no
// request/response pairing and no subscriptions. An implementer who
// dutifully builds a pending map here has imported the wrong abstraction and
// will spend the rest of the file fighting it.
package realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

const defaultBaseURL = "wss://api.openai.com/v1/realtime"

// Tunables. Overridable through Option for tests only; production uses these.
const (
	defaultDialTimeout = 15 * time.Second
	defaultBackoffBase = 500 * time.Millisecond
	defaultBackoffMax  = 30 * time.Second
	// One audio delta measured at 4800 bytes of PCM; a whole response is
	// many of those plus a transcript. 8 MiB is far above any single frame
	// and far below anything that threatens the process.
	defaultReadLimit = 8 << 20
	// Enough for roughly a second and a half of 100 ms audio deltas plus the
	// surrounding lifecycle events. Past this, a stalled consumer is a fact
	// worth saying out loud rather than absorbing into a larger queue.
	eventBufferSize = 64
)

// Sentinel errors a caller can match with errors.Is.
var (
	// ErrNotConnected is returned by every send when there is no live
	// session — including immediately after a disconnect, so a caller waiting
	// on a reply fails fast instead of blocking on a socket that is gone.
	ErrNotConnected = errors.New("realtime: not connected")

	// ErrAuthRejected means the API key was refused. Retrying with the same
	// key cannot succeed, so the connect loop stops rather than hammering.
	ErrAuthRejected = errors.New("realtime: authentication rejected")

	// ErrSessionExpired is the provider ending a session at its duration cap.
	// It is a normal end of life, not a fault: the engine opens a new session
	// and replays a bounded history.
	ErrSessionExpired = errors.New("realtime: session expired")

	// ErrNoPlaybackReport means the downstream consumer has not reported how
	// much of what we sent it has actually played, so there is no honest
	// heard position to truncate at.
	ErrNoPlaybackReport = errors.New("realtime: no playback report from the downstream consumer")
)

// PlaybackReporter is the downstream consumer's own account of what it has
// NOT played yet.
//
// That figure has to be reported by the consumer rather than computed from a
// local clock, because the two crystals drift: at ±50 ppm over an hour that
// is ~180 ms, which is enough to misplace a word permanently. So the number
// comes from the device or it does not come at all — Buffered returning
// false is an honest "I do not know", and Truncate refuses rather than
// inventing one.
type PlaybackReporter interface {
	// Buffered returns the duration of audio the downstream consumer has
	// been handed but has not yet played, and whether that figure is
	// current.
	Buffered() (time.Duration, bool)
}

// Event is one thing worth telling the engine about.
type Event struct {
	Kind EventKind

	// Session is set on EventSessionReady.
	Session SessionInfo

	// Audio carries decoded PCM for EventAudio: 24 kHz mono s16 little-endian.
	// Resampling to whatever rate the downstream consumer wants is the
	// caller's job; this package does not touch the samples.
	Audio []byte

	// Transcript is what the model said (EventTranscript) or what the human
	// said (EventInputTranscript).
	Transcript string

	// Calls carries the tool calls of one completed response
	// (EventResponseDone), correlated with the transcript that accompanied
	// them.
	Calls []ToolCall

	// ItemID names the assistant audio item currently playing. Truncate needs
	// it, so the engine has to have seen it before a barge-in can be honest.
	ItemID string

	// ResponseID correlates everything belonging to one response.
	ResponseID string

	// Usage is set on EventResponseDone when the provider reported it.
	Usage *Usage

	// Preamble is how much audio this response streamed before its first tool
	// call arrived — the speech the listener hears while the tool runs.
	//
	// Zero on a response that called a tool before saying anything, which is
	// the case where the room hears a gap. Also zero, uninterestingly, on a
	// response with no calls at all; read it together with len(Calls).
	Preamble time.Duration

	// Interrupted marks an EventResponseDone that was assembled from a
	// response the provider never finished — the socket dropped mid-turn.
	// Its Calls were fully parsed, so they are not lost; whether to ACT on a
	// decision the model never got to finish stating is the engine's call,
	// which is exactly why this is a flag and not a silent discard.
	Interrupted bool

	// Status is the provider's own verdict on an EventResponseDone:
	// "completed", "cancelled", "incomplete", "failed". Carried through
	// because a response the provider threw away arrives with no audio, no
	// transcript and no calls — which is byte-for-byte what a completed
	// response that had nothing to say looks like. On 2026-08-24 that
	// ambiguity read as "the errand is done" and hung up on the person
	// immediately after the preamble.
	Status string

	// IncompleteReason is the provider's detail for a Status that is not
	// "completed" — "turn_detected" when its VAD decided somebody started
	// speaking, "max_output_tokens" at the cap, and so on.
	//
	// A "failed" response carries its detail in an error object rather than
	// here, so this falls back to that error's text: a status of "failed"
	// with an empty reason says only that the caller got no answer, which is
	// the one thing already obvious from the silence.
	IncompleteReason string

	// Err is set on EventError. ErrSessionExpired is delivered this way too,
	// because expiry is something the engine acts on rather than something
	// the connection layer can resolve alone.
	Err error

	// CausedByEventID is set on EventError when the provider named which of
	// OUR client events its rejection is about (serverError.EventID —
	// documented as the event_id of the client event that caused the
	// error). Empty whenever the provider omitted it, OR when nothing we
	// sent was tagged with an event_id in the first place — see
	// HistoryItem.EventID. A caller must treat empty as "cannot be
	// correlated to anything we sent", never as "correlates to the item
	// that happened to carry the empty id", because untagged sends never
	// carry one.
	//
	// This exists because ReplayHistory's own return value only ever
	// reports a TRANSPORT failure (the socket write itself failing) — a
	// content rejection of one specific item is answered on the wire,
	// asynchronously, as this event, and without a way to trace it back to
	// the item that caused it, a caller has no way to react to just that
	// one item without guessing.
	CausedByEventID string
}

// EventKind discriminates Event.
type EventKind int

const (
	// EventSessionReady fires when the provider has accepted our session
	// configuration — not merely when the socket opened. The distinction
	// matters: a successful connection proves nothing about what was
	// configured.
	EventSessionReady EventKind = iota
	EventAudio
	EventTranscript
	EventInputTranscript
	EventSpeechStarted
	EventSpeechStopped
	EventResponseDone
	EventError
)

func (k EventKind) String() string {
	switch k {
	case EventSessionReady:
		return "session_ready"
	case EventAudio:
		return "audio"
	case EventTranscript:
		return "transcript"
	case EventInputTranscript:
		return "input_transcript"
	case EventSpeechStarted:
		return "speech_started"
	case EventSpeechStopped:
		return "speech_stopped"
	case EventResponseDone:
		return "response_done"
	case EventError:
		return "error"
	}
	return fmt.Sprintf("kind(%d)", int(k))
}

// ToolCall is one function call the model made.
type ToolCall struct {
	// CallID is what a result must be addressed to. It is NOT ItemID; both
	// travel on the same events, and using one for the other answers a
	// question the model did not ask.
	CallID string
	ItemID string
	Name   string

	// Arguments is the model's JSON, byte for byte. It is not re-encoded:
	// the provider emits its own whitespace, and a client that normalises
	// here is a client that has already started interpreting arguments the
	// guard has not seen yet.
	Arguments string

	// Preamble is what the model said aloud in the same response — "sting
	// lumina din hol" before the call. Correlated by response id, so it
	// arrives with the call whether the transcript came first or second.
	Preamble string
}

// Config configures a Client.
type Config struct {
	// APIKey is the bearer token. A plain string rather than a redacting
	// type: a protocol client should not impose one on its callers, and
	// nothing in this package ever formats a whole Config, so there is no
	// print path for it to protect. The caller keeps the secret a secret up
	// to this boundary and reveals it here.
	APIKey  string
	BaseURL string
	Session SessionConfig
}

// Validate checks the minimum shape a Client can use.
func (c Config) Validate() error {
	if c.APIKey == "" {
		return errors.New("realtime: Config.APIKey is required")
	}
	if err := c.Session.Validate(); err != nil {
		return err
	}
	if c.BaseURL != "" {
		if _, err := url.Parse(c.BaseURL); err != nil {
			return fmt.Errorf("realtime: Config.BaseURL: %w", err)
		}
	}
	return nil
}

// Client holds one reconnecting Realtime session.
type Client struct {
	cfg    Config
	logger *slog.Logger
	dialer wsDialer

	dialTimeout time.Duration
	backoffBase time.Duration
	backoffMax  time.Duration

	playback PlaybackReporter

	events chan Event

	started atomic.Bool
	cancel  context.CancelFunc
	doneCh  chan struct{}

	mu       sync.Mutex
	conn     wsConn
	connCtx  context.Context
	closed   bool
	fatalErr error
	session  SessionInfo
	ready    bool
	// readySignal is closed and replaced whenever readiness changes, so
	// WaitReady can block without polling.
	readySignal chan struct{}

	// writeMu serialises writes. coder/websocket permits one writer at a
	// time, and audio append races everything else by nature.
	writeMu sync.Mutex

	// turns accumulates per-response state so a tool call and the transcript
	// that accompanied it are reported together regardless of arrival order.
	turnsMu sync.Mutex
	turns   map[string]*turn

	// sentMu guards the per-item count of audio this client has handed to the
	// downstream consumer. Truncate subtracts the downstream consumer's
	// unplayed depth from it.
	sentMu   sync.Mutex
	sentByID map[string]time.Duration
}

// turn is the in-flight state of one response.
type turn struct {
	transcript string
	calls      []ToolCall

	// preambleAudio counts the audio bytes this response streamed BEFORE its
	// first tool call arrived on the wire.
	//
	// It is a byte count rather than a pair of timestamps on purpose. The
	// question it answers — "had the model started speaking before it reached
	// for a tool?" — is about the room, and what the room hears is bytes of
	// audio, not elapsed wall clock: a burst that arrives in one frame still
	// takes its full duration to play. A count is also deterministic, so a
	// test can assert on it without a fake clock, and this client has no
	// injectable clock to fake.
	//
	// Zero on a response whose first event was the tool call means the
	// listener would hear silence while the call runs. That is the whole
	// point of measuring it: progressive tool disclosure is only free if this
	// number is reliably large.
	preambleAudio int
	// sawCall latches once the first tool call has arrived, so audio that
	// follows the call is not counted as though it preceded it.
	sawCall bool
}

// Option configures a Client at construction.
type Option func(*Client)

// WithLogger overrides the default logger.
func WithLogger(l *slog.Logger) Option { return func(c *Client) { c.logger = l } }

// WithPlaybackReporter supplies the downstream consumer's playback account.
func WithPlaybackReporter(p PlaybackReporter) Option {
	return func(c *Client) { c.playback = p }
}

func withDialer(d wsDialer) Option { return func(c *Client) { c.dialer = d } }
func withBackoff(base, max time.Duration) Option {
	return func(c *Client) { c.backoffBase, c.backoffMax = base, max }
}

// New builds a Client. It does not connect — call Start.
func New(cfg Config, opts ...Option) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	c := &Client{
		cfg:         cfg,
		logger:      slog.Default(),
		dialTimeout: defaultDialTimeout,
		backoffBase: defaultBackoffBase,
		backoffMax:  defaultBackoffMax,
		events:      make(chan Event, eventBufferSize),
		readySignal: make(chan struct{}),
		turns:       make(map[string]*turn),
		sentByID:    make(map[string]time.Duration),
	}
	for _, o := range opts {
		o(c)
	}
	if c.dialer == nil {
		c.dialer = realDialer{readLimit: defaultReadLimit, timeout: c.dialTimeout}
	}
	return c, nil
}

// Events is the stream of things worth acting on. It is closed when the
// client closes.
func (c *Client) Events() <-chan Event { return c.events }

// Start begins connecting in the background. Calling it twice is a
// programming error and panics.
func (c *Client) Start(ctx context.Context) {
	if !c.started.CompareAndSwap(false, true) {
		panic("realtime: Client.Start called more than once")
	}
	runCtx, cancel := context.WithCancel(ctx)
	c.cancel = cancel
	c.doneCh = make(chan struct{})
	go c.connectLoop(runCtx)
}

// WaitReady blocks until the provider has ACCEPTED our session configuration,
// ctx is done, or the connect loop has given up permanently.
//
// Deliberately not "WaitConnected": an open socket carrying somebody else's
// default persona and threshold VAD is not a session a caller can use, and
// this method exists precisely so the two are never confused.
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

// SessionInfo returns what the provider said about the live session.
func (c *Client) SessionInfo() SessionInfo {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.session
}

// Close stops the connect loop and closes the live connection. It blocks
// until the background goroutine has exited.
func (c *Client) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	c.mu.Unlock()

	if c.cancel != nil {
		c.cancel()
	}
	if c.doneCh != nil {
		<-c.doneCh
	}
	c.mu.Lock()
	conn := c.conn
	c.conn = nil
	c.mu.Unlock()
	if conn != nil {
		_ = conn.Close(websocket.StatusNormalClosure, "client closing")
	}
	close(c.events)
	return nil
}

// ── connect / configure / reconnect ─────────────────────────────────────

func (c *Client) connectLoop(ctx context.Context) {
	defer close(c.doneCh)

	b := &backoff{base: c.backoffBase, max: c.backoffMax}
	for {
		if ctx.Err() != nil {
			return
		}
		conn, err := c.dial(ctx)
		if err != nil {
			if errors.Is(err, ErrAuthRejected) {
				// Retrying with the same key cannot succeed, and a client
				// that keeps trying turns one wrong character in a config
				// file into a rate-limit incident.
				c.logger.Error("realtime: authentication rejected; not retrying with the same key", "error", err)
				c.mu.Lock()
				c.fatalErr = err
				c.signalReadyLocked()
				c.mu.Unlock()
				c.emit(Event{Kind: EventError, Err: err})
				return
			}
			c.logger.Warn("realtime: connect failed, retrying", "error", err)
			if !sleepCtx(ctx, b.Next()) {
				return
			}
			continue
		}
		b.Reset()

		connCtx, cancelConn := context.WithCancel(ctx)
		c.mu.Lock()
		c.conn, c.connCtx = conn, connCtx
		c.mu.Unlock()

		// Configure BEFORE announcing readiness. The socket is open at this
		// point but the session still carries the provider's own default
		// instructions and threshold VAD, and handing that to the engine
		// would be exactly the "a 200 proves nothing about what was sent"
		// mistake.
		if err := c.configure(connCtx); err != nil {
			c.logger.Warn("realtime: configuring the session failed", "error", err)
			cancelConn()
			c.teardown(conn)
			if !sleepCtx(ctx, b.Next()) {
				return
			}
			continue
		}

		readErr := c.readLoop(connCtx, conn)
		cancelConn()
		c.teardown(conn)

		if ctx.Err() != nil {
			return
		}
		if errors.Is(readErr, ErrSessionExpired) {
			// The cap is a normal end of life. Reconnect at once rather
			// than backing off, and tell the engine so it can replay a
			// bounded history onto the new session.
			c.logger.Info("realtime: session reached the provider's duration cap; reconnecting")
			c.emit(Event{Kind: EventError, Err: ErrSessionExpired})
			continue
		}
		c.logger.Warn("realtime: connection lost", "error", readErr)
		if !sleepCtx(ctx, b.Next()) {
			return
		}
	}
}

func (c *Client) dial(ctx context.Context) (wsConn, error) {
	dialCtx, cancel := context.WithTimeout(ctx, c.dialTimeout)
	defer cancel()

	u, err := url.Parse(c.cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("realtime: parsing base URL: %w", err)
	}
	q := u.Query()
	q.Set("model", c.cfg.Session.Model)
	u.RawQuery = q.Encode()

	header := http.Header{"Authorization": {"Bearer " + c.cfg.APIKey}}
	conn, err := c.dialer.Dial(dialCtx, u.String(), header)
	if err != nil {
		// The handshake's own status is the only signal separating a refused
		// key from a refused network, and the two need opposite responses.
		if s := err.Error(); containsAny(s, "http 401", "http 403") {
			return nil, fmt.Errorf("%w: %v", ErrAuthRejected, err)
		}
		return nil, err
	}
	return conn, nil
}

// configure sends session.update and waits for the provider to echo it back.
func (c *Client) configure(ctx context.Context) error {
	params, err := c.cfg.Session.sessionParams()
	if err != nil {
		return err
	}
	return c.send(ctx, map[string]any{"type": ceSessionUpdate, "session": params})
}

func (c *Client) teardown(conn wsConn) {
	c.mu.Lock()
	if c.conn == conn {
		c.conn, c.connCtx = nil, nil
	}
	c.ready = false
	c.session = SessionInfo{}
	c.signalReadyLocked()
	c.mu.Unlock()

	c.flushTurns()

	c.sentMu.Lock()
	c.sentByID = make(map[string]time.Duration)
	c.sentMu.Unlock()

	_ = conn.Close(websocket.StatusNormalClosure, "reconnecting")
}

// signalReadyLocked wakes WaitReady callers. c.mu must be held.
func (c *Client) signalReadyLocked() {
	close(c.readySignal)
	c.readySignal = make(chan struct{})
}

// flushTurns hands over anything already parsed from a response the provider
// never finished, and clears the buffer for the next connection.
//
// The alternative — dropping it — loses a tool call that was fully received
// and understood, because the socket happened to close before response.done.
// The human asked for something, the model decided to do it, and the only
// record of that decision would be gone with no error anywhere. Reporting it
// as Interrupted keeps the decision and hands the judgement upward.
func (c *Client) flushTurns() {
	c.turnsMu.Lock()
	pending := c.turns
	c.turns = make(map[string]*turn)
	c.turnsMu.Unlock()

	for id, t := range pending {
		if t == nil || (len(t.calls) == 0 && t.transcript == "") {
			continue
		}
		calls := t.calls
		for i := range calls {
			calls[i].Preamble = t.transcript
		}
		c.emit(Event{
			Kind: EventResponseDone, ResponseID: id,
			Calls: calls, Transcript: t.transcript, Interrupted: true,
		})
	}
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if len(sub) <= len(s) {
			for i := 0; i+len(sub) <= len(s); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}

// backoff is full-jitter exponential backoff: delay is uniform in
// [0, min(max, base*2^attempt)]. Full jitter rather than a fixed schedule
// because a provider outage disconnects every client at once, and an
// unjittered schedule has them all return in lockstep.
type backoff struct {
	base, max time.Duration
	attempt   uint
}

func (b *backoff) Reset() { b.attempt = 0 }

func (b *backoff) Next() time.Duration {
	ceiling := b.base << min(b.attempt, 32)
	if ceiling > b.max || ceiling <= 0 {
		ceiling = b.max
	}
	b.attempt++
	// A floor of base keeps a "jittered" delay from collapsing to zero and
	// becoming a hot loop against a provider that is already struggling.
	d := b.base + time.Duration(rand.Int64N(int64(ceiling)))
	if d > b.max {
		d = b.max
	}
	return d
}

// ── reading and dispatch ────────────────────────────────────────────────

func (c *Client) readLoop(ctx context.Context, conn wsConn) error {
	for {
		// ctx here is the CONNECTION's context, never a per-read deadline:
		// coder/websocket closes the whole connection when a Read context
		// expires, so a short deadline does not abandon one frame, it
		// destroys the session. See transport.go's wsConn comment.
		_, data, err := conn.Read(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		if expired := c.dispatch(data); expired != nil {
			return expired
		}
	}
}

// dispatch turns one server event into zero or more Events. It returns a
// non-nil error only when the connection itself is finished — currently only
// session expiry, which the connect loop treats as a normal end of life.
//
// Unknown event types are ignored rather than treated as errors: the provider
// ships its own versions and volunteers events this client never asked for,
// and refusing to speak to a peer that says one unexpected thing is a fragile
// client, not a careful one.
func (c *Client) dispatch(data []byte) error {
	var ev serverEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		c.logger.Warn("realtime: undecodable server event", "error", err)
		return nil
	}

	switch ev.Type {
	case evSessionCreated, evSessionUpdated:
		var s sessionEnvelope
		if len(ev.Session) > 0 {
			_ = json.Unmarshal(ev.Session, &s)
		}
		info := SessionInfo{ID: s.ID, Model: s.Model}
		if s.ExpiresAt > 0 {
			info.ExpiresAt = time.Unix(s.ExpiresAt, 0)
		}
		c.mu.Lock()
		c.session = info
		// Readiness is session.UPDATED — the provider echoing our
		// configuration back. session.created only says a socket exists,
		// and it carries the provider's own default persona.
		if ev.Type == evSessionUpdated && !c.ready {
			c.ready = true
			c.signalReadyLocked()
			c.mu.Unlock()
			c.emit(Event{Kind: EventSessionReady, Session: info})
			return nil
		}
		c.mu.Unlock()
		return nil

	case evAudioDelta:
		pcm, err := base64.StdEncoding.DecodeString(ev.Delta)
		if err != nil {
			c.logger.Warn("realtime: undecodable audio delta", "error", err)
			return nil
		}
		// Count what we are about to hand downstream, per item. Truncate
		// subtracts the downstream consumer's unplayed depth from this to get
		// the heard position.
		c.addSent(ev.ItemID, bytesToDuration(len(pcm)))
		c.withTurn(ev.ResponseID, func(t *turn) {
			if !t.sawCall {
				t.preambleAudio += len(pcm)
			}
		})
		c.emit(Event{Kind: EventAudio, Audio: pcm, ItemID: ev.ItemID, ResponseID: ev.ResponseID})
		return nil

	case evAudioTranscriptDone:
		c.withTurn(ev.ResponseID, func(t *turn) { t.transcript = ev.Transcript })
		c.emit(Event{Kind: EventTranscript, Transcript: ev.Transcript,
			ItemID: ev.ItemID, ResponseID: ev.ResponseID})
		return nil

	case evFuncArgsDone:
		// Buffered rather than emitted: the accompanying transcript may not
		// have arrived yet, and a call reported without its preamble makes
		// the engine correlate it by hand. Both orderings converge here.
		c.withTurn(ev.ResponseID, func(t *turn) {
			t.sawCall = true
			t.calls = append(t.calls, ToolCall{
				CallID:    ev.CallID,
				ItemID:    ev.ItemID,
				Name:      ev.Name,
				Arguments: ev.Arguments,
			})
		})
		return nil

	case evInputTranscriptDone:
		c.emit(Event{Kind: EventInputTranscript, Transcript: ev.Transcript, ItemID: ev.ItemID})
		return nil

	case evSpeechStarted:
		c.emit(Event{Kind: EventSpeechStarted, ItemID: ev.ItemID})
		return nil

	case evSpeechStopped:
		c.emit(Event{Kind: EventSpeechStopped, ItemID: ev.ItemID})
		return nil

	case evResponseDone:
		var r responseEnvelope
		if len(ev.Response) > 0 {
			_ = json.Unmarshal(ev.Response, &r)
		}
		return c.finishResponse(r)

	case evError:
		err := error(ev.Error)
		if ev.Error != nil && isExpiry(ev.Error) {
			return ErrSessionExpired
		}
		c.logger.Warn("realtime: provider error event", "error", err)
		causedBy := ""
		if ev.Error != nil {
			causedBy = ev.Error.EventID
		}
		c.emit(Event{Kind: EventError, Err: err, CausedByEventID: causedBy})
		return nil
	}
	return nil
}

// finishResponse emits the correlated end of one response.
func (c *Client) finishResponse(r responseEnvelope) error {
	c.turnsMu.Lock()
	t := c.turns[r.ID]
	delete(c.turns, r.ID)
	c.turnsMu.Unlock()

	calls := []ToolCall(nil)
	transcript := ""
	preamble := time.Duration(0)
	if t != nil {
		calls, transcript = t.calls, t.transcript
		preamble = bytesToDuration(t.preambleAudio)
	}

	// The response's own output is the authority, and it is also the only
	// source when a whole response arrives without incremental events (a
	// reconnect mid-response, a provider that batches). Merge rather than
	// choose: a call already seen keeps its buffered arguments, and one seen
	// only here is still reported.
	for _, item := range r.Output {
		switch item.Type {
		case "function_call":
			if !hasCall(calls, item.CallID) {
				calls = append(calls, ToolCall{
					CallID: item.CallID, ItemID: item.ID,
					Name: item.Name, Arguments: item.Arguments,
				})
			}
		case "message":
			if transcript == "" {
				for _, part := range item.Content {
					if part.Transcript != "" {
						transcript = part.Transcript
					}
				}
			}
		}
	}
	for i := range calls {
		calls[i].Preamble = transcript
	}

	if r.StatusDetails.Error != nil && isExpiry(r.StatusDetails.Error) {
		return ErrSessionExpired
	}

	c.emit(Event{
		Kind: EventResponseDone, ResponseID: r.ID,
		Calls: calls, Transcript: transcript, Usage: r.Usage,
		Status: r.Status, IncompleteReason: reasonOf(r),
		Preamble: preamble,
	})
	return nil
}

// reasonOf is why a response is not "completed", from whichever field the
// provider used: incomplete responses populate status_details.reason, failed
// ones populate status_details.error instead.
func reasonOf(r responseEnvelope) string {
	if r.StatusDetails.Reason != "" {
		return r.StatusDetails.Reason
	}
	if e := r.StatusDetails.Error; e != nil {
		return e.Error()
	}
	return ""
}

func hasCall(calls []ToolCall, id string) bool {
	for _, c := range calls {
		if c.CallID == id {
			return true
		}
	}
	return false
}

// isExpiry recognises the provider ending a session at its duration cap.
//
// Matched on the code AND on the message, because the provider has shipped
// more than one spelling of this and the consequence of missing it is not
// cosmetic: an unrecognised expiry is treated as a transport fault, which
// backs off before reconnecting and drops the bounded-history replay a
// session-expiry reconnect depends on.
func isExpiry(e *serverError) bool {
	if e == nil {
		return false
	}
	return containsAny(e.Code, "session_expired", "session_timeout") ||
		containsAny(e.Message, "maximum duration", "session expired", "Session expired")
}

func (c *Client) withTurn(responseID string, fn func(*turn)) {
	if responseID == "" {
		return
	}
	c.turnsMu.Lock()
	defer c.turnsMu.Unlock()
	t := c.turns[responseID]
	if t == nil {
		t = &turn{}
		c.turns[responseID] = t
	}
	fn(t)
}

// emit delivers an event, blocking rather than dropping.
//
// Audio is never silently discarded: a dropped chunk is a hole in speech
// nobody can hear the shape of, so a stalled consumer blocks the reader and
// says so loudly instead.
func (c *Client) emit(ev Event) {
	c.mu.Lock()
	closed := c.closed
	c.mu.Unlock()
	if closed {
		return
	}
	select {
	case c.events <- ev:
	default:
		c.logger.Warn("realtime: event consumer is not keeping up; the reader is now blocking",
			"kind", ev.Kind.String())
		c.events <- ev
	}
}

func bytesToDuration(n int) time.Duration {
	// 24 kHz mono s16: two bytes per sample.
	return time.Duration(n) * time.Second / (SampleRateHz * 2)
}

func (c *Client) addSent(itemID string, d time.Duration) {
	if itemID == "" {
		return
	}
	c.sentMu.Lock()
	c.sentByID[itemID] += d
	c.sentMu.Unlock()
}

func (c *Client) sent(itemID string) time.Duration {
	c.sentMu.Lock()
	defer c.sentMu.Unlock()
	return c.sentByID[itemID]
}

// ── sending ─────────────────────────────────────────────────────────────

// send serialises one client event onto the socket.
func (c *Client) send(ctx context.Context, v any) error {
	c.mu.Lock()
	conn := c.conn
	c.mu.Unlock()
	if conn == nil {
		return ErrNotConnected
	}
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("realtime: encoding client event: %w", err)
	}
	// Serialised: coder/websocket permits one writer at a time, and audio
	// append races every other send by its nature.
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if err := conn.Write(ctx, websocket.MessageText, data); err != nil {
		return fmt.Errorf("realtime: writing client event: %w", err)
	}
	return nil
}

// sendLive sends on the current connection's own context.
func (c *Client) sendLive(v any) error {
	c.mu.Lock()
	ctx := c.connCtx
	c.mu.Unlock()
	if ctx == nil {
		return ErrNotConnected
	}
	return c.send(ctx, v)
}

// AppendAudio streams captured microphone audio to the provider: 24 kHz mono
// s16 little-endian, already resampled up from the downstream consumer's 16
// kHz.
//
// There is no matching Commit. Semantic turn detection decides when the
// human has finished, which is the whole reason for choosing it: a client
// that also committed by hand would be arbitrating turn ends against the
// model that was configured to arbitrate them.
func (c *Client) AppendAudio(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	return c.sendLive(map[string]any{
		"type":  ceAudioAppend,
		"audio": base64.StdEncoding.EncodeToString(pcm),
	})
}

// HeardPosition reports how much of item itemID the human has actually heard.
//
// This client knows what it SENT; only the downstream consumer knows what it
// PLAYED, because everything in between — the API socket, the device's 16
// KiB speaker buffer, the resampler's ring buffer, the mixer, the DAC — is
// latency this side cannot see. Reporting the sent position tells the model
// it said things nobody heard.
//
// It returns ErrNoPlaybackReport when the downstream consumer has not
// reported, rather than falling back to the sent position or to a clock. The
// number comes from the device or not at all: this process's crystal and the
// downstream consumer's drift, and at ±50 ppm over an hour that is ~180 ms —
// enough to misplace a word permanently, silently, and in a way no test
// would catch.
func (c *Client) HeardPosition(itemID string) (time.Duration, error) {
	if c.playback == nil {
		return 0, ErrNoPlaybackReport
	}
	buffered, ok := c.playback.Buffered()
	if !ok {
		return 0, ErrNoPlaybackReport
	}
	sent := c.sent(itemID)

	heard := sent - buffered
	// Clamp before the wire, not after. A position beyond what was sent
	// would have the model believe it played audio that does not exist, and
	// a negative one is not a position at all. Both are reachable from a
	// device report that is merely stale — the downstream consumer reporting
	// a depth from before the last chunk arrived is normal, not a fault.
	if heard < 0 {
		heard = 0
	}
	if heard > sent {
		heard = sent
	}
	return heard, nil
}

// Truncate tells the provider how much of an assistant audio item the human
// actually heard, so the model's context matches the room.
//
// Measured 2026-08-23: this cuts the item's AUDIO to exactly the reported
// position and leaves its TRANSCRIPT whole. So the position is load-bearing —
// and truncation alone does not make the record honest, because the model's
// text context keeps every word it generated. Handling that residual is the
// engine's responsibility; this method does not pretend to have solved it.
func (c *Client) Truncate(itemID string) (time.Duration, error) {
	if itemID == "" {
		return 0, errors.New("realtime: Truncate needs the item id of the audio being played")
	}
	heard, err := c.HeardPosition(itemID)
	if err != nil {
		return 0, err
	}
	if err := c.sendLive(map[string]any{
		"type":          ceItemTruncate,
		"item_id":       itemID,
		"content_index": 0,
		"audio_end_ms":  heard.Milliseconds(),
	}); err != nil {
		return 0, err
	}
	return heard, nil
}

// BargeIn is the pair, in the order a real interruption happens: stop
// generating, then correct the record.
//
// Cancel first because every millisecond of continued generation is more
// audio to account for; truncate second because the position is only
// meaningful once nothing further is being added. If the downstream consumer
// has not reported a playback position the cancel still happens — stopping
// the response is right regardless — and the truncation error is returned so
// the caller knows the record was left overstating what was heard.
func (c *Client) BargeIn(itemID string) (time.Duration, error) {
	if err := c.CancelResponse(); err != nil && !errors.Is(err, ErrNotConnected) {
		return 0, err
	}
	return c.Truncate(itemID)
}

// CancelResponse stops the in-flight response.
func (c *Client) CancelResponse() error {
	return c.sendLive(map[string]any{"type": ceResponseCancl})
}

// SendToolResult returns one tool's output, addressed to its call id.
//
// It deliberately does NOT create a response. The guard may have denied the
// call, the engine may be mid-confirmation, or several calls may be
// outstanding — speaking after the first result would be answering before the
// work is done. CreateResponse is a separate, explicit act.
func (c *Client) SendToolResult(callID, output string) error {
	if callID == "" {
		return errors.New("realtime: SendToolResult needs a call id; an unaddressed result answers the wrong call")
	}
	return c.sendLive(map[string]any{
		"type": ceItemCreate,
		"item": map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		},
	})
}

// CreateResponse asks the model to speak.
func (c *Client) CreateResponse() error {
	return c.sendLive(map[string]any{"type": ceResponseCreN})
}

// HistoryItem is one previously-spoken turn, for replay onto a new session.
type HistoryItem struct {
	// Role is "user" or "assistant".
	Role string
	// Text is what was said.
	Text string

	// EventID optionally tags this item's own conversation.item.create with
	// a caller-chosen client event id, so a later asynchronous rejection —
	// delivered as an EventError whose own CausedByEventID names the client
	// event that caused it, not correlated any other way — can be traced
	// back to precisely this item and nothing else ReplayHistory has ever
	// sent.
	//
	// Left empty (the default for every call site but one), the provider
	// assigns its own id and no later event can be traced back to
	// this item — which is correct for almost every caller: Say, the
	// mid-session notes (recheckScope, recheckRecording), the temporal
	// anchor, and the barge-in interruption note all have nothing defined
	// to fall back to, so a traceable rejection would not let them do
	// anything differently. Only the replay collector's assistant-role
	// items need this (see engine.go's conversationReplayItems and
	// fallbackAssistantReplay), because only there does a rejection have a
	// defined, different next step.
	EventID string
}

// ReplayHistory restores a bounded window of recent conversation after the
// provider ended a session at its duration cap.
//
// It replays as CONVERSATION ITEMS, one per turn. It must never fold this
// text into the session instructions, and the separation is not stylistic:
// instructions are trusted and conversation is not. Anything anyone in
// earshot said is untrusted input, so promoting it into the system prompt on
// a reconnect would launder a stranger's sentence into an instruction — and
// it would do so only on the reconnect path, which is the path least likely
// to be exercised in testing.
//
// A nil or empty error return means every item was WRITTEN TO THE SOCKET,
// not that the provider accepted each one — this client has never run a
// live probe of whether gpt-realtime wants "text" or "output_text" inside an
// assistant item's content (current documentation says the latter; this
// sends the former, unverified either way). A content rejection answers
// asynchronously, as an EventError, and correlates back to the rejected
// item only through that item's own HistoryItem.EventID, if the caller set
// one — see CausedByEventID's own comment for why this is a wire-protocol
// fact and not a gap in this function.
func (c *Client) ReplayHistory(items []HistoryItem) error {
	for _, it := range items {
		role := it.Role
		if role != "user" && role != "assistant" {
			return fmt.Errorf("realtime: history role %q is neither user nor assistant", role)
		}
		// The provider distinguishes the two by role, and the assistant's
		// value is "output_text" — NOT "text", which it rejects outright with
		// `Invalid value: 'text'. Value must be 'output_text'.`
		//
		// This branch shipped as "text" and was unproven for its whole life:
		// no call site passed an assistant role until conversation replay
		// did, and by then the mistake could only show up as a rejection
		// arriving asynchronously, after the reconnect had already been
		// reported as successful. Measured against the live API on
		// 2026-08-26 with cmd/vpeprobe, which sends all three shapes down one
		// session — assistant/text, assistant/output_text, and a
		// user/input_text control — so an acceptance can be told apart from a
		// session that had already died. Re-run it if this ever looks wrong
		// again; the answer is a fact about the provider, not about us.
		contentType := "input_text"
		if role == "assistant" {
			contentType = "output_text"
		}
		msg := map[string]any{
			"type": ceItemCreate,
			"item": map[string]any{
				"type": "message",
				"role": role,
				"content": []any{
					map[string]any{"type": contentType, "text": it.Text},
				},
			},
		}
		if it.EventID != "" {
			msg["event_id"] = it.EventID
		}
		if err := c.sendLive(msg); err != nil {
			return err
		}
	}
	return nil
}
