package live

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/joakimcarlsson/ai/tool"
)

// fakeTool is a tool.BaseTool carrying fixed metadata, to exercise
// serialisation.
type fakeTool struct{ info tool.Info }

// Info returns the tool's fixed metadata.
func (f fakeTool) Info() tool.Info { return f.info }

// Run is never called by these tests; only Info is exercised.
func (f fakeTool) Run(context.Context, tool.Call) (tool.Response, error) {
	return tool.Response{}, nil
}

// fakeServer is an httptest WebSocket server that accepts one connection
// per test.
type fakeServer struct {
	srv    *httptest.Server
	url    string
	header http.Header
	connCh chan *websocket.Conn
}

// newFakeServer starts a fakeServer, closed automatically at test cleanup.
func newFakeServer(t *testing.T) *fakeServer {
	t.Helper()
	fs := &fakeServer{connCh: make(chan *websocket.Conn, 1)}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fs.header = r.Header.Clone()
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("fake server: accept: %v", err)
			return
		}
		fs.connCh <- conn
	})
	fs.srv = httptest.NewServer(mux)
	t.Cleanup(fs.srv.Close)
	fs.url = "ws" + strings.TrimPrefix(fs.srv.URL, "http")
	return fs
}

// accept waits for the client to connect and returns the server's side of
// the socket.
func (fs *fakeServer) accept(t *testing.T) *websocket.Conn {
	t.Helper()
	select {
	case conn := <-fs.connCh:
		t.Cleanup(
			func() { _ = conn.Close(websocket.StatusNormalClosure, "test done") },
		)
		return conn
	case <-time.After(3 * time.Second):
		t.Fatal("fake server: the client never connected")
		return nil
	}
}

// readClientEvent reads and decodes the next message the client sent.
func readClientEvent(t *testing.T, conn *websocket.Conn) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatalf("reading client event: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("decoding client event %q: %v", data, err)
	}
	return m
}

// sendServerEvent writes one server event to the client.
func sendServerEvent(t *testing.T, conn *websocket.Conn, v any) {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshalling server event: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := conn.Write(ctx, websocket.MessageText, data); err != nil {
		t.Fatalf("writing server event: %v", err)
	}
}

// recvEvent reads the next Event off the client, or fails the test.
func recvEvent(t *testing.T, c *Client) Event {
	t.Helper()
	select {
	case ev, ok := <-c.Events():
		if !ok {
			t.Fatal("Events() closed unexpectedly")
		}
		return ev
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for an event")
		return Event{}
	}
}

// setupClient starts a Client against a fresh fake server and returns it
// alongside the server's side of the socket, before anything has been
// exchanged. mutate, when non-nil, edits the Config before New.
func setupClient(
	t *testing.T,
	mutate func(*Config),
) (*Client, *websocket.Conn, *fakeServer) {
	t.Helper()
	fs := newFakeServer(t)
	cfg := Config{
		APIKey:  "test-key",
		BaseURL: fs.url,
		Session: SessionConfig{
			Model:        "gpt-live-1",
			Instructions: "be brief",
		},
	}
	if mutate != nil {
		mutate(&cfg)
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	c.Start(ctx)
	conn := fs.accept(t)
	return c, conn, fs
}

// setupReadySession is setupClient plus the session.start/session.started
// handshake, for tests that only care about what happens afterward. It also
// drains the EventSessionStarted the handshake itself produces, so a test's
// first recvEvent sees what IT triggered, not this one.
func setupReadySession(
	t *testing.T,
	mutate func(*Config),
) (*Client, *websocket.Conn) {
	t.Helper()
	c, conn, _ := setupClient(t, mutate)
	readClientEvent(t, conn)
	sendServerEvent(t, conn, map[string]any{
		"type":     "session.started",
		"event_id": "evt_started_1",
		"session": map[string]any{
			"id":     "live_test_session",
			"model":  "gpt-live-1",
			"status": "active",
		},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := c.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady() error = %v", err)
	}
	startEv := recvEvent(t, c)
	if startEv.Kind != EventSessionStarted {
		t.Fatalf("first event = %+v, want EventSessionStarted", startEv)
	}
	return c, conn
}

// TestSessionStartPayloadSelectsClientDelegation checks that a session with
// no Responses backend asks for client delegation explicitly: without the
// delegation object the provider delegates nothing to the application.
func TestSessionStartPayloadSelectsClientDelegation(t *testing.T) {
	_, conn, _ := setupClient(t, nil)

	got := readClientEvent(t, conn)
	session, ok := got["session"].(map[string]any)
	if !ok {
		t.Fatalf("session is not an object: %v", got["session"])
	}
	delegation, ok := session["delegation"].(map[string]any)
	if !ok {
		t.Fatalf("session.delegation is not an object: %v", session["delegation"])
	}
	if delegation["type"] != "client" {
		t.Errorf("session.delegation.type = %v, want client", delegation["type"])
	}
	if _, ok := delegation["responses"]; ok {
		t.Errorf("session.delegation.responses = %v, want absent", delegation["responses"])
	}
}

// TestSessionStartPayload asserts the actual wire payload of the first
// message a session sends: model, instructions, audio format and voice, the
// API key on the Authorization header, and a Responses delegation with its
// backend model, instructions, tool choice and serialised tool schema.
func TestSessionStartPayload(t *testing.T) {
	lightTool := fakeTool{info: tool.Info{
		Name:        "light_off",
		Description: "Turn off a light by name.",
		Parameters: map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": "the light's name",
			},
		},
		Required: []string{"name"},
	}}
	parallel := false

	_, conn, fs := setupClient(t, func(cfg *Config) {
		cfg.APIKey = "sk-test-abc"
		cfg.Session.Voice = "marin"
		cfg.Session.Delegation = &ResponsesDelegation{
			Model:             "gpt-6-luna",
			Instructions:      "backend prompt",
			Tools:             []tool.BaseTool{lightTool},
			ToolChoice:        "auto",
			ParallelToolCalls: &parallel,
			ReasoningEffort:   "minimal",
			ServiceTier:       "priority",
		}
	})

	got := readClientEvent(t, conn)
	if got["type"] != "session.start" {
		t.Fatalf("type = %v, want session.start", got["type"])
	}
	if _, ok := got["event_id"].(string); !ok {
		t.Errorf("event_id missing or not a string: %v", got["event_id"])
	}

	session, ok := got["session"].(map[string]any)
	if !ok {
		t.Fatalf("session is not an object: %v", got["session"])
	}
	if session["model"] != "gpt-live-1" {
		t.Errorf("session.model = %v, want gpt-live-1", session["model"])
	}
	if session["instructions"] != "be brief" {
		t.Errorf(
			"session.instructions = %v, want %q",
			session["instructions"],
			"be brief",
		)
	}

	audio, ok := session["audio"].(map[string]any)
	if !ok {
		t.Fatalf("session.audio is not an object: %v", session["audio"])
	}
	format, ok := audio["format"].(map[string]any)
	if !ok {
		t.Fatalf("session.audio.format is not an object: %v", audio["format"])
	}
	if format["type"] != "audio/pcm" ||
		format["rate"] != float64(SampleRateHz) {
		t.Errorf(
			"session.audio.format = %v, want audio/pcm @ %d",
			format,
			SampleRateHz,
		)
	}
	output, ok := audio["output"].(map[string]any)
	if !ok {
		t.Fatalf("session.audio.output is not an object: %v", audio["output"])
	}
	if output["voice"] != "marin" {
		t.Errorf("session.audio.output.voice = %v, want marin", output["voice"])
	}

	delegation, ok := session["delegation"].(map[string]any)
	if !ok {
		t.Fatalf(
			"session.delegation is not an object: %v",
			session["delegation"],
		)
	}
	if delegation["type"] != "responses" {
		t.Errorf(
			"session.delegation.type = %v, want responses",
			delegation["type"],
		)
	}
	responses, ok := delegation["responses"].(map[string]any)
	if !ok {
		t.Fatalf(
			"session.delegation.responses is not an object: %v",
			delegation["responses"],
		)
	}
	if responses["model"] != "gpt-6-luna" {
		t.Errorf(
			"delegation.responses.model = %v, want gpt-6-luna",
			responses["model"],
		)
	}
	if responses["instructions"] != "backend prompt" {
		t.Errorf(
			"delegation.responses.instructions = %v, want %q",
			responses["instructions"],
			"backend prompt",
		)
	}
	if responses["tool_choice"] != "auto" {
		t.Errorf(
			"delegation.responses.tool_choice = %v, want auto",
			responses["tool_choice"],
		)
	}
	if responses["parallel_tool_calls"] != false {
		t.Errorf(
			"delegation.responses.parallel_tool_calls = %v, want false",
			responses["parallel_tool_calls"],
		)
	}
	if responses["service_tier"] != "priority" {
		t.Errorf(
			"delegation.responses.service_tier = %v, want priority",
			responses["service_tier"],
		)
	}
	if reasoning, _ := responses["reasoning"].(map[string]any); reasoning["effort"] != "minimal" {
		t.Errorf(
			"delegation.responses.reasoning = %v, want effort minimal",
			responses["reasoning"],
		)
	}

	tools, ok := responses["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf(
			"delegation.responses.tools = %v, want exactly one tool",
			responses["tools"],
		)
	}
	toolMap, ok := tools[0].(map[string]any)
	if !ok {
		t.Fatalf("tools[0] is not an object: %v", tools[0])
	}
	if toolMap["type"] != "function" || toolMap["name"] != "light_off" {
		t.Errorf("tools[0] = %v, want type=function name=light_off", toolMap)
	}
	params, ok := toolMap["parameters"].(map[string]any)
	if !ok {
		t.Fatalf(
			"tools[0].parameters is not an object: %v",
			toolMap["parameters"],
		)
	}
	if params["type"] != "object" {
		t.Errorf("tools[0].parameters.type = %v, want object", params["type"])
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatalf(
			"tools[0].parameters.properties is not an object: %v",
			params["properties"],
		)
	}
	if _, ok := props["name"]; !ok {
		t.Errorf(
			"tools[0].parameters.properties = %v, want a %q property",
			props,
			"name",
		)
	}
	required, ok := params["required"].([]any)
	if !ok || len(required) != 1 || required[0] != "name" {
		t.Errorf(
			"tools[0].parameters.required = %v, want [\"name\"]",
			params["required"],
		)
	}

	if got := fs.header.Get("Authorization"); got != "Bearer sk-test-abc" {
		t.Errorf(
			"Authorization header = %q, want %q",
			got,
			"Bearer sk-test-abc",
		)
	}
}

// TestWaitReadyAndSessionID checks that WaitReady blocks until
// session.started arrives, and that SessionID reflects what that event
// reported.
func TestWaitReadyAndSessionID(t *testing.T) {
	c, conn, _ := setupClient(t, nil)
	readClientEvent(t, conn)

	if err := c.WaitReady(mustNotBlock(t)); err == nil {
		t.Fatal(
			"WaitReady() returned before session.started; want it still blocked",
		)
	}

	sendServerEvent(t, conn, map[string]any{
		"type":     "session.started",
		"event_id": "evt_started_1",
		"session":  map[string]any{"id": "live_abc123", "model": "gpt-live-1"},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := c.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady() error = %v", err)
	}
	if got := c.SessionID(); got != "live_abc123" {
		t.Errorf("SessionID() = %q, want live_abc123", got)
	}
}

// mustNotBlock returns an already-expired context, so a WaitReady call
// against it returns immediately with ctx.Err() unless the client was
// already ready — used to probe "not ready yet" without a real wait.
func mustNotBlock(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	t.Cleanup(cancel)
	<-ctx.Done()
	return ctx
}

// TestAudioDeltasInOrder checks that output audio deltas decode to the
// right PCM bytes, and arrive in the order they were sent.
func TestAudioDeltasInOrder(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	chunk1 := []byte{1, 2, 3, 4, 5, 6}
	chunk2 := []byte{7, 8, 9, 10}
	sendServerEvent(t, conn, map[string]any{
		"type":  "session.output_audio.delta",
		"delta": base64.StdEncoding.EncodeToString(chunk1),
	})
	sendServerEvent(t, conn, map[string]any{
		"type":  "session.output_audio.delta",
		"delta": base64.StdEncoding.EncodeToString(chunk2),
	})

	ev1 := recvEvent(t, c)
	if ev1.Kind != EventAudio || !bytes.Equal(ev1.Audio, chunk1) {
		t.Fatalf("ev1 = %+v, want EventAudio with %v", ev1, chunk1)
	}
	ev2 := recvEvent(t, c)
	if ev2.Kind != EventAudio || !bytes.Equal(ev2.Audio, chunk2) {
		t.Fatalf("ev2 = %+v, want EventAudio with %v", ev2, chunk2)
	}
}

// TestTranscriptDeltas checks that input and output transcript deltas carry
// their text and start/end ms, on the matching event kind.
func TestTranscriptDeltas(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	sendServerEvent(t, conn, map[string]any{
		"type": "session.input_transcript.delta", "event_id": "e1",
		"delta": "a table for two", "start_ms": 1600, "end_ms": 3400,
	})
	ev := recvEvent(t, c)
	if ev.Kind != EventInputTranscript || ev.Text != "a table for two" ||
		ev.StartMs != 1600 ||
		ev.EndMs != 3400 {
		t.Fatalf("input transcript event = %+v", ev)
	}

	sendServerEvent(t, conn, map[string]any{
		"type": "session.output_transcript.delta", "event_id": "e2",
		"delta": "would you like that table", "start_ms": 5400, "end_ms": 7200,
	})
	ev = recvEvent(t, c)
	if ev.Kind != EventOutputTranscript ||
		ev.Text != "would you like that table" ||
		ev.StartMs != 5400 ||
		ev.EndMs != 7200 {
		t.Fatalf("output transcript event = %+v", ev)
	}
}

// TestDelegationFunctionCallAndUsage runs a delegation end to end: a
// response.event wrapping response.output_item.done with a function_call
// item yields EventFunctionCall; SendFunctionResult then CreateResponse
// send response.item.create and response.create; and a response.event
// wrapping the Responses completion with usage (input 1000, cached 800,
// output 50) yields BackendUsage{200, 800, 50} — the uncached input
// remainder, not the backend's raw total.
func TestDelegationFunctionCallAndUsage(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	sendServerEvent(t, conn, map[string]any{
		"type":      "session.delegation.created",
		"event_id":  "e1",
		"offset_ms": 3600,
		"delegation": map[string]any{
			"id":          "del_abc123",
			"type":        "delegation",
			"target":      "responses",
			"response_id": "resp_1",
		},
	})
	delegEv := recvEvent(t, c)
	if delegEv.Kind != EventDelegation {
		t.Fatalf("delegEv.Kind = %v, want EventDelegation", delegEv.Kind)
	}
	if delegEv.Delegation.ID != "del_abc123" ||
		delegEv.Delegation.Target != "responses" ||
		delegEv.Delegation.ResponseID != "resp_1" ||
		delegEv.Delegation.OffsetMs != 3600 {
		t.Fatalf("delegEv.Delegation = %+v", delegEv.Delegation)
	}

	sendServerEvent(t, conn, map[string]any{
		"type":          "response.event",
		"event_id":      "e2",
		"delegation_id": "del_abc123",
		"event": map[string]any{
			"type": "response.output_item.done",
			"item": map[string]any{
				"type":      "function_call",
				"call_id":   "call_1",
				"name":      "light_off",
				"arguments": `{"name":"lamp"}`,
			},
		},
	})
	fcEv := recvEvent(t, c)
	if fcEv.Kind != EventFunctionCall {
		t.Fatalf("fcEv.Kind = %v, want EventFunctionCall", fcEv.Kind)
	}
	want := FunctionCall{
		DelegationID: "del_abc123",
		ResponseID:   "resp_1",
		CallID:       "call_1",
		Name:         "light_off",
		Arguments:    `{"name":"lamp"}`,
	}
	if fcEv.FunctionCall != want {
		t.Fatalf("fcEv.FunctionCall = %+v, want %+v", fcEv.FunctionCall, want)
	}

	if err := c.SendFunctionResult("call_1", `{"ok":true}`); err != nil {
		t.Fatalf("SendFunctionResult() error = %v", err)
	}
	got := readClientEvent(t, conn)
	if got["type"] != "response.item.create" {
		t.Fatalf("type = %v, want response.item.create", got["type"])
	}
	item, ok := got["item"].(map[string]any)
	if !ok || item["type"] != "function_call_output" ||
		item["call_id"] != "call_1" ||
		item["output"] != `{"ok":true}` {
		t.Fatalf("item = %v, want function_call_output for call_1", got["item"])
	}

	if err := c.CreateResponse(); err != nil {
		t.Fatalf("CreateResponse() error = %v", err)
	}
	got = readClientEvent(t, conn)
	if got["type"] != "response.create" {
		t.Fatalf("type = %v, want response.create", got["type"])
	}

	sendServerEvent(t, conn, map[string]any{
		"type":          "response.event",
		"event_id":      "e3",
		"delegation_id": "del_abc123",
		"event": map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id": "resp_1", "status": "completed",
				"usage": map[string]any{
					"input_tokens": 1000,
					"input_tokens_details": map[string]any{
						"cached_tokens": 800,
					},
					"output_tokens": 50,
					"output_tokens_details": map[string]any{
						"reasoning_tokens": 0,
					},
				},
			},
		},
	})
	doneEv := recvEvent(t, c)
	if doneEv.Kind != EventBackendResponseDone {
		t.Fatalf("doneEv.Kind = %v, want EventBackendResponseDone", doneEv.Kind)
	}
	wantUsage := BackendUsage{
		InputTokens:     200,
		CacheReadTokens: 800,
		OutputTokens:    50,
	}
	if doneEv.BackendResponse.Usage != wantUsage {
		t.Fatalf(
			"doneEv.BackendResponse.Usage = %+v, want %+v",
			doneEv.BackendResponse.Usage,
			wantUsage,
		)
	}
	if doneEv.BackendResponse.DelegationID != "del_abc123" ||
		doneEv.BackendResponse.ResponseID != "resp_1" ||
		doneEv.BackendResponse.Status != "completed" {
		t.Fatalf("doneEv.BackendResponse = %+v", doneEv.BackendResponse)
	}
}

// TestSessionUsageUpdated checks that session.usage.updated yields
// EventUsage with the reported seconds and context-window ratio.
func TestSessionUsageUpdated(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	sendServerEvent(t, conn, map[string]any{
		"type":     "session.usage.updated",
		"event_id": "e1",
		"usage": map[string]any{
			"seconds": 32.5,
		},
		"context_window": map[string]any{"usage_ratio": 0.12},
	})
	ev := recvEvent(t, c)
	if ev.Kind != EventUsage {
		t.Fatalf("ev.Kind = %v, want EventUsage", ev.Kind)
	}
	if ev.Usage.Seconds != 32.5 || !ev.Usage.HasContextWindow ||
		ev.Usage.ContextWindowRatio != 0.12 {
		t.Fatalf("ev.Usage = %+v", ev.Usage)
	}
}

// TestCloseGraceful checks that Close sends session.close, returns what
// session.closed reported, and closes the socket; and that it is
// idempotent — a second call returns the same result without hanging.
func TestCloseGraceful(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	done := make(chan struct{})
	go func() {
		defer close(done)
		got := readClientEvent(t, conn)
		if got["type"] != "session.close" {
			t.Errorf("type = %v, want session.close", got["type"])
			return
		}
		sendServerEvent(t, conn, map[string]any{
			"type":     "session.closed",
			"event_id": "e1",
			"reason":   "close_requested",
			"session": map[string]any{
				"id":     "live_test_session",
				"model":  "gpt-live-1",
				"status": "active",
			},
			"usage": map[string]any{"seconds": 45.8},
		})
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	closed, err := c.Close(ctx)
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if closed.Reason != "close_requested" || closed.Seconds != 45.8 {
		t.Fatalf(
			"Close() = %+v, want reason close_requested and seconds 45.8",
			closed,
		)
	}
	<-done

	closed2, err2 := c.Close(context.Background())
	if err2 != nil || closed2 != closed {
		t.Fatalf(
			"second Close() = %+v, %v, want the same result with no error",
			closed2,
			err2,
		)
	}
}

// TestCloseTimesOutWithoutSessionClosed checks that a server that accepts
// session.close but never answers with session.closed makes Close return at
// its deadline, wrapping context.DeadlineExceeded.
func TestCloseTimesOutWithoutSessionClosed(t *testing.T) {
	c, conn := setupReadySession(t, nil)
	_ = conn

	ctx, cancel := context.WithTimeout(
		context.Background(),
		200*time.Millisecond,
	)
	defer cancel()
	_, err := c.Close(ctx)
	if err == nil {
		t.Fatal("Close() error = nil, want a deadline error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf(
			"Close() error = %v, want it to wrap context.DeadlineExceeded",
			err,
		)
	}
}

// TestCloseDoesNotHangOnAConsumerThatStoppedReading covers the ordinary way a
// caller ends a session: it has what it wanted, stops reading Events, and
// calls Close. The server keeps streaming audio past the buffer and never
// answers session.close. Close must still return at its deadline; a reader
// blocked on a full Events channel would otherwise keep the run goroutine,
// and Close's wait for it, alive forever.
//
// Negative controls: with emit sending on c.events unconditionally, this
// test fails every run with "Close() did not return within 3s". With
// doClose doing the graceful handshake instead of CloseNow after a missed
// session.closed, it fails 2 runs in 6 the same way, since whether the
// library then waits its five seconds depends on where its own read loop
// was — this test guards the handshake fix only loosely. It sleeps 200ms
// after starting the writer goroutine to let the buffer fill, since nobody
// reads it.
func TestCloseDoesNotHangOnAConsumerThatStoppedReading(t *testing.T) {
	c, conn := setupReadySession(t, nil)
	go func() {
		for range eventBufferSize * 4 {
			err := conn.Write(
				context.Background(),
				websocket.MessageText,
				[]byte(
					`{"type":"session.output_audio.delta","event_id":"a","delta":"AAAA"}`,
				),
			)
			if err != nil {
				return
			}
		}
	}()
	time.Sleep(200 * time.Millisecond)

	returned := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(
			context.Background(),
			300*time.Millisecond,
		)
		defer cancel()
		_, err := c.Close(ctx)
		returned <- err
	}()
	select {
	case err := <-returned:
		if err == nil {
			t.Error(
				"Close() error = nil, want the deadline error: session.closed never arrived",
			)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Close() did not return within 3s")
	}
}

// TestUnknownServerEvent checks that an unknown server event becomes
// EventOther rather than an error, and that the read loop keeps going: a
// recognised event sent right after is still delivered.
func TestUnknownServerEvent(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	sendServerEvent(
		t,
		conn,
		map[string]any{"type": "session.some_future_event", "event_id": "x"},
	)
	ev := recvEvent(t, c)
	if ev.Kind != EventOther || ev.Type != "session.some_future_event" {
		t.Fatalf("ev = %+v, want EventOther for session.some_future_event", ev)
	}

	sendServerEvent(
		t,
		conn,
		map[string]any{"type": "session.input_audio.muted", "event_id": "y"},
	)
	ev = recvEvent(t, c)
	if ev.Kind != EventMuted {
		t.Fatalf("ev.Kind = %v, want EventMuted", ev.Kind)
	}
}

// TestConnectionDroppedMidSession checks that a dropped connection yields
// exactly one EventError, then a closed Events channel.
func TestConnectionDroppedMidSession(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	_ = conn.Close(websocket.StatusNormalClosure, "server going away")

	ev := recvEvent(t, c)
	if ev.Kind != EventError {
		t.Fatalf("ev.Kind = %v, want EventError", ev.Kind)
	}

	select {
	case _, ok := <-c.Events():
		if ok {
			t.Fatal(
				"Events() delivered a second event; want exactly one EventError then closed",
			)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Events() never closed after the connection dropped")
	}
}

// TestErrorEventOutOfCredit checks that an insufficient_quota error event
// matches ErrOutOfCredit.
func TestErrorEventOutOfCredit(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	sendServerEvent(t, conn, map[string]any{
		"type": "error", "event_id": "e1",
		"error": map[string]any{
			"type": "insufficient_quota_error", "code": "insufficient_quota",
			"message": "the account is out of credit",
		},
	})
	ev := recvEvent(t, c)
	if ev.Kind != EventError {
		t.Fatalf("ev.Kind = %v, want EventError", ev.Kind)
	}
	if !errors.Is(ev.Err, ErrOutOfCredit) {
		t.Fatalf("ev.Err = %v, want it to match ErrOutOfCredit", ev.Err)
	}
	if ev.Code != "insufficient_quota" {
		t.Errorf("ev.Code = %q, want insufficient_quota", ev.Code)
	}
}

// TestSendUserText checks that SendUserText sends a user message item with
// an input_text content part.
func TestSendUserText(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	if err := c.SendUserText("I need help with my order."); err != nil {
		t.Fatalf("SendUserText() error = %v", err)
	}
	got := readClientEvent(t, conn)
	if got["type"] != "response.item.create" {
		t.Fatalf("type = %v, want response.item.create", got["type"])
	}
	item, ok := got["item"].(map[string]any)
	if !ok || item["type"] != "message" || item["role"] != "user" {
		t.Fatalf("item = %v, want a user message", got["item"])
	}
	content, ok := item["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("item.content = %v, want exactly one part", item["content"])
	}
	part, ok := content[0].(map[string]any)
	if !ok || part["type"] != "input_text" ||
		part["text"] != "I need help with my order." {
		t.Fatalf("item.content[0] = %v, want an input_text part", content[0])
	}
}

// TestOversizedAppendRefused checks that an oversized append is refused
// without sending. A properly sized append with a delegation ID sent
// immediately after still works and arrives first, proving the oversized
// one above was never written to the socket at all: TCP preserves ordering,
// so a leaked write would have arrived first and failed this same
// assertion.
func TestOversizedAppendRefused(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	big := strings.Repeat("a", maxAppendBytes+1)
	if err := c.AppendInstructions(big, ""); err == nil {
		t.Fatal("AppendInstructions() error = nil, want it refused")
	}

	if err := c.AppendThinking(
		"checking availability",
		"del_abc123",
	); err != nil {
		t.Fatalf("AppendThinking() error = %v", err)
	}
	got := readClientEvent(t, conn)
	if got["type"] != "session.thinking.append" ||
		got["content"] != "checking availability" ||
		got["delegation_id"] != "del_abc123" {
		t.Fatalf("got = %v", got)
	}

	if err := c.AppendCommentary("there is a table free", ""); err != nil {
		t.Fatalf("AppendCommentary() error = %v", err)
	}
	got = readClientEvent(t, conn)
	if got["type"] != "session.commentary.append" ||
		got["delegation_id"] != nil {
		t.Fatalf("got = %v, want delegation_id null", got)
	}
}

// TestMuteUnmute checks that Mute and Unmute send the right client events
// and that the server's muted/unmuted acknowledgements yield the matching
// event kind.
func TestMuteUnmute(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	if err := c.Mute(); err != nil {
		t.Fatalf("Mute() error = %v", err)
	}
	got := readClientEvent(t, conn)
	if got["type"] != "session.input_audio.mute" {
		t.Fatalf("type = %v, want session.input_audio.mute", got["type"])
	}
	sendServerEvent(
		t,
		conn,
		map[string]any{
			"type":            "session.input_audio.muted",
			"event_id":        "e1",
			"client_event_id": got["event_id"],
		},
	)
	ev := recvEvent(t, c)
	if ev.Kind != EventMuted {
		t.Fatalf("ev.Kind = %v, want EventMuted", ev.Kind)
	}

	if err := c.Unmute(); err != nil {
		t.Fatalf("Unmute() error = %v", err)
	}
	got = readClientEvent(t, conn)
	if got["type"] != "session.input_audio.unmute" {
		t.Fatalf("type = %v, want session.input_audio.unmute", got["type"])
	}
	sendServerEvent(
		t,
		conn,
		map[string]any{
			"type":            "session.input_audio.unmuted",
			"event_id":        "e2",
			"client_event_id": got["event_id"],
		},
	)
	ev = recvEvent(t, c)
	if ev.Kind != EventUnmuted {
		t.Fatalf("ev.Kind = %v, want EventUnmuted", ev.Kind)
	}
}

// TestAppendAudio checks that AppendAudio sends session.input_audio.append
// with the PCM payload base64-encoded, unmodified.
func TestAppendAudio(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	pcm := []byte{0, 1, 2, 3, 4, 5, 6, 7}
	if err := c.AppendAudio(pcm); err != nil {
		t.Fatalf("AppendAudio() error = %v", err)
	}
	got := readClientEvent(t, conn)
	if got["type"] != "session.input_audio.append" {
		t.Fatalf("type = %v, want session.input_audio.append", got["type"])
	}
	decoded, err := base64.StdEncoding.DecodeString(got["audio"].(string))
	if err != nil || !bytes.Equal(decoded, pcm) {
		t.Fatalf("audio = %v, decode error %v, want %v", got["audio"], err, pcm)
	}
}

// TestNewValidation checks that New rejects a missing APIKey, a missing
// Model, and a Delegation with no Delegation.Model, and accepts a minimal
// valid config.
func TestNewValidation(t *testing.T) {
	if _, err := New(
		Config{Session: SessionConfig{Model: "gpt-live-1"}},
	); err == nil {
		t.Error("New() with no APIKey: error = nil, want an error")
	}
	if _, err := New(Config{APIKey: "k"}); err == nil {
		t.Error("New() with no Model: error = nil, want an error")
	}
	if _, err := New(Config{APIKey: "k", Session: SessionConfig{
		Model: "gpt-live-1", Delegation: &ResponsesDelegation{},
	}}); err == nil {
		t.Error(
			"New() with Delegation set but no Delegation.Model: error = nil, want an error",
		)
	}
	if _, err := New(
		Config{APIKey: "k", Session: SessionConfig{Model: "gpt-live-1"}},
	); err != nil {
		t.Errorf("New() with a valid config: error = %v, want nil", err)
	}
}

// TestSendFunctionResultRequiresCallID checks that SendFunctionResult
// rejects an empty call id: an unaddressed result answers the wrong call.
func TestSendFunctionResultRequiresCallID(t *testing.T) {
	c, _ := setupReadySession(t, nil)
	if err := c.SendFunctionResult("", "output"); err == nil {
		t.Error("SendFunctionResult(\"\", ...) error = nil, want an error")
	}
}

// TestSendsBeforeConnectionReturnErrNotReady checks that a send attempted
// before Start has ever dialed a connection returns ErrNotReady.
func TestSendsBeforeConnectionReturnErrNotReady(t *testing.T) {
	c, err := New(
		Config{APIKey: "k", Session: SessionConfig{Model: "gpt-live-1"}},
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := c.AppendAudio([]byte{1}); !errors.Is(err, ErrNotReady) {
		t.Errorf(
			"AppendAudio() before Start: error = %v, want ErrNotReady",
			err,
		)
	}
	if err := c.Mute(); !errors.Is(err, ErrNotReady) {
		t.Errorf("Mute() before Start: error = %v, want ErrNotReady", err)
	}
}

// TestSendsAfterCloseReturnErrClosed checks that a send attempted after
// Close returns ErrClosed. The fake server never answers session.close
// here; Close is only used to drive the client into its terminal state, so
// its own return value (a deadline error, since nothing answers) is not
// what this test checks.
func TestSendsAfterCloseReturnErrClosed(t *testing.T) {
	c, _ := setupReadySession(t, nil)
	ctx, cancel := context.WithTimeout(
		context.Background(),
		300*time.Millisecond,
	)
	defer cancel()
	_, _ = c.Close(ctx)

	if err := c.AppendAudio([]byte{1}); !errors.Is(err, ErrClosed) {
		t.Errorf("AppendAudio() after Close: error = %v, want ErrClosed", err)
	}
	if err := c.CreateResponse(); !errors.Is(err, ErrClosed) {
		t.Errorf(
			"CreateResponse() after Close: error = %v, want ErrClosed",
			err,
		)
	}
}

// TestDialRejected checks that a dial refused at the handshake with HTTP
// 401 makes WaitReady and the emitted EventError both match ErrUnauthorized,
// and that the Events channel then closes.
func TestDialRejected(t *testing.T) {
	srv := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}),
	)
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")

	c, err := New(
		Config{
			APIKey:  "bad-key",
			BaseURL: wsURL,
			Session: SessionConfig{Model: "gpt-live-1"},
		},
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	c.Start(ctx)

	if err := c.WaitReady(ctx); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("WaitReady() error = %v, want ErrUnauthorized", err)
	}
	ev := recvEvent(t, c)
	if ev.Kind != EventError || !errors.Is(ev.Err, ErrUnauthorized) {
		t.Fatalf("ev = %+v, want EventError matching ErrUnauthorized", ev)
	}
	if _, ok := <-c.Events(); ok {
		t.Fatal(
			"Events() delivered a second event; want exactly one EventError then closed",
		)
	}
}

// TestErrorEventUnauthorized checks that an in-session authentication_error
// event, not just a rejected dial, also matches ErrUnauthorized.
func TestErrorEventUnauthorized(t *testing.T) {
	c, conn := setupReadySession(t, nil)

	sendServerEvent(t, conn, map[string]any{
		"type":     "error",
		"event_id": "e1",
		"error": map[string]any{
			"type":    "authentication_error",
			"code":    "invalid_api_key",
			"message": "bad key",
		},
	})
	ev := recvEvent(t, c)
	if ev.Kind != EventError {
		t.Fatalf("ev.Kind = %v, want EventError", ev.Kind)
	}
	if !errors.Is(ev.Err, ErrUnauthorized) {
		t.Fatalf("ev.Err = %v, want it to match ErrUnauthorized", ev.Err)
	}
}

// TestSessionStartInputHistory checks that seed history renders
// developer/user messages as input_text content and assistant messages as
// output_text.
func TestSessionStartInputHistory(t *testing.T) {
	_, conn, _ := setupClient(t, func(cfg *Config) {
		cfg.Session.Input = []InputMessage{
			{Role: "user", Text: "I need help with my recent order."},
			{Role: "assistant", Text: "What is the order number?"},
		}
	})
	got := readClientEvent(t, conn)
	session, ok := got["session"].(map[string]any)
	if !ok {
		t.Fatalf("session is not an object: %v", got["session"])
	}
	input, ok := session["input"].([]any)
	if !ok || len(input) != 2 {
		t.Fatalf("session.input = %v, want two messages", session["input"])
	}

	first, ok := input[0].(map[string]any)
	if !ok || first["role"] != "user" {
		t.Fatalf("input[0] = %v, want role user", input[0])
	}
	firstParts, ok := first["content"].([]any)
	if !ok || len(firstParts) != 1 {
		t.Fatalf("input[0].content = %v", first["content"])
	}
	firstPart := firstParts[0].(map[string]any)
	if firstPart["type"] != "input_text" ||
		firstPart["text"] != "I need help with my recent order." {
		t.Errorf("input[0].content[0] = %v, want an input_text part", firstPart)
	}

	second, ok := input[1].(map[string]any)
	if !ok || second["role"] != "assistant" {
		t.Fatalf("input[1] = %v, want role assistant", input[1])
	}
	secondParts, ok := second["content"].([]any)
	if !ok || len(secondParts) != 1 {
		t.Fatalf("input[1].content = %v", second["content"])
	}
	secondPart := secondParts[0].(map[string]any)
	if secondPart["type"] != "output_text" ||
		secondPart["text"] != "What is the order number?" {
		t.Errorf(
			"input[1].content[0] = %v, want an output_text part",
			secondPart,
		)
	}
}

// TestEventKindString checks that EventKind.String names every declared
// kind, and falls back to "kind(N)" for an unknown one.
func TestEventKindString(t *testing.T) {
	for k := EventSessionStarted; k <= EventOther; k++ {
		if got := k.String(); got == "" || strings.HasPrefix(got, "kind(") {
			t.Errorf(
				"EventKind(%d).String() = %q, want a real name",
				int(k),
				got,
			)
		}
	}
	if got := EventKind(999).String(); got != "kind(999)" {
		t.Errorf("EventKind(999).String() = %q, want kind(999)", got)
	}
}
