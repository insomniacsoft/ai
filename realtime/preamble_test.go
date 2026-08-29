package realtime

import (
	"testing"
	"time"
)

// pushCall delivers one function_call_arguments.done for response resp_1.
func pushCall(t *testing.T, conn *fakeConn, callID string) {
	t.Helper()
	conn.push(t, map[string]any{
		"type": evFuncArgsDone, "response_id": "resp_1",
		"item_id": "item_call", "call_id": callID,
		"name": "get_weather", "arguments": `{"city":"x"}`,
	})
}

// finishResp1 delivers response.done for resp_1 and returns the event.
func finishResp1(t *testing.T, c *Client, conn *fakeConn) Event {
	t.Helper()
	conn.push(t, map[string]any{
		"type":     evResponseDone,
		"response": map[string]any{"id": "resp_1", "status": "completed"},
	})
	return drainKind(t, c, EventResponseDone)
}

// TestPreambleCountsOnlyAudioThatPrecededTheCall checks that the preamble
// counter latches on the first call rather than summing the response's
// whole audio: the latency argument for deferring tool schemas rests on the
// model speaking before it reaches for a tool, and audio that arrives after
// the call is not preamble. It pushes 300 ms of audio before the call and
// 500 ms after.
func TestPreambleCountsOnlyAudioThatPrecededTheCall(t *testing.T) {
	conn := newFakeConn()
	c := startClient(t, []*fakeConn{conn})
	ready(t, c, conn)

	pushAudio(t, c, conn, "item_audio", 3)
	pushCall(t, conn, "call_1")
	pushAudio(t, c, conn, "item_audio", 5)

	ev := finishResp1(t, c, conn)
	if len(ev.Calls) != 1 {
		t.Fatalf("expected one call, got %d", len(ev.Calls))
	}
	got := ev.Preamble.Round(10 * time.Millisecond)
	if got != 300*time.Millisecond {
		t.Fatalf(
			"preamble = %v, want 300ms — audio after the call must not count",
			got,
		)
	}
}

// TestPreambleIsZeroWhenTheCallCameFirst.
//
// The failure case the measurement exists to catch. A response that calls a
// tool before saying anything leaves the listener hearing silence for
// the length of the round trip, and that is the condition under which
// progressive disclosure must not ship.
func TestPreambleIsZeroWhenTheCallCameFirst(t *testing.T) {
	conn := newFakeConn()
	c := startClient(t, []*fakeConn{conn})
	ready(t, c, conn)

	pushCall(t, conn, "call_1")
	pushAudio(t, c, conn, "item_audio", 4)

	ev := finishResp1(t, c, conn)
	if ev.Preamble != 0 {
		t.Fatalf(
			"preamble = %v, want 0 when the call preceded all audio",
			ev.Preamble,
		)
	}
}

// TestPreambleDoesNotReportAConstant.
//
// Negative control. A field wired to a fixed value, or to the response's whole
// audio, would satisfy either test above on its own.
func TestPreambleDoesNotReportAConstant(t *testing.T) {
	measure := func(before int) time.Duration {
		conn := newFakeConn()
		c := startClient(t, []*fakeConn{conn})
		ready(t, c, conn)
		pushAudio(t, c, conn, "item_audio", before)
		pushCall(t, conn, "call_1")
		return finishResp1(t, c, conn).Preamble
	}
	short, long := measure(1), measure(6)
	if short >= long {
		t.Fatalf(
			"one chunk of preamble (%v) did not measure below six (%v)",
			short,
			long,
		)
	}
}

// TestPreambleSurvivesAResponseAssembledOnlyFromItsOutput.
//
// A reconnect mid-response, or a provider that batches, delivers the call in
// response.done's output rather than as an incremental event. No incremental
// audio was seen either, so the honest answer is zero rather than a number
// invented from the output array.
func TestPreambleSurvivesAResponseAssembledOnlyFromItsOutput(t *testing.T) {
	conn := newFakeConn()
	c := startClient(t, []*fakeConn{conn})
	ready(t, c, conn)

	conn.push(t, map[string]any{
		"type": evResponseDone,
		"response": map[string]any{
			"id": "resp_1", "status": "completed",
			"output": []any{map[string]any{
				"type": "function_call", "id": "item_call",
				"call_id": "call_1", "name": "get_weather", "arguments": "{}",
			}},
		},
	})
	ev := drainKind(t, c, EventResponseDone)
	if len(ev.Calls) != 1 {
		t.Fatalf(
			"expected the call from the output array, got %d",
			len(ev.Calls),
		)
	}
	if ev.Preamble != 0 {
		t.Fatalf(
			"preamble = %v, want 0 when no audio was ever streamed",
			ev.Preamble,
		)
	}
}
