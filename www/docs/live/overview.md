# Live

The `live` module is a client for OpenAI's GPT-Live sessions
(`/v1/live/sessions`), over one WebSocket per conversation. GPT-Live is not
a newer Realtime model — it runs the spoken conversation on its own
endpoint, with its own event protocol, and hands reasoning and tool use to
a backend it delegates to. The `realtime` client cannot speak to it, so
`live` is a separate module that does not import `realtime` and stands on
its own.

## Quick start

```go
import "github.com/joakimcarlsson/ai/live"

c, err := live.New(live.Config{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Session: live.SessionConfig{
        Model:        "gpt-live-1",
        Instructions: "You are a concise voice assistant.",
        Delegation: &live.ResponsesDelegation{
            Model:        "gpt-6-luna",
            Instructions: "Backend business rules go here.",
            Tools:        []tool.BaseTool{myTool},
        },
    },
})
if err != nil {
    log.Fatal(err)
}
defer c.Close(context.Background())

c.Start(ctx)
if err := c.WaitReady(ctx); err != nil {
    log.Fatal(err)
}

go func() {
    for ev := range c.Events() {
        switch ev.Kind {
        case live.EventAudio:
            // ev.Audio is PCM16 at live.SampleRateHz, little-endian.
        case live.EventFunctionCall:
            // ev.FunctionCall names the backend call; answer with
            // SendFunctionResult.
        }
    }
}()

c.AppendAudio(microphonePCM)
```

## Delegation

`SessionConfig.Delegation` chooses who answers a request the model decides
needs more than conversation: a `*ResponsesDelegation` names a backend
Responses model, its own instructions (kept separate from the conversational
`Instructions`), and the function tools the application runs — GPT-Live
never executes a tool itself, it delegates the call and waits for
`SendFunctionResult`. Leaving `Delegation` nil selects client delegation
instead, the API's own default, where the application answers delegated
work directly with `SendFunctionResult` and `CreateResponse`.

`Client.AppendInstructions`, `AppendThinking` and `AppendCommentary` append
to the three context channels a delegation can extend mid-turn, each capped
at a conservative byte approximation of the API's 500-token limit.

## Backend usage

A delegation's `EventDelegation` and function-call events carry
`BackendUsage`, whose input count is already the uncached remainder rather
than the backend's raw total, so a cached token is never billed twice
against the same conversation.

## Closing

`Close` asks the session to end gracefully: it sends `session.close`, waits
(bounded by the caller's context) for `session.closed`, and closes the
socket either way. It never hangs on a caller that has stopped reading
`Events` — delivery gives up once the session is being torn down — and a
close that never received `session.closed` drops the socket immediately
rather than attempting a handshake with a peer that is not answering.

There is no reconnect: a Live session cannot be resumed, so a dropped
connection is reported once, as a single `EventError`, and ends the session.
