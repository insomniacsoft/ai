# Realtime

The `realtime` module is a client for the OpenAI Realtime API: one
speech-to-speech session per voice conversation, over a single reconnecting
WebSocket, carrying whatever toolset the caller hands it.

It is not a text-message transport and not an `llm.LLM`. Both seams are the
wrong shape — a text transport pays for two model turns to carry voice
through it, and `llm.LLM` is stateless-per-call with no audio, so reaching it
requires transcribing first and throws away everything voice is for. Session
policy — when to open, when to reconnect, what to say — belongs to the
caller; this package is the protocol and nothing else.

## Quick start

```go
import "github.com/joakimcarlsson/ai/realtime"

c, err := realtime.New(realtime.Config{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Session: realtime.SessionConfig{
        Model:        "gpt-realtime",
        Instructions: "You are a concise voice assistant.",
        Voice:        "alloy",
        Tools:        []tool.BaseTool{myTool},
    },
})
if err != nil {
    log.Fatal(err)
}
defer c.Close()

c.Start(ctx)
if err := c.WaitReady(ctx); err != nil {
    log.Fatal(err)
}

go func() {
    for ev := range c.Events() {
        switch ev.Kind {
        case realtime.EventAudio:
            // ev.Audio is 24 kHz mono s16 little-endian PCM.
        case realtime.EventTranscript:
            fmt.Println(ev.Transcript)
        case realtime.EventResponseDone:
            // ev.Calls carries any tool calls, correlated with ev.Transcript.
        }
    }
}()

c.AppendAudio(microphonePCM)
```

## Session configuration

| Field | Description | Default |
|---|---|---|
| `Model` | The Realtime model id | required |
| `Instructions` | The whole system prompt | required |
| `Voice` | Output voice id | provider default |
| `Eagerness` | Semantic turn-detection eagerness (`low`/`medium`/`high`/`auto`) | `auto` |
| `Tools` | The toolset offered to the model | none |
| `MaxOutputTokens` | Hard cap on a single response | provider default (unlimited) |
| `RetentionRatio` | Fraction of post-instruction conversation kept when the input window overflows | provider default (`1.0`) |
| `TranscribeInput` | Ask the provider to also transcribe what the human said | off |
| `TranscriptionLanguage` | ISO-639-1 hint for the transcriber | none |
| `TranscriptionPrompt` | Free text steering the transcriber | none |
| `TranscriptionModel` | Model used when `TranscribeInput` is set | required with `TranscribeInput` |

## Playback-aware truncation

`Client.Truncate` and `Client.BargeIn` cut an assistant audio item's context
to what the listener actually heard, not to what this client sent. That
figure has to come from a `PlaybackReporter` the caller supplies — the local
clock and the playback device drift against each other, so a position
computed from a timer rather than reported by the device eventually misplaces
a word permanently. `HeardPosition` returns `ErrNoPlaybackReport` rather than
falling back to a clock when no reporter is wired or it has nothing current
to say.

## Reconnection and history replay

A session ends at the provider's own duration cap; the client reconnects at
once rather than backing off, and reports `ErrSessionExpired` so the caller
can replay a bounded window of recent conversation with `ReplayHistory`.
Replayed turns are sent as conversation items, never folded into the session
instructions — instructions are trusted and conversation is not.

## Cost accounting

`Usage.Billable` decomposes a response's reported token usage into the
classes an invoice actually charges for (text/audio/image input, cached or
not, plus text/audio output), clamping negative deltas at zero.
`TextCacheHitRate` reports the share of text input served from the
provider's prefix cache.
