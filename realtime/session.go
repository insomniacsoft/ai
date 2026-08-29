package realtime

import (
	"fmt"
	"time"

	"github.com/joakimcarlsson/ai/tool"
	"github.com/openai/openai-go/v3/packages/param"
	oai "github.com/openai/openai-go/v3/realtime"
)

// SampleRateHz is the only rate the Realtime API's PCM format accepts, and
// the SDK's own type documents it as "Always 24000". Bridging it to whatever
// rate the downstream consumer wants is the caller's job.
const SampleRateHz = 24000

// Eagerness values for semantic turn detection. The provider documents max
// wait times of 8 s, 4 s and 2 s for low, medium and high; `auto` is medium.
// Operator-configurable rather than fixed because a speaker who pauses
// mid-sentence and one who fires commands want opposite settings.
const (
	EagernessLow    = "low"
	EagernessMedium = "medium"
	EagernessHigh   = "high"
	EagernessAuto   = "auto"
)

// SessionConfig is everything the caller decides about a session before it opens.
//
// The session opens on the wake event, while the human is still speaking, so
// everything here has to be known before a single word has been transcribed
// — which is precisely why domain terms and memory are rendered into
// Instructions rather than fetched on demand mid-turn.
type SessionConfig struct {
	// Model is the Realtime model id. No default is baked in: a hardcoded
	// model id is a config value pretending to be a constant, and it ages
	// into an outage the day it is retired.
	Model string

	// Instructions is the whole system prompt: persona, domain-specific
	// terms the assistant needs to recognize, and a memory digest. The
	// provider ships its OWN default instructions and applies them until
	// replaced — measured against the live API — so leaving this empty does
	// not yield a neutral assistant, it yields somebody else's.
	Instructions string

	// Voice is the output voice id. Empty leaves the provider's default.
	Voice string

	// Eagerness tunes semantic turn detection. Empty means EagernessAuto.
	Eagerness string

	// Tools is the GUARDED toolset — the output of guard.BuildToolset, never
	// a raw registry. This type takes tool.BaseTool precisely so an
	// unguarded []agent.Tool cannot be passed by accident.
	Tools []tool.BaseTool

	// MaxOutputTokens caps a single response. Zero leaves the provider's own
	// default, which is unlimited.
	//
	// This is a hard stop, not guidance: an instruction to be brief is
	// something the model can talk itself out of mid-sentence, and in a spoken
	// channel the listener pays for every extra clause in seconds of waiting.
	MaxOutputTokens int

	// RetentionRatio is the fraction of POST-INSTRUCTION conversation tokens
	// the provider keeps when a session outgrows its input window. Zero leaves
	// the provider's default of 1.0.
	//
	// The default is the expensive one. At 1.0 the conversation is trimmed to
	// exactly the limit, so the very next turn exceeds it again and truncates
	// again — and every truncation rewrites the cached prefix's neighbourhood
	// and costs the whole prefix at the uncached rate. A ratio below 1.0 drops
	// a block of old turns at once and leaves headroom, so truncation happens
	// rarely instead of continuously. The SDK's own note on the field says as
	// much: it "helps reduce the frequency of truncations and improve cache
	// rates."
	//
	// Instructions and tool schemas are never what gets dropped — the ratio
	// applies to the conversation AFTER them — so this trades a little
	// conversational memory for a prefix that stays cached.
	RetentionRatio float64

	// TranscribeInput asks the provider to transcribe what the human said.
	// Off by default: it is a second billed model over the same audio, and
	// only the caller's own transcript policy can decide whether those words
	// are wanted at all.
	TranscribeInput bool

	// TranscriptionLanguage is an ISO-639-1 hint for the transcriber. Optional,
	// and only a hint: it improves accuracy and latency where one language is
	// spoken, and does not stop the transcriber recognising another one.
	TranscriptionLanguage string

	// TranscriptionPrompt is free text steering the transcriber — the only
	// place more than one expected language can be declared, since Language
	// above takes exactly one code. Supported by the gpt-4o-*-transcribe
	// family; the API documents it as unsupported for gpt-realtime-whisper in
	// GA sessions, so a caller switching model must revisit this.
	TranscriptionPrompt string

	// TranscriptionModel names the model used when TranscribeInput is set.
	TranscriptionModel string
}

// Validate rejects a config that cannot open a usable session.
func (c SessionConfig) Validate() error {
	if c.Model == "" {
		return fmt.Errorf("realtime: SessionConfig.Model is required")
	}
	if c.Instructions == "" {
		return fmt.Errorf("realtime: SessionConfig.Instructions is required; " +
			"an empty value leaves the provider's own default persona in place, not a neutral one")
	}
	switch c.Eagerness {
	case "", EagernessLow, EagernessMedium, EagernessHigh, EagernessAuto:
	default:
		return fmt.Errorf("realtime: SessionConfig.Eagerness %q is not one of low/medium/high/auto", c.Eagerness)
	}
	if c.TranscribeInput && c.TranscriptionModel == "" {
		return fmt.Errorf("realtime: SessionConfig.TranscriptionModel is required when TranscribeInput is set")
	}
	return nil
}

// sessionParams renders the config into the SDK's generated session type.
//
// The session-configuration surface is the highest-drift-risk part of this
// integration and the SDK tracks it for free, so it is the one thing here
// that is NOT hand-rolled. Measured 2026-08-23: the server echoes this
// payload back unchanged in session.updated.
func (c SessionConfig) sessionParams() (oai.RealtimeSessionCreateRequestParam, error) {
	tools, err := realtimeTools(c.Tools)
	if err != nil {
		return oai.RealtimeSessionCreateRequestParam{}, err
	}

	eagerness := c.Eagerness
	if eagerness == "" {
		eagerness = EagernessAuto
	}
	pcm := func() oai.RealtimeAudioFormatsUnionParam {
		return oai.RealtimeAudioFormatsUnionParam{
			OfAudioPCM: &oai.RealtimeAudioFormatsAudioPCMParam{Rate: SampleRateHz, Type: "audio/pcm"},
		}
	}

	in := oai.RealtimeAudioConfigInputParam{
		Format: pcm(),
		// Turn detection defaults to server_vad, so this is a REPLACEMENT
		// and not a refinement — a session that omits it silently gets
		// threshold VAD, and semantic turn detection is quietly
		// unimplemented.
		TurnDetection: oai.RealtimeAudioInputTurnDetectionUnionParam{
			OfSemanticVad: &oai.RealtimeAudioInputTurnDetectionSemanticVadParam{Eagerness: eagerness},
		},
	}
	if c.TranscribeInput {
		in.Transcription = oai.AudioTranscriptionParam{Model: oai.AudioTranscriptionModel(c.TranscriptionModel)}
		if c.TranscriptionPrompt != "" {
			in.Transcription.Prompt = param.NewOpt(c.TranscriptionPrompt)
		}
		if c.TranscriptionLanguage != "" {
			in.Transcription.Language = param.NewOpt(c.TranscriptionLanguage)
		}
	}

	out := oai.RealtimeAudioConfigOutputParam{Format: pcm()}
	if c.Voice != "" {
		out.Voice = oai.RealtimeAudioConfigOutputVoiceUnionParam{OfString: param.NewOpt(c.Voice)}
	}

	params := oai.RealtimeSessionCreateRequestParam{
		Model:            c.Model,
		Instructions:     param.NewOpt(c.Instructions),
		OutputModalities: []string{"audio"},
		Audio:            oai.RealtimeAudioConfigParam{Input: in, Output: out},
		Tools:            tools,
	}
	if c.MaxOutputTokens > 0 {
		params.MaxOutputTokens = oai.RealtimeSessionCreateRequestMaxOutputTokensUnionParam{
			OfInt: param.NewOpt(int64(c.MaxOutputTokens)),
		}
	}
	if c.RetentionRatio > 0 {
		params.Truncation = oai.RealtimeTruncationUnionParam{
			OfRetentionRatioTruncation: &oai.RealtimeTruncationRetentionRatioParam{
				RetentionRatio: c.RetentionRatio,
			},
		}
	}
	return params, nil
}

// realtimeTools converts the guarded toolset into the provider's function
// shape.
//
// tool.Info splits a JSON Schema in two: Parameters holds the PROPERTIES map
// alone, and Required is a sibling slice. The provider wants one whole schema
// object, so this reassembles it. A tool whose properties are nil still gets
// an explicit empty object — omitting `properties` entirely describes a
// different (unconstrained) schema, and a no-argument tool that appears to
// accept anything invites the model to invent arguments for it.
func realtimeTools(tools []tool.BaseTool) (oai.RealtimeToolsConfigParam, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	out := make(oai.RealtimeToolsConfigParam, 0, len(tools))
	seen := make(map[string]bool, len(tools))
	for _, t := range tools {
		info := t.Info()
		if info.Name == "" {
			return nil, fmt.Errorf("realtime: a tool has no name; the model cannot address it")
		}
		// Two tools with one name is not a cosmetic problem: the provider
		// answers with a name, so a duplicate makes the call ambiguous and
		// the wrong tool runs. Refuse at build time, where it is a config
		// error, rather than at call time, where it is an action.
		if seen[info.Name] {
			return nil, fmt.Errorf("realtime: duplicate tool name %q; a call to it could not be routed", info.Name)
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
		out = append(out, oai.RealtimeToolsConfigUnionParam{
			OfFunction: &oai.RealtimeFunctionToolParam{
				Type:        "function",
				Name:        param.NewOpt(info.Name),
				Description: param.NewOpt(info.Description),
				Parameters:  schema,
			},
		})
	}
	return out, nil
}

// SessionInfo is what the provider says about the live session.
type SessionInfo struct {
	ID        string
	Model     string
	ExpiresAt time.Time
}

// TimeToExpiry reports how long the provider says this session has left.
//
// Measured 2026-08-23: expires_at is creation + 3600 s exactly on this model
// and tier. It is reported for logging and for deciding whether a new turn is
// worth starting on this socket — NOT as a timer to reconnect against. An
// announced cap can change without notice, so the reconnect fires on the
// expiry the server actually reports (see errSessionExpired), which cannot be
// wrong about it.
func (s SessionInfo) TimeToExpiry(now time.Time) time.Duration {
	if s.ExpiresAt.IsZero() {
		return 0
	}
	return s.ExpiresAt.Sub(now)
}
