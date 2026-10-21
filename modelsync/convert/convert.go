// Package convert turns model-sync entries into the catalog types of this
// repository's llm, tts and hostedtool modules.
//
// It is a module of its own so the modelsync parser, and the generator that
// is built on it, depend on the standard library alone: a catalog that does
// not compile must not stop the tool that regenerates it.
package convert

import (
	"github.com/joakimcarlsson/ai/hostedtool"
	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/modelsync"
	"github.com/joakimcarlsson/ai/tts"
)

// The converters build the catalog types from an entry. Each keys the result
// by the entry's API ID: ID and APIModel both hold [modelsync.Entry.APIID], and
// Provider holds the enclosing provider, the one the model is billed through.
// They do not check the entry's kind; the caller picks the entries.
//
// A rate the source does not publish is left at zero, as in a compiled-in
// catalog. A caller that must tell "free" from "unpublished" reads the rates
// from [modelsync.Entry.ChatRates] and its siblings, which say so.

// ToLLM converts a chat entry to an llm.Model.
//
// Cache-write pricing has no field in llm.Model and is not folded into
// CostPer1MOutCached, which prices cached output; read it from
// [modelsync.Entry.ChatRates]. DefaultMaxTokens is the published output
// limit, else a quarter of the context window capped at 8192, the same
// default the generator seeds a new entry with.
//
// A capability the entry does not publish is false here, because llm.Model
// has no unknown state. A caller that must not exclude a model for a missing
// fact asks [modelsync.Entry.CanReason] and
// [modelsync.Entry.SupportsAttachments], which say whether it was published.
func ToLLM(e modelsync.Entry) llm.Model {
	r := e.ChatRates()
	window := e.ContextWindow()

	maxTokens := e.MaxOutputTokens()
	if maxTokens == 0 && window > 0 {
		maxTokens = min(window/4, 8192)
	}

	reasons, _ := e.CanReason()
	attachments, _ := e.SupportsAttachments()

	id := e.APIID()
	return llm.Model{
		ID:                      id,
		Name:                    e.Name,
		Provider:                e.EnclosingProvider,
		APIModel:                id,
		Currency:                e.Currency(),
		CostPer1MIn:             r.Input.Per,
		CostPer1MOut:            r.Output.Per,
		CostPer1MInCached:       r.CacheRead.Per,
		ContextWindow:           window,
		DefaultMaxTokens:        maxTokens,
		CanReason:               reasons,
		SupportsAttachments:     attachments,
		SupportsStructuredOut:   e.Feature("structured_outputs"),
		SupportsImageGeneration: e.EmitsModality("image"),
		State:                   e.Attrs["state"],
		ReleaseDate:             e.Attrs["release_date"],
		LastUpdated:             e.Attrs["last_updated"],
		RetirementDate:          e.Attrs["retirement_date"],
		ReplacedBy:              e.Attrs["recommended_replacement"],
	}
}

// ToTTS converts a speech entry to a tts.AudioModel.
func ToTTS(e modelsync.Entry) tts.AudioModel {
	id := e.APIID()
	return tts.AudioModel{
		ID:                id,
		Name:              e.Name,
		Provider:          e.EnclosingProvider,
		APIModel:          id,
		Currency:          e.Currency(),
		CostPer1MChars:    e.SpeechRate().Per,
		MaxCharacters:     e.Limit("character_limit"),
		SupportedFormats:  e.Lists["output_formats"],
		SupportsStreaming: e.Feature("streaming"),
	}
}

// ToHostedTool converts a hosted-tool entry to a hostedtool.Tool.
func ToHostedTool(e modelsync.Entry) hostedtool.Tool {
	id := e.APIID()
	return hostedtool.Tool{
		ID:             id,
		Name:           e.Name,
		Provider:       e.EnclosingProvider,
		APIModel:       id,
		Currency:       e.Currency(),
		CostPer1KCalls: e.ToolRate().Per,
	}
}
