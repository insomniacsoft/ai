package llm

// Model represents a Large Language Model with its configuration and
// capabilities.
//
// Each LLM provider package publishes its own catalog of these under the name
// Models, so a configuration is selected as, for example,
// openai.Models[openai.GPT4o]. A model that is not in a catalog can still be
// used by constructing one directly or via [NewCustomModel].
type Model struct {
	// ID is the unique identifier for this model within the library.
	ID string `json:"id"`
	// Name is the human-readable name of the model.
	Name string `json:"name"`
	// Provider identifies which AI service provides this model.
	Provider string `json:"provider"`
	// APIModel is the model identifier used in API requests.
	APIModel string `json:"api_model"`
	// Currency is the ISO 4217 code the Cost fields are denominated in, for
	// example "USD" or "EUR". An empty value means "USD".
	Currency string `json:"currency"`
	// CostPer1MIn is the cost per 1 million input tokens, in Currency.
	CostPer1MIn float64 `json:"cost_per_1m_in"`
	// CostPer1MOut is the cost per 1 million output tokens, in Currency.
	CostPer1MOut float64 `json:"cost_per_1m_out"`
	// CostPer1MInCached is the cost per 1 million cached input tokens, in
	// Currency.
	CostPer1MInCached float64 `json:"cost_per_1m_in_cached"`
	// CostPer1MOutCached is the cost per 1 million cached output tokens, in
	// Currency.
	CostPer1MOutCached float64 `json:"cost_per_1m_out_cached"`
	// ContextWindow is the maximum number of tokens the model can process.
	ContextWindow int64 `json:"context_window"`
	// DefaultMaxTokens is the recommended maximum tokens for responses.
	DefaultMaxTokens int64 `json:"default_max_tokens"`
	// CanReason indicates if the model supports chain-of-thought reasoning.
	CanReason bool `json:"can_reason"`
	// SupportsAttachments indicates if the model can process images and files.
	SupportsAttachments bool `json:"supports_attachments"`
	// SupportsStructuredOut indicates if the model supports structured JSON output.
	SupportsStructuredOut bool `json:"supports_structured_output"`
	// SupportsImageGeneration indicates if the model can generate images.
	SupportsImageGeneration bool `json:"supports_image_generation"`

	// State is what the provider says about this model's life: "active",
	// "deprecated", or empty where the provider publishes nothing.
	//
	// Carried because a catalog that lists a deprecated model exactly like a
	// current one leaves every consumer to guess, and the guesses are worse
	// than the fact. The alternative -- dropping deprecated models at
	// generation time -- would break the catalog's other job: a ledger has to
	// be able to price a model somebody is still being billed for, including
	// one retired last week. Both jobs are served by carrying the field and
	// letting each caller decide.
	State string `json:"state,omitempty"`

	// ReleaseDate is when the provider published this model, as YYYY-MM-DD.
	// Empty where the provider publishes no date -- which is common for
	// aliases and for older entries, so an ordering built on it must have an
	// answer for the empty case rather than sorting those to one end by
	// accident.
	ReleaseDate string `json:"release_date,omitempty"`

	// LastUpdated is when the provider last changed anything it publishes
	// about this model, as YYYY-MM-DD.
	//
	// It is a different fact from ReleaseDate and is carried alongside it
	// rather than folded into it. A floating alias is released once and then
	// repointed at snapshot after snapshot, so its release date says when the
	// NAME first appeared and this says when what the name currently resolves
	// to last moved. The source publishes one, the other, both, or neither --
	// thirty-one of the sixty-seven OpenAI chat entries carry this and not a
	// release date -- so a consumer ordering by recency needs both fields and
	// a stated rule for combining them, which is exactly why neither is
	// synthesised here.
	LastUpdated string `json:"last_updated,omitempty"`

	// RetirementDate is when the provider stops serving this model, as
	// YYYY-MM-DD, and is set only for a model with a published end. A caller
	// pointing somebody at a model can say how long it has left instead of
	// discovering it on the morning the calls start failing.
	RetirementDate string `json:"retirement_date,omitempty"`

	// ReplacedBy is the model the provider recommends instead, for a
	// deprecated entry. It is the provider's own recommendation and not a
	// judgement made here.
	ReplacedBy string `json:"replaced_by,omitempty"`
}

// Deprecated reports whether the provider has marked this model deprecated.
//
// A method rather than a comparison at each call site: "deprecated" is one
// spelling out of the provider's own vocabulary, and a caller writing the
// string itself is a caller that keeps working when the vocabulary grows a
// second retired state and silently starts offering those models again.
func (m Model) Deprecated() bool { return m.State == "deprecated" }
