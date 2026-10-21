// Package modelsync reads the model-sync api.json document and turns what it
// publishes into the catalog types the rest of this repository uses.
//
// https://github.com/JoakimCarlsson/model-sync publishes one api.json
// describing every provider and every model it serves. Two programs read it
// and must agree on what it says: the cmd/modelsync generator, which writes
// the compiled-in catalogs, and an application that refreshes its catalog at
// run time. Both go through this package, so a rate is selected by one piece
// of code. A second mapping would drift from the first, and a freshly
// fetched catalog would report price changes that are only the two mappings
// disagreeing.
//
// The package keeps everything the document says about a model. Anything
// beyond the identity fields sits in open maps, so a key the source starts
// publishing is readable without a change here.
package modelsync

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

// Price is one published rate. The same metric appears many times per model,
// once per priced combination of tier, region, deployment and the rest, which
// Dims names.
type Price struct {
	Metric   string            `json:"metric"`
	Unit     string            `json:"unit"`
	Amount   float64           `json:"amount"`
	Currency string            `json:"currency"`
	Dims     map[string]string `json:"dims"`
}

// Entry is one model as api.json describes it, with the provider it was filed
// under.
//
// The identity a request names the model with is [Entry.APIID], not ID: ID is
// the key api.json files the model under, and the two differ for some
// models.
type Entry struct {
	// EnclosingProvider is the id of the provider object the model was filed
	// under in api.json. It is set by the parser, never by the document.
	EnclosingProvider string `json:"-"`
	// EnclosingProviderName is that provider object's display name.
	EnclosingProviderName string `json:"-"`

	// ID is the key api.json files the model under.
	ID string `json:"id"`
	// Provider is the provider the entry itself names. It is not always the
	// enclosing one: an aggregator files models other providers make under
	// its own object. Empty when the entry names none.
	Provider string `json:"provider"`
	Name     string `json:"name"`
	Kind     string `json:"kind"`

	Prices []Price `json:"prices"`
	// Attrs holds the single-valued facts: state, release_date,
	// last_updated, retirement_date, retirement_date_qualifier,
	// default_snapshot, api_id, thinking_default, summary and whatever the
	// source adds next.
	Attrs map[string]string `json:"attrs"`
	// Limits holds the numeric bounds: context_window, max_input_tokens,
	// max_output_tokens and the like.
	Limits map[string]int64 `json:"limits"`
	// Lists holds the multi-valued facts: endpoints, features,
	// input_modalities, output_modalities, tools, snapshots, aliases,
	// effort_levels, thinking_types and the like.
	Lists map[string][]string `json:"lists"`
	// Sources lists the pages the source read the entry from.
	Sources []string `json:"sources"`
}

// APIID is the identifier a request names the model with, which is not always
// the catalog key api.json files the model under: attrs.api_id when the
// source publishes one, else ID. It is the canonical model ID everywhere a
// model is named: in a setting, on the wire, in a ledger and in a rate lookup.
func (e Entry) APIID() string {
	if id := e.Attrs["api_id"]; id != "" {
		return id
	}
	return e.ID
}

// Feature reports whether the model lists name among its features.
func (e Entry) Feature(name string) bool {
	return slices.Contains(e.Lists["features"], name)
}

// AnyFeature reports whether the model lists any of the names as a feature.
func (e Entry) AnyFeature(names ...string) bool {
	for _, n := range names {
		if e.Feature(n) {
			return true
		}
	}
	return false
}

// TakesModality reports whether the model accepts any of the named input
// modalities.
func (e Entry) TakesModality(names ...string) bool {
	for _, n := range names {
		if slices.Contains(e.Lists["input_modalities"], n) {
			return true
		}
	}
	return false
}

// EmitsModality reports whether the model produces the named output modality.
func (e Entry) EmitsModality(name string) bool {
	return slices.Contains(e.Lists["output_modalities"], name)
}

// List returns the first of the named lists the model publishes, so a caller
// can state a preference order between equivalent keys.
func (e Entry) List(keys ...string) []string {
	for _, k := range keys {
		if v := e.Lists[k]; len(v) > 0 {
			return v
		}
	}
	return nil
}

// Limit returns the first of the named limits the model publishes, so a
// caller can state a preference order between equivalent keys. Zero means
// none of them is published.
func (e Entry) Limit(keys ...string) int64 {
	for _, k := range keys {
		if v := e.Limits[k]; v > 0 {
			return v
		}
	}
	return 0
}

// Currency is the code the model's rates are quoted in. A provider bills in
// one currency, so the first rate settles it; a model with no published rate
// is assumed to be priced in USD.
func (e Entry) Currency() string {
	for _, p := range e.Prices {
		if p.Currency != "" {
			return p.Currency
		}
	}
	return "USD"
}

// ErrNoProviders is returned for a document that has no "providers" key: it
// is not an api.json, and reading it as one would report an empty catalog.
var ErrNoProviders = errors.New("document has no providers key")

// EntryError reports one model the parser could not read, such as an entry
// whose limit is not a number. The rest of the document is still read: one
// badly typed entry does not stop its siblings from loading. A caller decides
// what a bad entry costs it; a document that is syntactically broken or cut
// short fails as a whole instead.
type EntryError struct {
	// Provider is the id of the provider object the entry was filed under.
	Provider string
	// Index is the entry's position in that provider's model list.
	Index int
	// APIID is the entry's API ID when it could still be read, else empty.
	APIID string
	// Err is why the entry could not be read.
	Err error
}

func (e EntryError) Error() string {
	id := e.APIID
	if id == "" {
		id = "unreadable id"
	}
	return fmt.Sprintf(
		"provider %q model %d (%s): %v", e.Provider, e.Index, id, e.Err,
	)
}

func (e EntryError) Unwrap() error { return e.Err }

// Result is what [Parse] read from a document.
type Result struct {
	// Entries are the wanted models, in document order.
	Entries []Entry
	// EntryErrors are the wanted models that could not be read.
	EntryErrors []EntryError
	// ProvidersSeen lists the id of every provider object in the document,
	// wanted or not, in document order, once each. It tells a feed that
	// lacks a provider from one that lists the provider with no models.
	ProvidersSeen []string
}

// Visitor receives a document as [Walk] reads it. Any field may be nil.
type Visitor struct {
	// Provider is called once for each provider object whose id is read,
	// wanted or not, before that provider's models are visited.
	Provider func(id string)
	// Entry is called for each wanted model, in document order.
	Entry func(Entry) error
	// EntryError is called for each wanted model that could not be read.
	EntryError func(EntryError) error
}

// Parse reads an api.json document and returns its models in document order,
// restricted to the providers named. With no providers named it returns every
// model. A provider is matched by the id of the object the model is filed
// under, not by the entry's own provider field.
//
// A document that ends early, is not valid JSON, has no "providers" key or
// carries data after its end returns an error and an empty Result: a partial
// catalog is never handed back. A model that cannot be read on its own is
// reported in [Result.EntryErrors] and its siblings are still returned.
func Parse(r io.Reader, providers ...string) (Result, error) {
	var res Result
	err := Walk(r, providers, Visitor{
		Provider: func(id string) {
			if !slices.Contains(res.ProvidersSeen, id) {
				res.ProvidersSeen = append(res.ProvidersSeen, id)
			}
		},
		Entry: func(e Entry) error {
			res.Entries = append(res.Entries, e)
			return nil
		},
		EntryError: func(e EntryError) error {
			res.EntryErrors = append(res.EntryErrors, e)
			return nil
		},
	})
	if err != nil {
		return Result{}, err
	}
	return res, nil
}

// Walk streams an api.json document into v, restricted to the providers named
// (all of them when none is). Models are visited in document order.
//
// The document is read as it arrives. A provider that is not wanted is
// skipped without being decoded into anything: its bytes are still read and
// syntax-checked, which is most of the cost of a large document, but nothing
// is allocated for them per token or per value. A wanted model is decoded on
// its own, which is what lets a badly typed one be reported without losing
// the rest. Providers are matched on their id alone, so a wanted provider's
// models are visited as soon as they are read. Only a provider object that
// lists its models before its id has them held as raw bytes until the id is
// known. A provider's name is available to its entries only when the object
// lists it before its models, as api.json does.
//
// v may already have been called for earlier models when Walk returns an
// error, because the error is found later in the document. A caller that must
// not act on a partial read collects what v hands it and discards it on
// error, as [Parse] does. An error from v stops the walk and is returned.
func Walk(r io.Reader, providers []string, v Visitor) error {
	w := walker{dec: json.NewDecoder(r), want: providers, v: v}
	if err := w.document(); err != nil {
		return fmt.Errorf("modelsync: %w", err)
	}
	return nil
}

type walker struct {
	dec  *json.Decoder
	want []string
	v    Visitor
}

func (w *walker) wanted(provider string) bool {
	return len(w.want) == 0 || slices.Contains(w.want, provider)
}

func (w *walker) document() error {
	null, err := w.begin('{')
	if err != nil {
		return err
	}
	if null {
		return ErrNoProviders
	}

	sawProviders := false
	for w.dec.More() {
		key, err := w.key()
		if err != nil {
			return err
		}
		if !strings.EqualFold(key, "providers") {
			if err := w.skip(); err != nil {
				return err
			}
			continue
		}
		sawProviders = true
		if err := w.providers(); err != nil {
			return err
		}
	}
	if err := w.end('}'); err != nil {
		return err
	}
	if !sawProviders {
		return ErrNoProviders
	}

	// Nothing but whitespace may follow the document.
	switch tok, err := w.dec.Token(); {
	case errors.Is(err, io.EOF):
		return nil
	case err != nil:
		return err
	default:
		return fmt.Errorf("unexpected data after the document: %v", tok)
	}
}

func (w *walker) providers() error {
	null, err := w.begin('[')
	if err != nil || null {
		return err
	}
	for w.dec.More() {
		if err := w.provider(); err != nil {
			return err
		}
	}
	return w.end(']')
}

// provider reads one provider object. A null provider reads as an empty one.
func (w *walker) provider() error {
	null, err := w.begin('{')
	if err != nil || null {
		return err
	}

	// id and name are read raw and converted afterwards, so a provider whose
	// id or name is not a string is skipped as unusable instead of failing
	// the whole document -- one malformed provider the caller never asked for
	// must not block the ones it did.
	var (
		id, name string
		haveID   bool
		badID    bool
		held     json.RawMessage
	)
	for w.dec.More() {
		key, err := w.key()
		if err != nil {
			return err
		}
		switch {
		case strings.EqualFold(key, "id"):
			var raw json.RawMessage
			if err = w.dec.Decode(&raw); err == nil {
				haveID = true
				if json.Unmarshal(raw, &id) != nil {
					id, badID = "", true
				} else if w.v.Provider != nil {
					w.v.Provider(id)
				}
			}
		case strings.EqualFold(key, "name"):
			var raw json.RawMessage
			if err = w.dec.Decode(&raw); err == nil && json.Unmarshal(raw, &name) != nil {
				name = ""
			}
		case strings.EqualFold(key, "models"):
			switch {
			case haveID && (badID || !w.wanted(id)):
				err = w.skip()
			case haveID:
				err = w.models(id, name)
			default:
				err = w.dec.Decode(&held)
			}
		default:
			err = w.skip()
		}
		if err != nil {
			return unexpected(err)
		}
	}
	if err := w.end('}'); err != nil {
		return err
	}

	if held == nil || badID || !w.wanted(id) {
		return nil
	}
	inner := walker{
		dec:  json.NewDecoder(bytes.NewReader(held)),
		want: w.want,
		v:    w.v,
	}
	return inner.models(id, name)
}

func (w *walker) models(provider, providerName string) error {
	null, err := w.begin('[')
	if err != nil || null {
		return err
	}
	for index := 0; w.dec.More(); index++ {
		var raw json.RawMessage
		if err := w.dec.Decode(&raw); err != nil {
			return unexpected(err)
		}

		// A null entry is no model, as encoding/json has always read it: the
		// generator dropped the zero value it decoded to, so it is skipped
		// here rather than reported.
		if string(raw) == "null" {
			continue
		}
		var e Entry
		if err := json.Unmarshal(raw, &e); err != nil {
			if w.v.EntryError == nil {
				continue
			}
			bad := EntryError{
				Provider: provider, Index: index,
				APIID: readableAPIID(raw), Err: err,
			}
			if err := w.v.EntryError(bad); err != nil {
				return err
			}
			continue
		}

		e.EnclosingProvider = provider
		e.EnclosingProviderName = providerName
		if w.v.Entry != nil {
			if err := w.v.Entry(e); err != nil {
				return err
			}
		}
	}
	return w.end(']')
}

// readableAPIID recovers an entry's API ID, as far as the entry allows, for
// the report on an entry that could not be read as a whole.
func readableAPIID(raw []byte) string {
	var probe struct {
		ID    any            `json:"id"`
		Attrs map[string]any `json:"attrs"`
	}
	// The error is the one already reported; the fields decoded before it
	// are exactly what is wanted here.
	_ = json.Unmarshal(raw, &probe)
	if id, ok := probe.Attrs["api_id"].(string); ok && id != "" {
		return id
	}
	id, _ := probe.ID.(string)
	return id
}

func (w *walker) key() (string, error) {
	tok, err := w.dec.Token()
	if err != nil {
		return "", unexpected(err)
	}
	s, ok := tok.(string)
	if !ok {
		return "", fmt.Errorf("expected an object key, found %v", tok)
	}
	return s, nil
}

// begin consumes the opening delimiter of an object or array. A JSON null
// reads as an empty one, which is how encoding/json treats it, and is
// reported as null.
func (w *walker) begin(want json.Delim) (null bool, err error) {
	tok, err := w.dec.Token()
	if err != nil {
		return false, unexpected(err)
	}
	if tok == nil {
		return true, nil
	}
	if d, ok := tok.(json.Delim); !ok || d != want {
		return false, fmt.Errorf("expected %q, found %v", want, tok)
	}
	return false, nil
}

func (w *walker) end(want json.Delim) error {
	tok, err := w.dec.Token()
	if err != nil {
		return unexpected(err)
	}
	if d, ok := tok.(json.Delim); !ok || d != want {
		return fmt.Errorf("expected %q, found %v", want, tok)
	}
	return nil
}

// discard is a decode target that keeps nothing: the decoder reads and
// syntax-checks the value and hands its bytes here without copying them.
type discard struct{}

func (*discard) UnmarshalJSON([]byte) error { return nil }

// skip consumes one JSON value of any shape without keeping any of it.
func (w *walker) skip() error {
	var d discard
	return unexpected(w.dec.Decode(&d))
}

// unexpected turns a clean end of input in the middle of a value into the
// error it is: the document was cut short.
func unexpected(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}
