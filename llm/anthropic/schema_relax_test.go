package anthropic

import (
	"reflect"
	"testing"
)

// A repeated object type (the same shape used at several sites, differing only in per-site description)
// must be hoisted into a single shared $def and referenced by $ref everywhere — the descriptions kept at
// the ref sites. This is what collapses the optional-parameter count under Anthropic's 24 cap.
func TestFactorSharedObjectDefs_HoistsRepeatedTypes(t *testing.T) {
	duration := func(desc string) map[string]any {
		return map[string]any{
			"type":        "object",
			"description": desc,
			"properties": map[string]any{
				"canonical": map[string]any{"type": "number"},
				"display":   map[string]any{"type": "string"},
			},
			"required":             []string{"display"},
			"additionalProperties": false,
		}
	}
	props := map[string]any{
		"title":      map[string]any{"type": "string"},
		"prep_time":  duration("prep"),
		"cook_time":  duration("cook"),
		"total_time": duration("total"),
		"steps": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"timer": duration("step timer")},
				"required":             []string{},
				"additionalProperties": false,
			},
		},
	}

	defs, out := factorSharedObjectDefs(props)

	if len(defs) != 1 {
		t.Fatalf("defs = %d, want exactly 1 shared duration def", len(defs))
	}
	for field, wantDesc := range map[string]string{"prep_time": "prep", "cook_time": "cook", "total_time": "total"} {
		ref, ok := out[field].(map[string]any)
		if !ok || ref["$ref"] == nil {
			t.Errorf("%s = %v, want a $ref", field, out[field])
			continue
		}
		if ref["description"] != wantDesc {
			t.Errorf("%s description = %v, want %q preserved at the ref site", field, ref["description"], wantDesc)
		}
	}
	var def map[string]any
	for _, d := range defs {
		def = d.(map[string]any)
	}
	if _, has := def["description"]; has {
		t.Error("shared def body must not carry a per-site description")
	}
	if _, ok := def["properties"].(map[string]any)["canonical"]; !ok {
		t.Error("shared def lost the duration properties")
	}
	stepItems := out["steps"].(map[string]any)["items"].(map[string]any)
	timer, ok := stepItems["properties"].(map[string]any)["timer"].(map[string]any)
	if !ok || timer["$ref"] == nil {
		t.Errorf("step timer = %v, want a $ref to the shared duration", stepItems["properties"])
	}
	if out["title"].(map[string]any)["type"] != "string" {
		t.Error("unshared scalar field should stay inline")
	}
}

// strictSchema mimics what the shared schema generator emits: every field in `required`, optionals as
// nullable unions ["T","null"], including a nullable nested object and array-of-object items that carry
// their own nullable fields. relaxNullableUnions must turn this into standard optionality with no "null"
// anywhere and the nullable fields dropped from every `required` list — and must NOT mutate the input.
func strictSchema() (map[string]any, []string) {
	props := map[string]any{
		"title":    map[string]any{"type": "string"},
		"servings": map[string]any{"type": []string{"integer", "null"}},
		"prep_time": map[string]any{
			"type": []string{"object", "null"},
			"properties": map[string]any{
				"minutes": map[string]any{"type": []string{"number", "null"}},
				"display": map[string]any{"type": "string"},
			},
			"required":             []string{"minutes", "display"},
			"additionalProperties": false,
		},
		"components": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"qty":  map[string]any{"type": []string{"number", "null"}},
				},
				"required":             []string{"name", "qty"},
				"additionalProperties": false,
			},
		},
	}
	return props, []string{"title", "servings", "prep_time", "components"}
}

// hasNullType reports whether any "type" anywhere in the schema tree still lists "null".
func hasNullType(v any) bool {
	switch node := v.(type) {
	case map[string]any:
		if t, ok := node["type"]; ok {
			for _, e := range asAnySlice(t) {
				if s, ok := e.(string); ok && s == "null" {
					return true
				}
			}
		}
		for _, child := range node {
			if hasNullType(child) {
				return true
			}
		}
	case []any:
		for _, e := range node {
			if hasNullType(e) {
				return true
			}
		}
	}
	return false
}

func asAnySlice(t any) []any {
	switch v := t.(type) {
	case []string:
		out := make([]any, len(v))
		for i, s := range v {
			out[i] = s
		}
		return out
	case []any:
		return v
	default:
		return nil
	}
}

func TestRelaxNullableUnions_StripsNullAndOptionalizesEverywhere(t *testing.T) {
	props, required := strictSchema()

	gotProps, gotReq := relaxNullableUnions(props, required)

	// 1. No "null" survives anywhere in the relaxed tree.
	if hasNullType(map[string]any{"properties": gotProps}) {
		t.Errorf("relaxed schema still contains a nullable-union type:\n%#v", gotProps)
	}

	// 2. Nullable top-level fields dropped from required; plain fields kept, order preserved.
	if want := []string{"title", "components"}; !reflect.DeepEqual(gotReq, want) {
		t.Errorf("top-level required = %v, want %v (servings + prep_time are optional now)", gotReq, want)
	}

	// 3. Plain field keeps its scalar type.
	if got := gotProps["title"].(map[string]any)["type"]; got != "string" {
		t.Errorf("title type = %v, want plain \"string\"", got)
	}
	// 4. Nullable scalar de-nulled to the single base type.
	if got := gotProps["servings"].(map[string]any)["type"]; got != "integer" {
		t.Errorf("servings type = %v, want plain \"integer\"", got)
	}
	// 5. Nullable nested object de-nulled AND its own nullable child relaxed + dropped from nested required.
	prep := gotProps["prep_time"].(map[string]any)
	if prep["type"] != "object" {
		t.Errorf("prep_time type = %v, want plain \"object\"", prep["type"])
	}
	if got := prep["required"]; !reflect.DeepEqual(got, []string{"display"}) {
		t.Errorf("prep_time.required = %v, want [display] (minutes is optional now)", got)
	}
	// 6. Array-of-object items relaxed: qty de-nulled and dropped from items.required.
	items := gotProps["components"].(map[string]any)["items"].(map[string]any)
	if got := items["required"]; !reflect.DeepEqual(got, []string{"name"}) {
		t.Errorf("components.items.required = %v, want [name] (qty is optional now)", got)
	}

	// 7. The INPUT must be untouched (deep copy) — the OpenAI/Gemini paths still see the strict form.
	if !hasNullType(map[string]any{"properties": props}) {
		t.Error("input schema was mutated: its nullable unions were stripped in place")
	}
	if want := []string{"title", "servings", "prep_time", "components"}; !reflect.DeepEqual(required, want) {
		t.Errorf("input required slice was mutated: %v", required)
	}
}
