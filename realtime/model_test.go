package realtime

import "testing"

// A pair of classes swapped in Rate's mapping would price a session wrong
// while every number stayed plausible; each field is given a distinct value
// so a swap cannot pass unnoticed.
func TestModelRate_MapsEveryClassToItsOwnField(t *testing.T) {
	m := Model{
		CostPer1MTextIn:        1,
		CostPer1MTextInCached:  2,
		CostPer1MTextOut:       3,
		CostPer1MAudioIn:       4,
		CostPer1MAudioInCached: 5,
		CostPer1MAudioOut:      6,
		CostPer1MImageIn:       7,
		CostPer1MImageInCached: 8,
	}

	for _, tc := range []struct {
		class TokenClass
		want  float64
	}{
		{ClassTextInput, 1},
		{ClassTextInputCached, 2},
		{ClassTextOutput, 3},
		{ClassAudioInput, 4},
		{ClassAudioInputCached, 5},
		{ClassAudioOutput, 6},
		{ClassImageInput, 7},
		{ClassImageInputCached, 8},
	} {
		got, ok := m.Rate(tc.class)
		if !ok {
			t.Errorf("Rate(%q) reported no rate, though one is set", tc.class)
			continue
		}
		if got != tc.want {
			t.Errorf(
				"Rate(%q) = %v, want %v -- two fields are crossed",
				tc.class,
				got,
				tc.want,
			)
		}
	}
}

// Guards against a class added to TokenClasses without a matching field:
// that should fail loudly rather than price at zero. Every field is set
// non-zero here, so the only way Rate can report "no rate" is a class it
// does not know about.
func TestModelRate_CoversEveryDeclaredClass(t *testing.T) {
	m := Model{
		CostPer1MTextIn: 1, CostPer1MTextInCached: 1, CostPer1MTextOut: 1,
		CostPer1MAudioIn: 1, CostPer1MAudioInCached: 1, CostPer1MAudioOut: 1,
		CostPer1MImageIn: 1, CostPer1MImageInCached: 1,
	}
	for _, class := range TokenClasses {
		if _, ok := m.Rate(class); !ok {
			t.Errorf(
				"Rate(%q) reports no rate with every field set; the class has no field",
				class,
			)
		}
	}
}

// A source that stops listing a rate leaves the field zero; calling that
// free would put real spend into a ledger as nothing.
func TestModelRate_AZeroIsUnpublishedNotFree(t *testing.T) {
	var m Model
	for _, class := range TokenClasses {
		if rate, ok := m.Rate(class); ok {
			t.Errorf(
				"Rate(%q) = (%v, true) on an unpriced model, want no rate",
				class,
				rate,
			)
		}
	}
}
