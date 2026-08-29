package realtime

import "testing"

// TestModelRate_MapsEveryClassToItsOwnField is the guard against the one
// mistake this type can make invisibly.
//
// Rate transcribes eight classes onto eight fields. A pair swapped -- audio
// output onto text output, say -- would price a voice conversation at a
// fortieth of what it cost, and nothing would report anything wrong: the
// numbers are plausible, the total is produced, and only somebody adding up an
// invoice by hand would ever find it.
//
// Each field is given a distinct value so a swap cannot pass.
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
			t.Errorf("Rate(%q) = %v, want %v -- two fields are crossed", tc.class, got, tc.want)
		}
	}
}

// TestModelRate_CoversEveryDeclaredClass keeps the test above honest. It checks
// the pairs it lists; this checks that the list is all of them, so a ninth
// class added to TokenClasses without a field here fails loudly instead of
// pricing at zero.
func TestModelRate_CoversEveryDeclaredClass(t *testing.T) {
	// Every field non-zero, so the only way Rate can report "no rate" is a
	// class it does not know about.
	m := Model{
		CostPer1MTextIn: 1, CostPer1MTextInCached: 1, CostPer1MTextOut: 1,
		CostPer1MAudioIn: 1, CostPer1MAudioInCached: 1, CostPer1MAudioOut: 1,
		CostPer1MImageIn: 1, CostPer1MImageInCached: 1,
	}
	for _, class := range TokenClasses {
		if _, ok := m.Rate(class); !ok {
			t.Errorf("Rate(%q) reports no rate with every field set; the class has no field", class)
		}
	}
}

// TestModelRate_AZeroIsUnpublishedNotFree. A source that stops listing a rate
// leaves the field zero, which is indistinguishable from a rate of zero.
// Calling it free would put real spend into a ledger as nothing.
func TestModelRate_AZeroIsUnpublishedNotFree(t *testing.T) {
	var m Model
	for _, class := range TokenClasses {
		if rate, ok := m.Rate(class); ok {
			t.Errorf("Rate(%q) = (%v, true) on an unpriced model, want no rate", class, rate)
		}
	}
}
