package llm

import "testing"

// TestDeprecatedReadsTheProvidersOwnWord.
//
// A method rather than a string comparison at each call site. "deprecated" is
// one spelling out of the provider's vocabulary, and a caller that writes the
// literal is one that keeps compiling, keeps passing, and quietly starts
// offering retired models again the day that vocabulary grows a second word
// for the same thing.
func TestDeprecatedReadsTheProvidersOwnWord(t *testing.T) {
	for _, tc := range []struct {
		state string
		want  bool
	}{
		{"deprecated", true},
		{"active", false},
		// The provider publishing nothing is NOT a claim that the model is
		// fine. It is the absence of a claim, and a caller hiding models on it
		// would be acting on no evidence.
		{"", false},
	} {
		if got := (Model{State: tc.state}).Deprecated(); got != tc.want {
			t.Errorf("State %q: Deprecated() = %v, want %v", tc.state, got, tc.want)
		}
	}
}
