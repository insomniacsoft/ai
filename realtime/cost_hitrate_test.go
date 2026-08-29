package realtime

import "testing"

func TestTextCacheHitRateSeparatesNothingFromNothingCached(t *testing.T) {
	// The distinction this function exists for: a session that has sent no
	// text at all has no hit rate, and reporting 0% for it would read in a log
	// as a total cache miss.
	if _, ok := TextCacheHitRate(map[TokenClass]int64{}); ok {
		t.Fatal("a usage record with no text input reported a hit rate")
	}
	rate, ok := TextCacheHitRate(map[TokenClass]int64{ClassTextInput: 100})
	if !ok {
		t.Fatal("100 uncached text tokens reported no hit rate")
	}
	if rate != 0 {
		t.Fatalf("nothing cached should be 0%%, got %v", rate)
	}
}

func TestTextCacheHitRateIgnoresAudio(t *testing.T) {
	// Audio input is what the room just said — never a cached prefix. Folding
	// it in would make the rate fall whenever somebody speaks for longer,
	// which is the opposite of what the number is for.
	quiet := map[TokenClass]int64{ClassTextInput: 50, ClassTextInputCached: 50}
	talkative := map[TokenClass]int64{
		ClassTextInput: 50, ClassTextInputCached: 50,
		ClassAudioInput: 5000, ClassAudioInputCached: 10,
	}
	a, _ := TextCacheHitRate(quiet)
	b, _ := TextCacheHitRate(talkative)
	if a != b {
		t.Fatalf("audio changed the text cache hit rate: %v vs %v", a, b)
	}
	if a != 0.5 {
		t.Fatalf("half cached should be 0.5, got %v", a)
	}
}

func TestTextCacheHitRateIsNotAlwaysTheSame(t *testing.T) {
	// Negative control. A function returning a constant would satisfy every
	// assertion above that checks a single value.
	low, _ := TextCacheHitRate(map[TokenClass]int64{ClassTextInput: 900, ClassTextInputCached: 100})
	high, _ := TextCacheHitRate(map[TokenClass]int64{ClassTextInput: 100, ClassTextInputCached: 900})
	if !(low < high) {
		t.Fatalf("a mostly-uncached turn (%v) did not rate below a mostly-cached one (%v)", low, high)
	}
}
