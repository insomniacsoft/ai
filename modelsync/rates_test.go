package modelsync

import "testing"

func TestRealtimeModelYieldsOneRatePerMetricAndModality(t *testing.T) {
	e := entriesByID(t, fixture, "openai")["gpt-realtime-2.1-mini"]
	r := e.RealtimeRates()

	for name, c := range map[string]struct {
		got  Rate
		want float64
	}{
		"text in":   {r.TextIn, 0.6},
		"text out":  {r.TextOut, 2.4},
		"audio in":  {r.AudioIn, 10},
		"audio out": {r.AudioOut, 20},
	} {
		if !c.got.Published || c.got.Per != c.want {
			t.Errorf("%s = %+v, want %v", name, c.got, c.want)
		}
	}
	if r.ImageIn.Published || r.TextInCached.Published {
		t.Errorf("rates the source does not state must not be published: %+v", r)
	}
}

func TestRealClaudeEntryYieldsStandardTierRatesAndNoCacheRead(t *testing.T) {
	e := entriesByID(t, fixture, "anthropic")["claude-sonnet-5"]
	r := e.ChatRates()

	for name, c := range map[string]struct {
		got  Rate
		want float64
	}{
		"input":       {r.Input, 2},
		"output":      {r.Output, 10},
		"cache write": {r.CacheWrite, 2.5},
	} {
		if !c.got.Published || c.got.Per != c.want {
			t.Errorf("%s = %+v, want %v", name, c.got, c.want)
		}
	}
	if r.CacheRead.Published {
		t.Errorf("the feed states no cache-read rate for Claude: %+v", r.CacheRead)
	}
}

func TestSyntheticEntryWithACacheReadRateReportsIt(t *testing.T) {
	e := entriesByID(t, fixture, "synthetic")["synth-sonnet"]
	if r := e.ChatRates().CacheRead; !r.Published || r.Per != 0.3 {
		t.Errorf("cache read = %+v, want 0.3", r)
	}
}

func cacheWrite(amount float64, dims map[string]string) Price {
	return Price{
		Metric: "cache_write_tokens", Unit: "per_1m_tokens",
		Amount: amount, Currency: "USD", Dims: dims,
	}
}

func TestCacheWritePrefersTheFiveMinuteRateEvenWhenAnotherIsCheaperOrSimpler(
	t *testing.T,
) {
	// Synthetic. The one-hour rate names no cache_ttl, so it is less
	// qualified, and it is cheaper: the ordinary tie-break would pick it.
	e := Entry{Prices: []Price{
		cacheWrite(1, map[string]string{"tier": "standard"}),
		cacheWrite(3.75, map[string]string{
			"cache_ttl": "5m", "tier": "standard",
		}),
		cacheWrite(6, map[string]string{
			"cache_ttl": "1h", "tier": "standard",
		}),
	}}

	if r := e.ChatRates().CacheWrite; !r.Published || r.Per != 3.75 {
		t.Errorf("cache write = %+v, want the 5m rate 3.75", r)
	}
}

func TestCacheWriteFallsBackToRatesThatNameNoLifetime(t *testing.T) {
	e := Entry{Prices: []Price{
		cacheWrite(6, map[string]string{}),
		cacheWrite(4, map[string]string{"cache_ttl": "1h"}),
	}}

	if r := e.ChatRates().CacheWrite; !r.Published || r.Per != 6 {
		t.Errorf("cache write = %+v, want the rate that names no lifetime, 6", r)
	}
}

// A provider that publishes only a longer lifetime has no 5-minute rate, and
// none of its lifetimes may stand in for one.
func TestCacheWriteIsUnpublishedWhenOnlyLongerLifetimesArePublished(t *testing.T) {
	e := Entry{Prices: []Price{
		cacheWrite(6, map[string]string{"cache_ttl": "1h"}),
		cacheWrite(4, map[string]string{"cache_ttl": "30m", "region": "eu"}),
	}}

	if r := e.ChatRates().CacheWrite; r.Published {
		t.Errorf("cache write = %+v, want unpublished", r)
	}
}

func TestCapabilitiesDistinguishUnpublishedFromAbsent(t *testing.T) {
	claude := entriesByID(t, fixture, "anthropic")["claude-sonnet-5"]
	if got, published := claude.SupportsAttachments(); got || published {
		t.Errorf("Claude lists no input modalities: got %v, %v", got, published)
	}
	if reasons, published := claude.CanReason(); !reasons || !published {
		t.Errorf("Claude lists the reasoning feature: got %v, %v",
			reasons, published)
	}
	if claude.HasList("endpoints") || !claude.HasList("features") {
		t.Errorf("HasList disagrees with the entry: %v", claude.Lists)
	}

	chat := entriesByID(t, fixture, "openai")["demo-chat"]
	if got, published := chat.SupportsAttachments(); !got || !published {
		t.Errorf("demo-chat takes images: got %v, %v", got, published)
	}

	textOnly := Entry{Lists: map[string][]string{
		"input_modalities": {"text"}, "features": {"streaming"},
	}}
	if got, published := textOnly.SupportsAttachments(); got || !published {
		t.Errorf("text only: got %v, %v, want false, true", got, published)
	}
	if reasons, published := textOnly.CanReason(); reasons || !published {
		t.Errorf("features without reasoning: got %v, %v", reasons, published)
	}

	if reasons, published := (Entry{}).CanReason(); reasons || published {
		t.Errorf("an empty entry says nothing: got %v, %v", reasons, published)
	}
}

func TestSelectRatePrefersTheLeastQualifiedStandardRate(t *testing.T) {
	prices := []Price{
		{Metric: "input_tokens", Unit: "per_1m_tokens", Amount: 9, Currency: "USD",
			Dims: map[string]string{"tier": "batch"}},
		{Metric: "input_tokens", Unit: "per_1m_tokens", Amount: 5, Currency: "USD",
			Dims: map[string]string{"region": "eu"}},
		{Metric: "input_tokens", Unit: "per_1m_tokens", Amount: 4, Currency: "USD",
			Dims: map[string]string{"tier": "standard"}},
		{Metric: "input_tokens", Unit: "per_1k_tokens", Amount: 0.002, Currency: "EUR"},
	}
	if got, ok := SelectRate(prices, "USD", TokenUnits, "input_tokens"); !ok || got != 4 {
		t.Errorf("got %v, %v, want 4", got, ok)
	}
	if got, ok := SelectRate(prices, "EUR", TokenUnits, "input_tokens"); !ok || got != 2 {
		t.Errorf("got %v, %v, want 2 per million", got, ok)
	}
	if _, ok := SelectRate(prices, "USD", TokenUnits, "output_tokens"); ok {
		t.Error("an unstated metric must not be found")
	}
}
