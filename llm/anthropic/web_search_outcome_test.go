package anthropic

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/message"
)

const (
	usageOneSearch = `{"input_tokens":10,"output_tokens":20,` +
		`"server_tool_use":{"web_search_requests":1}}`
	usageNoCount = `{"input_tokens":10,"output_tokens":20}`
	usageNull    = `{"input_tokens":10,"output_tokens":20,` +
		`"server_tool_use":null}`
	usageZero = `{"input_tokens":10,"output_tokens":20,` +
		`"server_tool_use":{"web_search_requests":0}}`

	resultsFour = `[
  {"type":"web_search_result","url":"https://example.com/a",
   "title":"Example A","encrypted_content":"enc-a","page_age":"1 day ago"},
  {"type":"web_search_result","url":"https://example.com/b",
   "title":"Example B","encrypted_content":"enc-b","page_age":null},
  {"type":"web_search_result","url":"https://example.org/c",
   "title":"Example C","encrypted_content":"enc-c","page_age":"3 days ago"},
  {"type":"web_search_result","url":"https://example.org/d",
   "title":"Example D","encrypted_content":"enc-d","page_age":null}
]`
	resultsOne = `[
  {"type":"web_search_result","url":"https://example.com/shop",
   "title":"Example Shop","encrypted_content":"enc-s","page_age":null}
]`
	resultsNone = `[]`
	resultsErr  = `{"type":"web_search_tool_result_error",` +
		`"error_code":"max_uses_exceeded"}`
)

// wsText returns a text content block with no citations.
func wsText(text string) string {
	return fmt.Sprintf(`{"type":"text","text":%q}`, text)
}

// wsCited returns a text content block carrying one
// web_search_result_location citation.
func wsCited(text, url, title string) string {
	return fmt.Sprintf(`{"type":"text","text":%q,"citations":[`+
		`{"type":"web_search_result_location","url":%q,"title":%q,`+
		`"cited_text":"an excerpt","encrypted_index":"idx-1"}]}`,
		text, url, title)
}

// wsSearch returns a server_tool_use content block for a web_search call.
func wsSearch(id, query string) string {
	return fmt.Sprintf(`{"type":"server_tool_use","id":%q,`+
		`"name":"web_search","input":{"query":%q}}`, id, query)
}

// wsResult returns a web_search_tool_result content block for the search
// with the given id; content is raw JSON, either a result list or an error.
func wsResult(id, content string) string {
	return fmt.Sprintf(`{"type":"web_search_tool_result",`+
		`"tool_use_id":%q,"content":%s}`, id, content)
}

// wsBody returns a Messages API response body holding blocks, with the given
// stop reason and raw JSON usage.
func wsBody(stop, usage string, blocks ...string) string {
	return fmt.Sprintf(`{"id":"msg_01","type":"message","role":"assistant",`+
		`"model":"claude","content":[%s],"stop_reason":%q,`+
		`"stop_sequence":null,"usage":%s}`,
		strings.Join(blocks, ","), stop, usage)
}

// wsSend serves body from a test server, calls SendMessages through it and
// returns the response. With search set the client is built WithWebSearch.
func wsSend(t *testing.T, body string, search bool) *llm.Response {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
				t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, body)
		}))
	defer srv.Close()

	var n int
	opts := []Option{
		WithAPIKey("test-key"),
		WithModel(llm.Model{APIModel: "claude"}),
		WithHTTPClient(&http.Client{
			Transport: redirectRT{
				base: http.DefaultTransport,
				host: srv.Listener.Addr().String(),
				n:    &n,
			},
		}),
	}
	if search {
		opts = append(opts, WithWebSearch(WebSearchConfig{}))
	}
	resp, err := NewLLM(opts...).SendMessages(
		context.Background(),
		[]message.Message{message.NewUserMessage("when does it open?")},
		nil,
	)
	if err != nil {
		t.Fatalf("SendMessages: %v", err)
	}
	return resp
}

// wsKeys lists the metadata keys that only a client built WithWebSearch sets.
var wsKeys = []string{
	"anthropic.web_search_requests",
	"anthropic.web_search_citations",
	"anthropic.web_search_errors",
	"anthropic.answer_text",
}

// wsWant fails the test unless resp carries key in its provider metadata with
// a value equal to want.
func wsWant(t *testing.T, resp *llm.Response, key string, want any) {
	t.Helper()
	got, ok := resp.ProviderMetadata[key]
	if !ok {
		t.Fatalf("%s missing; metadata %v", key, resp.ProviderMetadata)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%s = %#v, want %#v", key, got, want)
	}
}

// wsWantNo fails the test if resp carries key in its provider metadata.
func wsWantNo(t *testing.T, resp *llm.Response, key string) {
	t.Helper()
	if got, ok := resp.ProviderMetadata[key]; ok {
		t.Errorf("%s = %#v, want it absent", key, got)
	}
}

// TestSearchOutcomeCountsRequestsNotResults confirms the billed search count
// comes from usage, not from the number of results a search returned.
func TestSearchOutcomeCountsRequestsNotResults(t *testing.T) {
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsSearch("srvtoolu_1", "shop opening hours"),
		wsResult("srvtoolu_1", resultsFour),
		wsText("Four pages came back."),
	), true)

	wsWant(t, resp, "anthropic.web_search_requests", int64(1))
	raw := resp.ProviderMetadata["anthropic.web_search_results"]
	results, ok := raw.([]map[string]any)
	if !ok || len(results) != 4 {
		t.Fatalf("web_search_results = %#v, want 4 entries", raw)
	}
	if results[2]["url"] != "https://example.org/c" {
		t.Errorf("third result url = %v", results[2]["url"])
	}
	wsWantNo(t, resp, "anthropic.web_search_errors")
}

// TestSearchOutcomeEmptyResultsHaveNoCitations confirms a search that
// returns no results is still counted and yields no citations or errors.
func TestSearchOutcomeEmptyResultsHaveNoCitations(t *testing.T) {
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsSearch("srvtoolu_1", "an obscure query"),
		wsResult("srvtoolu_1", resultsNone),
		wsText("Nothing relevant turned up."),
	), true)

	wsWant(t, resp, "anthropic.web_search_requests", int64(1))
	wsWantNo(t, resp, "anthropic.web_search_citations")
	wsWantNo(t, resp, "anthropic.web_search_errors")
	wsWant(t, resp, "anthropic.answer_text", "Nothing relevant turned up.")
}

// TestSearchOutcomeReportsErrorCodeFromRawContent confirms the error_code of
// a failed search is reported, and that a successful search beside it is not.
func TestSearchOutcomeReportsErrorCodeFromRawContent(t *testing.T) {
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsSearch("srvtoolu_1", "first query"),
		wsResult("srvtoolu_1", resultsOne),
		wsSearch("srvtoolu_2", "second query"),
		wsResult("srvtoolu_2", resultsErr),
		wsText("Only one search was allowed."),
	), true)

	wsWant(t, resp, "anthropic.web_search_requests", int64(1))
	wsWant(t, resp, "anthropic.web_search_errors",
		[]string{"max_uses_exceeded"})
}

// TestSearchOutcomeAnswerDropsNarrationBeforeSearch confirms answer_text
// omits the uncited text before a search while Content keeps it.
func TestSearchOutcomeAnswerDropsNarrationBeforeSearch(t *testing.T) {
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsText("I'll search for that."),
		wsSearch("srvtoolu_1", "shop opening hours"),
		wsResult("srvtoolu_1", resultsOne),
		wsText("Based on the results, "),
		wsCited("the store opens at 9.",
			"https://example.com/shop", "Example Shop"),
	), true)

	wsWant(t, resp, "anthropic.answer_text",
		"Based on the results, the store opens at 9.")
	wsWant(t, resp, "anthropic.web_search_citations",
		[]map[string]any{{
			"url":   "https://example.com/shop",
			"title": "Example Shop",
		}})
	wsWant(t, resp, "anthropic.web_search_requests", int64(1))
	wsWantNo(t, resp, "anthropic.web_search_errors")
	wsWant(t, resp, "anthropic.stop_reason", "end_turn")
	want := "I'll search for that.Based on the results, " +
		"the store opens at 9."
	if resp.Content != want {
		t.Errorf("Content = %q, want %q", resp.Content, want)
	}
}

// TestSearchOutcomeAnswerBreaksOnlyBetweenSearches confirms answer_text puts
// a line break between the answers to two searches and nowhere else.
func TestSearchOutcomeAnswerBreaksOnlyBetweenSearches(t *testing.T) {
	usageTwo := `{"input_tokens":10,"output_tokens":20,` +
		`"server_tool_use":{"web_search_requests":2}}`
	resp := wsSend(t, wsBody("end_turn", usageTwo,
		wsText("Let me search."),
		wsSearch("srvtoolu_1", "first query"),
		wsResult("srvtoolu_1", resultsOne),
		wsCited("Finding A.", "https://example.com/a", "Example A"),
		wsText("Let me also check."),
		wsSearch("srvtoolu_2", "second query"),
		wsResult("srvtoolu_2", resultsOne),
		wsCited("Finding B.", "https://example.org/b", "Example B"),
	), true)

	wsWant(t, resp, "anthropic.answer_text", "Finding A.\nFinding B.")
	wsWant(t, resp, "anthropic.web_search_requests", int64(2))
	wsWant(t, resp, "anthropic.web_search_citations",
		[]map[string]any{
			{"url": "https://example.com/a", "title": "Example A"},
			{"url": "https://example.org/b", "title": "Example B"},
		})
}

// TestSearchOutcomeAnswerJoinsSplitSentenceWithoutBreak confirms a sentence
// split across text blocks at a citation is joined without a separator.
func TestSearchOutcomeAnswerJoinsSplitSentenceWithoutBreak(t *testing.T) {
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsSearch("srvtoolu_1", "shop opening hours"),
		wsResult("srvtoolu_1", resultsOne),
		wsText("The store opens "),
		wsCited("at nine", "https://example.com/shop", "Example Shop"),
		wsText(" on weekdays."),
	), true)

	wsWant(t, resp, "anthropic.answer_text",
		"The store opens at nine on weekdays.")
}

// TestSearchOutcomeRequestCountPresence confirms the request count is
// written when the response reports it, zero included, and left out when the
// response does not.
func TestSearchOutcomeRequestCountPresence(t *testing.T) {
	cases := []struct {
		name  string
		usage string
		want  any
	}{
		{"field absent", usageNoCount, nil},
		{"server_tool_use null", usageNull, nil},
		{"present zero", usageZero, int64(0)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := wsSend(t, wsBody("end_turn", tc.usage,
				wsText("Answered from memory."),
			), true)

			if tc.want == nil {
				wsWantNo(t, resp, "anthropic.web_search_requests")
			} else {
				wsWant(t, resp, "anthropic.web_search_requests", tc.want)
			}
			wsWant(t, resp, "anthropic.answer_text",
				"Answered from memory.")
		})
	}
}

// TestSearchOutcomeReportsPauseTurnStopReason confirms a paused turn reports
// its raw stop reason and an empty answer_text.
func TestSearchOutcomeReportsPauseTurnStopReason(t *testing.T) {
	resp := wsSend(t, wsBody("pause_turn", usageOneSearch,
		wsSearch("srvtoolu_1", "a long running query"),
	), true)

	wsWant(t, resp, "anthropic.stop_reason", "pause_turn")
	wsWant(t, resp, "anthropic.answer_text", "")
}

// TestPlainCallReportsStopReasonAndNoSearchKeys confirms a client built
// without WithWebSearch reports the stop reason and none of the search keys.
func TestPlainCallReportsStopReasonAndNoSearchKeys(t *testing.T) {
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsCited("The store opens at 9.",
			"https://example.com/shop", "Example Shop"),
		wsSearch("srvtoolu_1", "stray"),
		wsResult("srvtoolu_1", resultsErr),
	), false)

	wsWant(t, resp, "anthropic.stop_reason", "end_turn")
	for _, key := range wsKeys {
		wsWantNo(t, resp, key)
	}
}

// TestSearchOutcomeIgnoresCitationsOfOtherTypes confirms only
// web_search_result_location citations are reported, while text carrying
// other citation types stays in answer_text.
func TestSearchOutcomeIgnoresCitationsOfOtherTypes(t *testing.T) {
	pageCited := `{"type":"text","text":"From the attached file.",` +
		`"citations":[{"type":"page_location","cited_text":"an excerpt",` +
		`"document_index":0,"document_title":"Attached Guide",` +
		`"start_page_number":2,"end_page_number":3}]}`
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsSearch("srvtoolu_1", "shop opening hours"),
		wsResult("srvtoolu_1", resultsOne),
		pageCited,
		wsCited("The store opens at 9.",
			"https://example.com/shop", "Example Shop"),
	), true)

	wsWant(t, resp, "anthropic.web_search_citations",
		[]map[string]any{{
			"url":   "https://example.com/shop",
			"title": "Example Shop",
		}})
	wsWant(t, resp, "anthropic.answer_text",
		"From the attached file.The store opens at 9.")
}

// TestSearchOutcomeAnswerBreaksAtSearchWithoutNarration confirms a search
// between two cited answers produces a line break even with no narration
// block before it.
func TestSearchOutcomeAnswerBreaksAtSearchWithoutNarration(t *testing.T) {
	resp := wsSend(t, wsBody("end_turn", usageOneSearch,
		wsCited("Finding A.", "https://example.com/a", "Example A"),
		wsSearch("srvtoolu_2", "second query"),
		wsResult("srvtoolu_2", resultsOne),
		wsCited("Finding B.", "https://example.org/b", "Example B"),
	), true)

	wsWant(t, resp, "anthropic.answer_text", "Finding A.\nFinding B.")
}
