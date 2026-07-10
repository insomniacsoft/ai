package batch

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"

	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/storage"
	"github.com/google/uuid"
	"github.com/joakimcarlsson/ai/llm"
	"github.com/joakimcarlsson/ai/message"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/genai"
)

// VertexGCS is an ASYNC batch path for the Vertex AI backend, which (unlike the
// Gemini Developer API) does NOT accept inlined requests — it requires GCS-file
// input/output. It is deliberately split into Submit + Collect so the caller
// never blocks on Vertex's multi-minute batch turnaround: Submit uploads the
// requests and creates the batch job (returning a handle); Collect polls the
// job and, once terminal, downloads + parses the GCS output. A River collect
// job snoozes between Collect calls.
//
// Correlation: Vertex batch output is NOT ordered to match the input, and the
// GenerateContent batch format has no per-request custom id. So Submit injects
// a `[[ovreq:<id>]]` marker into each request's first user part; Collect reads
// it back from the echoed request in the output. The marker is a short opaque
// prefix the model ignores.

const vertexReqMarkerPrefix = "ovreq"

var vertexReqMarkerRE = regexp.MustCompile(`\[\[ovreq:([^\]]+)\]\]`)

// VertexGCSConfig configures a VertexGCSProcessor.
type VertexGCSConfig struct {
	Project  string // GCP project id (required)
	Location string // Vertex location — "global" for Gemini 3+ models (required)
	// CredentialsFile is a service-account JSON key path. Empty → Application
	// Default Credentials (GOOGLE_APPLICATION_CREDENTIALS / metadata server).
	CredentialsFile string
	Bucket          string // GCS bucket for batch input/output (required)
	Prefix          string // GCS object prefix (e.g. "editorial")
	Model           string // Vertex model id, e.g. "gemini-3.1-flash-lite-preview"
	MaxTokens       int64  // max output tokens per request (0 → provider default)
}

// VertexGCSProcessor submits and collects Vertex GCS batch jobs.
type VertexGCSProcessor struct {
	genai *genai.Client
	store *storage.Client
	cfg   VertexGCSConfig
}

// VertexBatchHandle identifies a submitted batch for later collection. It is
// the serializable unit the caller persists (e.g. on a River collect job).
type VertexBatchHandle struct {
	BatchName string `json:"batch_name"` // genai batch job resource name
	InputURI  string `json:"input_uri"`  // gs://… input jsonl
	OutputURI string `json:"output_uri"` // gs://… output dir prefix
	Count     int    `json:"count"`      // number of requests submitted
}

// NewVertexGCS builds a processor with a Vertex genai client + a GCS storage
// client, both authenticated from the same credentials.
func NewVertexGCS(ctx context.Context, cfg VertexGCSConfig) (*VertexGCSProcessor, error) {
	if cfg.Project == "" || cfg.Location == "" || cfg.Bucket == "" || cfg.Model == "" {
		return nil, fmt.Errorf("batch: vertex gcs requires Project, Location, Bucket, Model")
	}
	var stOpts []option.ClientOption
	gc := &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  cfg.Project,
		Location: cfg.Location,
	}
	if cfg.CredentialsFile != "" {
		creds, err := credentials.DetectDefault(&credentials.DetectOptions{
			CredentialsFile: cfg.CredentialsFile,
			Scopes:          []string{"https://www.googleapis.com/auth/cloud-platform"},
		})
		if err != nil {
			return nil, fmt.Errorf("batch: vertex gcs detect credentials: %w", err)
		}
		gc.Credentials = creds
		stOpts = append(stOpts, option.WithCredentialsFile(cfg.CredentialsFile))
	}
	gClient, err := genai.NewClient(ctx, gc)
	if err != nil {
		return nil, fmt.Errorf("batch: vertex gcs genai client: %w", err)
	}
	stClient, err := storage.NewClient(ctx, stOpts...)
	if err != nil {
		return nil, fmt.Errorf("batch: vertex gcs storage client: %w", err)
	}
	return &VertexGCSProcessor{genai: gClient, store: stClient, cfg: cfg}, nil
}

// Close releases the storage client.
func (p *VertexGCSProcessor) Close() error { return p.store.Close() }

// Submit builds the input JSONL, uploads it to GCS, and creates a Vertex batch
// job. It returns immediately with a handle — it does NOT wait for completion.
func (p *VertexGCSProcessor) Submit(ctx context.Context, requests []Request) (*VertexBatchHandle, error) {
	if len(requests) == 0 {
		return nil, fmt.Errorf("batch: vertex gcs submit: no requests")
	}
	runID := uuid.NewString()
	base := strings.Trim(p.cfg.Prefix, "/")
	if base != "" {
		base += "/"
	}
	inputObj := fmt.Sprintf("%svertex-batch/%s/input.jsonl", base, runID)
	outputDir := fmt.Sprintf("gs://%s/%svertex-batch/%s/output/", p.cfg.Bucket, base, runID)

	var sb strings.Builder
	for i := range requests {
		line, err := p.buildJSONLLine(&requests[i])
		if err != nil {
			return nil, fmt.Errorf("batch: vertex gcs submit: build request %s: %w", requests[i].ID, err)
		}
		sb.Write(line)
		sb.WriteByte('\n')
	}

	w := p.store.Bucket(p.cfg.Bucket).Object(inputObj).NewWriter(ctx)
	w.ContentType = "application/jsonl"
	if _, err := io.WriteString(w, sb.String()); err != nil {
		_ = w.Close()
		return nil, fmt.Errorf("batch: vertex gcs upload: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("batch: vertex gcs upload close: %w", err)
	}

	job, err := p.genai.Batches.Create(ctx, p.cfg.Model,
		&genai.BatchJobSource{
			Format: "jsonl",
			GCSURI: []string{fmt.Sprintf("gs://%s/%s", p.cfg.Bucket, inputObj)},
		},
		&genai.CreateBatchJobConfig{
			DisplayName: "overtura-extract-" + runID,
			Dest:        &genai.BatchJobDestination{Format: "jsonl", GCSURI: outputDir},
		},
	)
	if err != nil {
		return nil, fmt.Errorf("batch: vertex gcs create job: %w", err)
	}
	return &VertexBatchHandle{
		BatchName: job.Name,
		InputURI:  fmt.Sprintf("gs://%s/%s", p.cfg.Bucket, inputObj),
		OutputURI: outputDir,
		Count:     len(requests),
	}, nil
}

// Collect polls the batch job. If it is not yet terminal it returns
// (nil, false, nil) — the caller should snooze and retry. On a terminal
// success it downloads + parses the GCS output and returns (resp, true, nil).
// A terminal failure returns (nil, true, error).
func (p *VertexGCSProcessor) Collect(ctx context.Context, h *VertexBatchHandle) (*Response, bool, error) {
	job, err := p.genai.Batches.Get(ctx, h.BatchName, nil)
	if err != nil {
		return nil, false, fmt.Errorf("batch: vertex gcs get job: %w", err)
	}
	switch job.State {
	case genai.JobStateSucceeded, genai.JobStatePartiallySucceeded:
		resp, err := p.downloadResults(ctx, job, h)
		if err != nil {
			return nil, true, err
		}
		return resp, true, nil
	case genai.JobStateFailed, genai.JobStateCancelled, genai.JobStateExpired:
		msg := string(job.State)
		if job.Error != nil {
			msg = job.Error.Message
		}
		return nil, true, fmt.Errorf("batch: vertex gcs job %s: %s", job.State, msg)
	default:
		return nil, false, nil // PENDING / QUEUED / RUNNING
	}
}

// CleanupGCS best-effort deletes the input + output objects for a collected
// batch. Errors are returned for logging but are non-fatal to the caller.
func (p *VertexGCSProcessor) CleanupGCS(ctx context.Context, h *VertexBatchHandle) error {
	var errs []string
	for _, uri := range []string{h.InputURI, h.OutputURI} {
		prefix := strings.TrimPrefix(strings.TrimPrefix(uri, "gs://"+p.cfg.Bucket+"/"), "/")
		it := p.store.Bucket(p.cfg.Bucket).Objects(ctx, &storage.Query{Prefix: prefix})
		for {
			attr, err := it.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				errs = append(errs, err.Error())
				break
			}
			if derr := p.store.Bucket(p.cfg.Bucket).Object(attr.Name).Delete(ctx); derr != nil {
				errs = append(errs, derr.Error())
			}
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("batch: vertex gcs cleanup: %s", strings.Join(errs, "; "))
	}
	return nil
}

// buildJSONLLine renders one Vertex batch input line:
// {"request": {contents, systemInstruction, generationConfig}} with the
// correlation marker injected into the first user part.
func (p *VertexGCSProcessor) buildJSONLLine(req *Request) ([]byte, error) {
	contents, system := convertMessagesToGemini(req.Messages)
	injectVertexMarker(contents, req.ID)

	reqObj := map[string]any{"contents": contents}
	if len(system) > 0 {
		reqObj["systemInstruction"] = map[string]any{
			"parts": []any{map[string]any{"text": strings.Join(system, "\n\n")}},
		}
	}
	genCfg := map[string]any{}
	if p.cfg.MaxTokens > 0 {
		genCfg["maxOutputTokens"] = p.cfg.MaxTokens
	}
	if req.OutputSchema != nil {
		genCfg["responseMimeType"] = "application/json"
		genCfg["responseSchema"] = convertToGenaiSchema(req.OutputSchema.Parameters, req.OutputSchema.Required)
	}
	if len(genCfg) > 0 {
		reqObj["generationConfig"] = genCfg
	}
	return json.Marshal(map[string]any{"request": reqObj})
}

// injectVertexMarker prepends the correlation marker to the first user
// content's first text part (or adds one).
func injectVertexMarker(contents []*genai.Content, id string) {
	marker := fmt.Sprintf("[[%s:%s]] ", vertexReqMarkerPrefix, id)
	for _, c := range contents {
		if c.Role != string(message.User) && c.Role != "user" {
			continue
		}
		if len(c.Parts) > 0 && c.Parts[0].Text != "" {
			c.Parts[0].Text = marker + c.Parts[0].Text
		} else {
			c.Parts = append([]*genai.Part{{Text: marker}}, c.Parts...)
		}
		return
	}
}

// downloadResults lists the job's output directory, downloads each predictions
// JSONL, and parses every line into a Result correlated by the embedded marker.
func (p *VertexGCSProcessor) downloadResults(ctx context.Context, job *genai.BatchJob, h *VertexBatchHandle) (*Response, error) {
	outURI := h.OutputURI
	if job.Dest != nil && job.Dest.GCSURI != "" {
		outURI = job.Dest.GCSURI
	}
	if job.OutputInfo != nil && job.OutputInfo.GCSOutputDirectory != "" {
		outURI = job.OutputInfo.GCSOutputDirectory
	}
	prefix := strings.TrimPrefix(strings.TrimPrefix(outURI, "gs://"+p.cfg.Bucket+"/"), "/")

	var results []Result
	it := p.store.Bucket(p.cfg.Bucket).Objects(ctx, &storage.Query{Prefix: prefix})
	for {
		attr, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("batch: vertex gcs list output: %w", err)
		}
		if !strings.HasSuffix(attr.Name, ".jsonl") {
			continue
		}
		r, err := p.store.Bucket(p.cfg.Bucket).Object(attr.Name).NewReader(ctx)
		if err != nil {
			return nil, fmt.Errorf("batch: vertex gcs open output: %w", err)
		}
		data, rerr := io.ReadAll(r)
		_ = r.Close()
		if rerr != nil {
			return nil, fmt.Errorf("batch: vertex gcs read output: %w", rerr)
		}
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			results = append(results, parseVertexResultLine(line))
		}
	}

	completed, failed := 0, 0
	for i := range results {
		if results[i].Err != nil {
			failed++
		} else {
			completed++
		}
	}
	return &Response{Results: results, Completed: completed, Failed: failed, Total: len(results)}, nil
}

type vertexOutLine struct {
	Request struct {
		Contents []struct {
			Role  string `json:"role"`
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"contents"`
	} `json:"request"`
	Response *struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		UsageMetadata *struct {
			PromptTokenCount        int64 `json:"promptTokenCount"`
			CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
			CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
		} `json:"usageMetadata"`
	} `json:"response"`
	Status json.RawMessage `json:"status"`
}

// parseVertexResultLine turns one output JSONL line into a Result, correlating
// by the marker echoed in the request. A line without a usable response becomes
// a per-item error (so the caller leaves that record pending for re-sweep).
func parseVertexResultLine(line string) Result {
	var ol vertexOutLine
	if err := json.Unmarshal([]byte(line), &ol); err != nil {
		return Result{Err: fmt.Errorf("batch: vertex gcs parse line: %w", err)}
	}
	id := extractVertexMarker(&ol)
	statusErr := vertexStatusError(ol.Status)
	if ol.Response == nil || len(ol.Response.Candidates) == 0 {
		if statusErr != "" {
			return Result{ID: id, Err: fmt.Errorf("batch: vertex gcs item error: %s", statusErr)}
		}
		return Result{ID: id, Err: fmt.Errorf("batch: vertex gcs empty response")}
	}
	var content string
	for _, part := range ol.Response.Candidates[0].Content.Parts {
		if part.Text != "" {
			content = part.Text
			break
		}
	}
	usage := llm.TokenUsage{}
	if ol.Response.UsageMetadata != nil {
		usage = llm.TokenUsage{
			InputTokens:     ol.Response.UsageMetadata.PromptTokenCount,
			OutputTokens:    ol.Response.UsageMetadata.CandidatesTokenCount,
			CacheReadTokens: ol.Response.UsageMetadata.CachedContentTokenCount,
		}
	}
	if content == "" {
		return Result{ID: id, Err: fmt.Errorf("batch: vertex gcs empty content")}
	}
	return Result{
		ID: id,
		ChatResponse: &llm.Response{
			Content:      content,
			Usage:        usage,
			FinishReason: message.FinishReasonEndTurn,
		},
	}
}

func extractVertexMarker(ol *vertexOutLine) string {
	for _, c := range ol.Request.Contents {
		for _, part := range c.Parts {
			if m := vertexReqMarkerRE.FindStringSubmatch(part.Text); m != nil {
				return m[1]
			}
		}
	}
	return ""
}

func vertexStatusError(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == `""` || s == "null" {
		return ""
	}
	return s
}
