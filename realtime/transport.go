package realtime

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/coder/websocket"
)

// wsConn is the subset of *coder/websocket.Conn this package uses. An
// interface so tests drive an in-memory fake instead of opening a socket;
// *websocket.Conn satisfies it structurally.
//
// # The context on Read and Write is not a deadline
//
// coder/websocket closes the ENTIRE connection when the context passed to
// Read or Write expires — there is no way to abandon one frame and keep the
// session. It cost a probe run to learn, so it is stated here at the seam
// rather than left to be rediscovered: callers pass the connection's own
// long-lived context and bound their waiting some other way.
type wsConn interface {
	Read(ctx context.Context) (websocket.MessageType, []byte, error)
	Write(ctx context.Context, typ websocket.MessageType, data []byte) error
	Close(code websocket.StatusCode, reason string) error
}

// wsDialer abstracts establishing a wsConn so tests inject a fake transport.
type wsDialer interface {
	Dial(ctx context.Context, urlStr string, header http.Header) (wsConn, error)
}

// realDialer is the production wsDialer: a genuine WebSocket handshake.
//
// This dialer deliberately carries NO SSRF-safe IP allowlist. Such a guard
// belongs on a client whose URL is operator-supplied and may point inside a
// private network, where a hostile value could aim it at a metadata endpoint.
// This endpoint is a fixed public host compiled into the package (see
// defaultBaseURL), not configuration, so there is nothing for an
// allowlist to protect and pretending otherwise would only imply a guard that
// is not doing the work.
type realDialer struct {
	readLimit int64
	timeout   time.Duration
}

func (d realDialer) Dial(ctx context.Context, urlStr string, header http.Header) (wsConn, error) {
	client := &http.Client{
		Timeout: d.timeout,
		Transport: &http.Transport{
			DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
			ForceAttemptHTTP2:     false, // the WS handshake needs HTTP/1.1 Upgrade
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	}
	conn, resp, err := websocket.Dial(ctx, urlStr, &websocket.DialOptions{
		HTTPClient: client,
		HTTPHeader: header,
	})
	if err != nil {
		// The handshake's status is the only thing that distinguishes a bad
		// key from a bad network, and callers act on that difference: one is
		// worth retrying and the other never is.
		if resp != nil {
			return nil, fmt.Errorf("realtime: dialing %s: %w (http %s)", urlStr, err, resp.Status)
		}
		return nil, fmt.Errorf("realtime: dialing %s: %w", urlStr, err)
	}
	if d.readLimit > 0 {
		conn.SetReadLimit(d.readLimit)
	}
	return conn, nil
}
