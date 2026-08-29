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
// interface so tests drive an in-memory fake instead of opening a socket.
//
// The context passed to Read or Write is not a per-call deadline: expiring it
// closes the whole connection, so callers pass its long-lived context and
// bound their waiting some other way.
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
// It carries no SSRF allowlist: the endpoint is a fixed host compiled into
// the package (defaultBaseURL), not caller-supplied configuration.
type realDialer struct {
	readLimit int64
	timeout   time.Duration
}

// Dial opens a WebSocket connection to urlStr. HTTP/2 is disabled on the
// underlying transport because the WebSocket handshake needs HTTP/1.1
// Upgrade.
func (d realDialer) Dial(
	ctx context.Context,
	urlStr string,
	header http.Header,
) (wsConn, error) {
	client := &http.Client{
		Timeout: d.timeout,
		Transport: &http.Transport{
			DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
			ForceAttemptHTTP2:     false,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	}
	//nolint:bodyclose // coder/websocket closes resp.Body itself; see its Dial doc.
	conn, resp, err := websocket.Dial(ctx, urlStr, &websocket.DialOptions{
		HTTPClient: client,
		HTTPHeader: header,
	})
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf(
				"realtime: dialing %s: %w (http %s)",
				urlStr,
				err,
				resp.Status,
			)
		}
		return nil, fmt.Errorf("realtime: dialing %s: %w", urlStr, err)
	}
	if d.readLimit > 0 {
		conn.SetReadLimit(d.readLimit)
	}
	return conn, nil
}
