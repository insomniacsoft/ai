package live

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/coder/websocket"
)

// wsConn is the subset of *coder/websocket.Conn this package uses. An
// interface so tests drive a fake WebSocket server instead of the real
// endpoint.
//
// The context passed to Read or Write is not a per-call deadline: expiring
// it closes the whole connection, so callers pass its long-lived context
// and bound their waiting some other way (see Close).
type wsConn interface {
	Read(ctx context.Context) (websocket.MessageType, []byte, error)
	Write(ctx context.Context, typ websocket.MessageType, data []byte) error
	Close(code websocket.StatusCode, reason string) error
	// CloseNow drops the connection without a close handshake. Close waits
	// for the peer's close frame (up to the library's own five seconds);
	// after a session that already failed to answer session.close, that is
	// time spent on a peer known not to be answering.
	CloseNow() error
}

// wsDialer abstracts establishing a wsConn so tests inject a fake transport.
type wsDialer interface {
	Dial(ctx context.Context, urlStr string, header http.Header) (wsConn, error)
}

// realDialer is the production wsDialer: a genuine WebSocket handshake.
//
// It carries no SSRF allowlist: Config.BaseURL is caller-supplied
// configuration, trusted at the same level as the API key, not untrusted
// input from a hostile source.
type realDialer struct {
	httpClient *http.Client
	readLimit  int64
}

// Dial opens a WebSocket connection to urlStr. HTTP/2 is disabled on the
// underlying transport because the WebSocket handshake needs HTTP/1.1
// Upgrade. The handshake's own HTTP status is the only thing that
// distinguishes a bad key from a bad network, and callers act on that
// difference: one is worth reporting as ErrUnauthorized and the other is
// not.
func (d realDialer) Dial(
	ctx context.Context,
	urlStr string,
	header http.Header,
) (wsConn, error) {
	client := d.httpClient
	if client == nil {
		client = &http.Client{
			Transport: &http.Transport{
				DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
				ForceAttemptHTTP2:     false,
				TLSHandshakeTimeout:   10 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
			},
		}
	}
	//nolint:bodyclose // coder/websocket closes resp.Body itself; see its Dial doc.
	conn, resp, err := websocket.Dial(ctx, urlStr, &websocket.DialOptions{
		HTTPClient: client,
		HTTPHeader: header,
	})
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf(
				"live: dialing %s: %w (http %s)",
				urlStr,
				err,
				resp.Status,
			)
		}
		return nil, fmt.Errorf("live: dialing %s: %w", urlStr, err)
	}
	if d.readLimit > 0 {
		conn.SetReadLimit(d.readLimit)
	}
	return conn, nil
}
