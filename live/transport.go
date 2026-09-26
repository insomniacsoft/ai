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
// endpoint; *websocket.Conn satisfies it structurally.
//
// # The context on Read and Write is not a deadline
//
// coder/websocket closes the ENTIRE connection when the context passed to
// Read or Write expires — there is no way to abandon one frame and keep the
// session. Callers pass the connection's own long-lived context and bound
// their waiting some other way (see Close, which cancels that context only
// once it is actually done with the socket).
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
// Unlike a client whose endpoint is compiled in, Config.BaseURL is
// caller-supplied — tests point it at an httptest server, and an operator
// could in principle point it at a compatible endpoint. That makes it
// configuration the caller already trusts at the same level as the API key,
// not untrusted input from a hostile source, so this dialer carries no
// SSRF-safe IP allowlist: there is nothing here for one to protect that the
// caller has not already accepted by supplying the URL.
type realDialer struct {
	httpClient *http.Client
	readLimit  int64
}

func (d realDialer) Dial(ctx context.Context, urlStr string, header http.Header) (wsConn, error) {
	client := d.httpClient
	if client == nil {
		client = &http.Client{
			Transport: &http.Transport{
				DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
				ForceAttemptHTTP2:     false, // the WS handshake needs HTTP/1.1 Upgrade
				TLSHandshakeTimeout:   10 * time.Second,
				ExpectContinueTimeout: 1 * time.Second,
			},
		}
	}
	conn, resp, err := websocket.Dial(ctx, urlStr, &websocket.DialOptions{
		HTTPClient: client,
		HTTPHeader: header,
	})
	if err != nil {
		// The handshake's status is the only thing that distinguishes a bad
		// key from a bad network, and callers act on that difference: one is
		// worth reporting as ErrUnauthorized and the other is not.
		if resp != nil {
			return nil, fmt.Errorf("live: dialing %s: %w (http %s)", urlStr, err, resp.Status)
		}
		return nil, fmt.Errorf("live: dialing %s: %w", urlStr, err)
	}
	if d.readLimit > 0 {
		conn.SetReadLimit(d.readLimit)
	}
	return conn, nil
}
