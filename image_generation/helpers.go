package image_generation

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
)

// DownloadImage downloads an image from a URL and returns its binary data.
// This is a helper function for processing image generation responses that return URLs.
func DownloadImage(url string) ([]byte, error) {
	resp, err := http.Get(url) //nolint:noctx // helper function, no context available
	if err != nil {
		return nil, fmt.Errorf("failed to download image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"failed to download image: status code %d",
			resp.StatusCode,
		)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read image data: %w", err)
	}

	return data, nil
}

// DecodeBase64Image decodes a base64-encoded image string into binary data.
// This is a helper function for processing image generation responses with base64 format.
func DecodeBase64Image(b64 string) ([]byte, error) {
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64 image: %w", err)
	}
	return data, nil
}
