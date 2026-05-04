package embedding

// DefaultConfig returns the ecosystem default Config: Ollama at localhost
// with nomic-embed-text. It currently aliases OllamaLocalNomic.
//
// External users should prefer constructing their own Config explicitly;
// this default exists so internal callers can centralise the choice in
// one place and bump it ecosystem-wide by upgrading this module.
func DefaultConfig() Config {
	return OllamaLocalNomic()
}

// OllamaLocalNomic returns a Config for Ollama running on its default port
// (11434) serving nomic-embed-text.
func OllamaLocalNomic() Config {
	return Config{
		Backend: BackendOllama,
		BaseURL: "http://localhost:11434",
		Model:   "nomic-embed-text",
	}
}

// LemonadeNomic returns a Config for Lemonade Server running on its default
// port (13305) serving nomic-embed-text via the OpenAI-compatible endpoint.
func LemonadeNomic() Config {
	return Config{
		Backend: BackendOpenAI,
		BaseURL: "http://localhost:13305",
		Model:   "nomic-embed-text",
	}
}
