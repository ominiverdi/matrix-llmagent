# matrix-llmagent

A modern Python-based agentic LLM chatbot service for Matrix with persistent memory, tool use, and local model support.

## Features

- **🟣 Matrix Protocol**: Native Matrix client with full room support
- **🧠 Local LLMs**: llama.cpp support for local model inference
- **🤖 AI Integrations**: Anthropic Claude, OpenAI, DeepSeek, any OpenRouter model, Perplexity AI
- **🛠️ Agentic Capability**: Visit websites, view images, deep research, execute Python code, publish artifacts
- **📚 Persistent Memory**: Chronicle system maintains continuous memory across conversations
- **🎭 Multiple Modes**: Sarcastic, serious, unsafe, agent, and perplexity modes
- **🎯 Proactive Interjecting**: Automatically joins relevant conversations
- **⚡ Async Architecture**: Non-blocking message processing
- **🔒 Rate Limiting**: Configurable rate limiting per user

## Installation

1. Install dependencies: `uv sync --dev`
2. Copy `config.json.example` to `config.json`
3. Configure Matrix credentials and API keys
4. Run: `uv run matrix-llmagent`

## Matrix Setup

Get your Matrix credentials:
1. Create a Matrix account (or use existing)
2. Get access token from Element: Settings → Help & About → Access Token
3. Add to `config.json`:

```json
{
  "matrix": {
    "homeserver": "https://matrix.org",
    "user_id": "@yourbot:matrix.org",
    "access_token": "your_token_here",
    "device_id": "MATRIX_LLMAGENT"
  }
}
```

## Local LLM Setup (llama.cpp)

Run local models with llama.cpp:

1. Install llama.cpp server:
```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

2. Start llama.cpp server:
```bash
# Example with Llama 3.1 8B
./build/bin/llama-server -m models/llama-3.1-8b-instruct.gguf --port 8080
```

3. Configure in `config.json`:
```json
{
  "providers": {
    "llamacpp": {
      "base_url": "http://localhost:8080/v1",
      "key": "not-needed",
      "max_tokens": 2048
    }
  },
  "matrix": {
    "command": {
      "modes": {
        "serious": {
          "model": "llamacpp:llama-3.1-8b-instruct"
        }
      }
    }
  }
}
```

Recommended models for llama.cpp:
- **Llama 3.1/3.2** (8B, 70B) - Great all-around performance
- **Qwen 2.5** (7B, 14B, 32B) - Excellent reasoning
- **Mistral** (7B) - Fast and efficient
- **DeepSeek** - Good for coding tasks

Download models from [Hugging Face](https://huggingface.co/models?library=gguf&sort=trending)

## Configuration

Edit `config.json` based on `config.json.example` to set:
- API keys
- Paths for tools
- Custom prompts for various modes
- Matrix connection settings (homeserver, credentials)

## Commands

- `@botname: message` - Automatic mode
- `@botname: !h` - Show help and info about other modes

## CLI Testing Mode

You can test the bot's message handling including command parsing from the command line:

```bash
uv run matrix-llmagent --message "!h"
uv run matrix-llmagent --message "tell me a joke"
uv run matrix-llmagent --message "!d tell me a joke"
uv run matrix-llmagent --message "!a summarize https://python.org" --config /path/to/config.json
```

This simulates message handling including command parsing and automatic mode classification, useful for testing your configuration and API keys.

### Chronicler

The Chronicler maintains persistent memory across conversations using a Chronicle (arcs → chapters → paragraphs) provided via a NLI-based subagent.

```bash
# Record information
uv run matrix-llmagent --chronicler "Record: Completed API migration" --arc "project-x"

# View current chapter
uv run matrix-llmagent --chronicler "Show me the current chapter" --arc "project-x"
```

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright

# Install pre-commit hooks
uv run pre-commit install
```

## Evaluation of some internal components

### Classifier Analysis

Evaluate the performance of the automatic mode classifier on historical data:

```bash
# Analyze classifier performance on database history
uv run python analyze_classifier.py --db chat_history.db

# Analyze classifier performance on IRC log files
uv run python analyze_classifier.py --logs ~/.irssi/logs/freenode/*.log

# Combine both sources with custom config
uv run python analyze_classifier.py --db chat_history.db --logs ~/.irssi/logs/ --config config.json
```

Results are saved to `classifier_analysis.csv` with detailed metrics and misclassification analysis.

### Proactive Interjecting Analysis

Evaluate the performance of the proactive interjecting feature on historical data:

```bash
# Analyze proactive interjecting performance on database history
uv run python analyze_proactive.py --limit 20

# Analyze proactive interjecting on IRC log files with channel exclusions
uv run python analyze_proactive.py --logs ~/.irssi/logs/ --limit 50 --exclude-news

# Combine both sources with custom config
uv run python analyze_proactive.py --db chat_history.db --logs ~/.irssi/logs/ --config config.json
```

Results are saved to `proactive_analysis.csv` with detailed interjection decisions and reasoning.
