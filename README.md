# matrix-llmagent

A modern Python-based agentic LLM chatbot service for Matrix with persistent memory, tool use, and local model support.

## Features

- **AI Integrations**: Anthropic Claude, OpenAI, DeepSeek, any OpenRouter model, Perplexity AI
- **Agentic Capability**: Ability to visit websites, view images, perform deep research, execute Python code, publish artifacts
- **Command System**: Extensible command-based interaction with prefixes for various modes
- **Proactive Interjecting**: Channel-based whitelist system for automatic participation in relevant conversations
- **Restartable and Persistent Memory**: All state is persisted; AI agent maintains a continuous chronicle of events and experiences to refer to
- **Async Architecture**: Non-blocking message processing with concurrent handling
- **Modern Python**: Built with uv, type safety, and comprehensive testing
- **Rate Limiting**: Configurable rate limiting and user management

## Installation

1. Install dependencies: `uv sync --dev`
2. Copy `config.json.example` to `config.json` and configure your API keys
3. Set up Matrix credentials in config.json
4. Run the service: `uv run matrix-llmagent`

Note: This is currently in migration from IRC to Matrix. See PLAN.md for details.

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
