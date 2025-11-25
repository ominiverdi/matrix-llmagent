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

- `@botname: message` - Automatic mode (classifier decides which mode to use)
- `@botname: !h` - Show help and info about other modes
- `@botname: !s <message>` - Serious mode (thoughtful, informative responses)
- `@botname: !d <message>` - Sarcastic mode (witty, humorous responses)
- `@botname: !a <message>` - Agent mode (full tool access for research and code execution)
- `@botname: !p <message>` - Perplexity mode (web search with Perplexity AI)
- `@botname: !u <message>` - Unsafe mode (uncensored responses)

## 🤖 Bot Modes

### Serious Mode (`!s`)
Standard mode for helpful, informative responses. Ideal for:
- Technical questions
- General knowledge queries
- Documentation requests

### Sarcastic Mode (`!d`)
Witty, humorous responses with a cynical edge. Great for:
- Casual banter
- Jokes and memes
- Light-hearted conversations

### Agent Mode (`!a`) - Full Agentic Capabilities
**Most powerful mode** with access to all tools for autonomous research and code execution. The agent can:
- Search the web
- Visit and analyze webpages
- Execute Python code in a sandbox
- Generate images
- Create shareable artifacts

**Example interactions:**
```
@bot: !a summarize the latest Python 3.13 release notes
@bot: !a create a visualization of the Fibonacci sequence
@bot: !a search for async Python best practices and show examples
@bot: !a analyze https://github.com/python/cpython and summarize recent changes
```

### Perplexity Mode (`!p`)
Uses Perplexity AI for web-enhanced responses with real-time information.

### Unsafe Mode (`!u`)
Unrestricted responses that bypass typical LLM safety filters. Use responsibly.

## 🛠️ Agent Tools & Capabilities

When using **Agent Mode** (`!a`), the bot has access to powerful tools for autonomous task completion:

### 🌐 Web Search (`web_search`)
Search the web and get top results with titles, URLs, and descriptions.

**Supported Providers:**
- **Wikipedia** - Free, no API key required
- **DuckDuckGo** - Free, no API key required  
- **Brave Search** - Requires API key ([get one](https://brave.com/search/api/))
- **Jina AI** - Requires API key, best quality ([get one](https://jina.ai/))

**Configuration:**
```json
{
  "tools": {
    "search_provider": "jina",
    "jina": {
      "api_key": "your-jina-api-key"
    },
    "brave": {
      "api_key": "your-brave-api-key"
    }
  }
}
```

### 📄 Web Crawler (`visit_webpage`)
Visit and analyze any webpage, converting HTML to clean Markdown.

**Features:**
- Converts webpages to readable Markdown using Jina.ai
- Direct image fetching (JPG, PNG, GIF, WebP)
- Handles up to 40KB of text content
- Automatic retries on failures
- Content truncation warnings

**Example:**
```
User: !a visit https://python.org/downloads and tell me about the latest version
Bot: [Fetches page, converts to Markdown, analyzes content]
     Python 3.13.1 is now available with improved performance...
```

### 🐍 Python Code Execution (`execute_python`)
Execute Python code in a secure **E2B sandbox** environment.

**Features:**
- Persistent sandbox across multiple calls
- 180-second timeout per execution
- Captures stdout, stderr, and return values
- Supports plots and images (matplotlib, seaborn, etc.)
- Full isolation - safe for untrusted code
- Includes popular libraries (numpy, pandas, requests, etc.)

**Setup:**
```bash
# Get API key from https://e2b.dev
export E2B_API_KEY="your-e2b-api-key"
```

**Configuration:**
```json
{
  "tools": {
    "e2b": {
      "api_key": "your-e2b-api-key"
    }
  }
}
```

**Example:**
```
User: !a create a bar chart of top 5 programming languages
Bot: [Executes matplotlib code in sandbox]
     ✅ Generated plot showing Python, JavaScript, Java, C++, Go
     [Returns plot data/description]
```

### 📦 Artifact Sharing (`share_artifact`)
Create shareable public links for scripts, reports, and data.

**Configuration:**
```json
{
  "tools": {
    "artifacts": {
      "path": "/path/to/artifacts",
      "url": "https://yourdomain.com/artifacts"
    }
  }
}
```

**Use Cases:**
- Share generated scripts
- Publish detailed reports
- Provide downloadable data files

### 🎨 Image Generation (`generate_image`)
Generate AI images using state-of-the-art models via OpenRouter.

**Configuration:**
```json
{
  "tools": {
    "image_gen": {
      "model": "openrouter:google/gemini-2.5-flash-preview-image"
    }
  }
}
```

**Supported models:**
- Google Gemini 2.5 Flash (fast, high quality)
- Stability AI models (Stable Diffusion)
- Flux models (artistic style)

**Example:**
```
User: !a generate an image of a futuristic city at sunset
Bot: [Generates image using configured model]
     ✅ Image generated: [URL to image]
```

### 📊 Other Agent Tools

- **`progress_report`** - Send real-time progress updates during long operations
- **`make_plan`** - Formulate research/execution strategy before acting
- **`final_answer`** - Structured final response with thinking process
- **Chronicle tools** - Access conversation history and memory

### 🔧 Tool Configuration

**Enable/Disable Tools Per Mode:**
```json
{
  "matrix": {
    "command": {
      "modes": {
        "serious": {
          "model": "anthropic:claude-sonnet-4",
          "allowed_tools": []  // Disable all tools
        },
        "agent": {
          "model": "anthropic:claude-sonnet-4",
          "allowed_tools": null  // Enable all tools (default)
        }
      }
    }
  }
}
```

**Tool Requirements:**
| Tool | API Key Required | Cost | Setup |
|------|------------------|------|-------|
| `web_search` (Wikipedia/DDG) | ❌ No | Free | None |
| `web_search` (Brave) | ✅ Yes | Paid | [Brave API](https://brave.com/search/api/) |
| `web_search` (Jina) | ✅ Yes | Free tier available | [Jina AI](https://jina.ai/) |
| `visit_webpage` | ⚠️ Optional | Free (rate limited) / Paid | [Jina API](https://jina.ai/) for higher limits |
| `execute_python` | ✅ Yes | Paid | [E2B](https://e2b.dev) |
| `generate_image` | ✅ Yes | Paid | OpenRouter account |
| `share_artifact` | ❌ No | Free | Configure local path + URL |

**Note:** For llama.cpp models to support tools, start the server with `--jinja` flag:
```bash
./llama-server -m model.gguf --port 8080 --jinja
```

### 🎯 Multi-Turn Agent Flow

When you invoke agent mode, the bot operates in a **multi-turn loop**:

1. **Planning** - Analyzes request, formulates approach
2. **Tool Execution** - Calls necessary tools (search, code, web visits)
3. **Iteration** - Can chain multiple tools (up to 5 iterations)
4. **Synthesis** - Combines results into final answer

**Example Flow:**
```
User: "Compare the performance of Python list vs deque"

Agent:
  Turn 1: make_plan("Research Python collections, run benchmarks")
  Turn 2: web_search("python list vs deque performance")
  Turn 3: visit_webpage("https://docs.python.org/3/library/collections.html")
  Turn 4: execute_python("import timeit; # benchmark code...")
  Turn 5: final_answer("Based on benchmarks, deque is 5x faster for...")
```

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

# Analyze classifier performance on chat log files
uv run python analyze_classifier.py --logs ~/logs/*.log

# Combine both sources with custom config
uv run python analyze_classifier.py --db chat_history.db --logs ~/logs/ --config config.json
```

Results are saved to `classifier_analysis.csv` with detailed metrics and misclassification analysis.

### Proactive Interjecting Analysis

Evaluate the performance of the proactive interjecting feature on historical data:

```bash
# Analyze proactive interjecting performance on database history
uv run python analyze_proactive.py --limit 20

# Analyze proactive interjecting on chat log files with channel exclusions
uv run python analyze_proactive.py --logs ~/logs/ --limit 50 --exclude-news

# Combine both sources with custom config
uv run python analyze_proactive.py --db chat_history.db --logs ~/logs/ --config config.json
```

Results are saved to `proactive_analysis.csv` with detailed interjection decisions and reasoning.
