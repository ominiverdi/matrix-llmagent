# matrix-llmagent

A modern Python-based agentic LLM chatbot service for Matrix with persistent memory, tool use, and local model support.

---

*Forked from [pasky/irssi-llmagent](https://github.com/pasky/irssi-llmagent) - adapted for Matrix protocol*

---

## Features

- **Matrix Protocol**: Native Matrix client with full room support
- **Local LLMs**: llama.cpp support for local model inference
- **AI Integrations**: Anthropic Claude, OpenAI, DeepSeek, any OpenRouter model, Perplexity AI
- **Agentic Capability**: Visit websites, view images, deep research, execute Python code, publish artifacts
- **Persistent Memory**: Chronicle system maintains continuous memory across conversations
- **Multiple Modes**: Sarcastic, serious, unsafe, agent, and perplexity modes
- **Proactive Interjecting**: Automatically joins relevant conversations
- **Async Architecture**: Non-blocking message processing
- **Rate Limiting**: Configurable rate limiting per user

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

2. Start llama.cpp server with tool support:
```bash
# Example with Llama 3.1 8B
# IMPORTANT: Use --jinja flag for tool calling support (web browsing, code execution, etc.)
./build/bin/llama-server -m models/llama-3.1-8b-instruct.gguf --port 8080 --jinja
```

> **Note**: The `--jinja` flag enables tool calling, which is required for web browsing, code execution, and other agentic features. Without it, the bot will work but won't be able to use tools.

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

### Quick Start: Webpage Visitor

The webpage visitor is **enabled by default** in local mode (no setup needed):

```json
{
  "tools": {
    "webpage_visitor": "local"  // Already configured, works out of the box
  }
}
```

**Want better quality?** Switch to Jina mode (optional):
```json
{
  "tools": {
    "webpage_visitor": "jina",
    "jina": {
      "api_key": "your-jina-api-key"  // Get from https://jina.ai
    }
  }
}
```

**Customize the User-Agent** (optional for local mode):
```json
{
  "tools": {
    "user_agent": "Mozilla/5.0 (compatible; YourBotName/1.0; +https://your-repo-url)"
  }
}
```

## Commands

- `@botname: message` - **Default (serious) mode** with web browsing enabled
- `@botname: !h` - Show help and info about other modes
- `@botname: !s <message>` - Serious mode (same as default - web tools enabled)
- `@botname: !d <message>` - Sarcastic mode (witty, humorous responses)
- `@botname: !a <message>` - Agent mode (advanced multi-turn research)
- `@botname: !p <message>` - Perplexity mode (web search with Perplexity AI)
- `@botname: !u <message>` - Unsafe mode (uncensored responses)

**Quick Examples:**
```
@bot: visit https://python.org and summarize
@bot: search for the latest news about AI
@bot: !a research async Python patterns and create a comparison guide
@bot: !d why do programmers prefer dark mode?
```

## Bot Modes

### Help (`!h`)
Show available commands, modes, and tools. Use this to discover what the bot can do.

```
@bot: !h
```

### Serious Mode (`!s`) - Default Mode with Web Tools
**Default mode** with web browsing capabilities enabled. The bot can:
- Visit and analyze webpages (using local webpage visitor)
- Search the web
- Execute Python code
- Provide thoughtful, informative responses

Ideal for:
- Technical questions with web research
- Visiting and summarizing webpages
- General knowledge queries with sources
- Documentation requests

**Example interactions:**
```
@bot: visit https://python.org and tell me what's new
@bot: summarize the article at https://example.com/blog/post
@bot: search for Python async best practices
```

### Sarcastic Mode (`!d`)
Witty, humorous responses with a cynical edge. Great for:
- Casual banter
- Jokes and memes
- Light-hearted conversations

### Agent Mode (`!a`) - Advanced Multi-Turn Research
**Advanced mode** for complex multi-turn research tasks. Same tools as serious mode, but with:
- Multi-turn autonomous task breakdown
- Progress reporting
- More thorough research approach
- Better for complex, multi-step tasks

**Example interactions:**
```
@bot: !a analyze the Python 3.13 release notes and compare with 3.12
@bot: !a research async Python patterns and create a detailed comparison
@bot: !a investigate https://github.com/python/cpython recent changes and summarize top 5
```

> **Note**: For simple webpage visits or single searches, use **serious mode** (default) without `!a` prefix!

### Perplexity Mode (`!p`)
Uses Perplexity AI for web-enhanced responses with real-time information.

### Unsafe Mode (`!u`)
Unrestricted responses that bypass typical LLM safety filters. Use responsibly.

## Tools & Capabilities

**By default**, the bot has access to powerful tools in **serious mode** (no prefix needed). These same tools are available in agent mode (`!a`) for more complex multi-turn tasks:

### Web Search (`web_search`)
Search the web and get top results with titles, URLs, and descriptions.

**Supported Providers:**
- **Jina AI** - Requires API key, best quality ([get one](https://jina.ai/)) - **recommended**
- **Google** - Requires API key + Custom Search Engine ID ([setup guide](https://developers.google.com/custom-search/v1/overview))
- **Brave Search** - Requires API key ([get one](https://brave.com/search/api/))
- **DuckDuckGo** - Free, no API key required (use `"ddgs"` or `"auto"`)
- **Wikipedia** - Free, no API key required

**Configuration:**
```json
{
  "tools": {
    "search_provider": "jina",
    "jina": {
      "api_key": "your-jina-api-key"
    },
    "google": {
      "api_key": "your-google-api-key",
      "cx": "your-custom-search-engine-id"
    },
    "brave": {
      "api_key": "your-brave-api-key"
    }
  }
}
```

**Google Custom Search Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the "Custom Search API"
3. Create an API key in "Credentials"
4. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/) and create a search engine
5. Copy the Search Engine ID (cx) from "Setup" > "Basic"
6. Add both `api_key` and `cx` to your config

**Disable web search entirely:**
Remove `search_provider` from config or set it to `""`, `"none"`, or `"disabled"`. The `web_search` tool will not be offered to the agent.

### 📄 Web Crawler (`visit_webpage`)
Visit and analyze any webpage, converting HTML to clean Markdown.

**Two Modes Available:**

**1. Local Mode** (Privacy-Friendly, Default)
- Fetches and converts HTML locally using readability + markdownify
- No external API calls - your URLs stay private
- Extracts main article content, removes navigation/ads
- Works offline (no rate limits)
- Supports custom User-Agent configuration

**2. Jina.ai Mode** (Higher Quality)
- Uses Jina.ai Reader service for conversion
- Better quality Markdown output
- Handles JavaScript-rendered content better
- Requires internet connection
- Free tier: 20 requests/min, paid: 500 req/min

**Common Features:**
- Direct image fetching (JPG, PNG, GIF, WebP)
- Handles up to 40KB of text content
- Automatic retries on failures
- Content truncation warnings

**Configuration:**
```json
{
  "tools": {
    "webpage_visitor": "local",  // or "jina"
    "user_agent": "Mozilla/5.0 (compatible; matrix-llmagent/1.0; +https://github.com/yourusername/matrix-llmagent)",
    "jina": {
      "api_key": "your-jina-api-key"  // Only needed for jina mode
    }
  }
}
```

**Example:**
```
User: !a visit https://python.org/downloads and tell me about the latest version
Bot: [Fetches page locally, converts to Markdown, analyzes content]
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
     Generated plot showing Python, JavaScript, Java, C++, Go
     [Returns plot data/description]
```

### Artifact Sharing (`share_artifact`)
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

### Image Generation (`generate_image`)
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
     Image generated: [URL to image]
```

### Knowledge Base Search (`knowledge_base`)
Search a custom PostgreSQL knowledge base with semantic extractions. Perfect for domain-specific Q&A using your own wiki or document corpus.

**Features:**
- Full-text search on page summaries, keywords, and titles
- Entity search with fuzzy matching (people, organizations, projects)
- Relationship queries (knowledge graph)
- Fast async PostgreSQL queries via connection pooling
- Configurable result limits

**Use Cases:**
- Organization wikis (OSGeo, company docs, etc.)
- Domain-specific knowledge bases
- Any corpus with entity extraction

**Configuration:**
```json
{
  "tools": {
    "knowledge_base": {
      "enabled": true,
      "database_url": "postgresql:///your_database",
      "name": "Your Knowledge Base",
      "description": "Search for information about projects, people, and events.",
      "max_results": 5,
      "max_entities": 10
    }
  }
}
```

**Pro Tip:** Add a hint in your system prompt to prioritize the knowledge base:
```json
{
  "modes": {
    "serious": {
      "system_prompt": "You are a helpful assistant. You have access to a Knowledge Base tool - use it FIRST for domain-specific questions before trying web search."
    }
  }
}
```

**Database Schema:** See [docs/KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md) for the expected PostgreSQL schema and setup guide.

**Example:**
```
User: Who is strk in the OSGeo community?
Bot: [Searches knowledge_base for "strk"]
     Based on the OSGeo Wiki, "strk" is Sandro Santilli, a prominent
     OSGeo contributor involved in PostGIS and infrastructure projects...
```

### Other Agent Tools

- **`progress_report`** - Send real-time progress updates during long operations
- **`make_plan`** - Formulate research/execution strategy before acting
- **`final_answer`** - Structured final response with thinking process
- **Chronicle tools** - Access conversation history and memory

### Collapsible Long Messages

Long bot responses can optionally be wrapped in a collapsible `<details>` tag for Matrix clients that support HTML rendering.

**Configuration:**
```json
{
  "behavior": {
    "collapsible_messages": true,
    "max_message_length": 300
  }
}
```

- `collapsible_messages`: Set to `true` to enable (default: `false`)
- `max_message_length`: Character threshold for collapsing (default: 300)

### Tool Configuration

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
| `web_search` (Wikipedia/DDG) | No | Free | None |
| `web_search` (Google) | Yes | Free tier (100/day) | [Google Custom Search](https://developers.google.com/custom-search/v1/overview) |
| `web_search` (Brave) | Yes | Paid | [Brave API](https://brave.com/search/api/) |
| `web_search` (Jina) | Yes | Free tier available | [Jina AI](https://jina.ai/) |
| `visit_webpage` (local) | No | Free | Built-in (default) |
| `visit_webpage` (jina) | Optional | Free tier / Paid | [Jina API](https://jina.ai/) for higher quality |
| `execute_python` | Yes | Paid | [E2B](https://e2b.dev) |
| `generate_image` | Yes | Paid | OpenRouter account |
| `share_artifact` | No | Free | Configure local path + URL |
| `knowledge_base` | No | Free | PostgreSQL database ([schema](docs/KNOWLEDGE_BASE.md)) |

**Note:** For llama.cpp models to support tools, start the server with `--jinja` flag:
```bash
./llama-server -m model.gguf --port 8080 --jinja
```

### Multi-Turn Agent Flow

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

Test the bot locally without connecting to Matrix. This is useful for:
- Testing your configuration and API keys
- Verifying tool functionality (knowledge base, web search, etc.)
- Debugging prompts and model behavior
- Development and experimentation

### Basic Usage

```bash
# Default (serious mode)
uv run matrix-llmagent --message "What is QGIS?"

# With custom config file
uv run matrix-llmagent --message "Hello" --config /path/to/config.json
```

### Mode Prefixes

Use the same mode prefixes as in Matrix:

```bash
# Show help with all available commands
uv run matrix-llmagent --message "!h"

# Serious mode (default) - with web tools
uv run matrix-llmagent --message "What is PostGIS?"
uv run matrix-llmagent --message "!s What is PostGIS?"

# Sarcastic mode - witty responses
uv run matrix-llmagent --message "!d Tell me a GIS joke"

# Agent mode - multi-turn research
uv run matrix-llmagent --message "!a Research FOSS4G 2024 and summarize"

# Unsafe mode - uncensored
uv run matrix-llmagent --message "!u Controversial question"

# Perplexity mode - web-enhanced AI
uv run matrix-llmagent --message "!p Latest news about open source GIS"
```

### Output

The CLI shows:
- Mode and model being used
- Tool calls and results (in logs)
- Final response

```
Mode: serious
Model: llamacpp:qwen3-coder-30b-32k
Query: What is QGIS?
------------------------------------------------------------
[tool calls and processing...]
------------------------------------------------------------
QGIS is an Open Source Geographic Information System...
```

### Testing Knowledge Base

If you have a knowledge base configured:

```bash
# Test knowledge base queries
uv run matrix-llmagent --message "Who is strk in OSGeo?"
uv run matrix-llmagent --message "What FOSS4G conferences happened in 2023?"
```

### Requirements

- `config.json` must exist with valid configuration
- For llama.cpp: server must be running with `--jinja` flag
- For cloud providers: API keys must be configured

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
