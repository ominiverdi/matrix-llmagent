# Matrix-LLMAgent Migration Plan

## Project Overview

**Goal:** Transform irssi-llmagent (IRC-only) into matrix-llmagent (Matrix-only) with llama.cpp support.

**New Repository Name:** `matrix-llmagent`

**Description:** Agentic LLM chatbot for Matrix with persistent memory, tool use, and local model support via llama.cpp.

---

## Migration Strategy

### What We're Doing
- ✅ **Remove** all IRC-specific code (varlink, irssi integration)
- ✅ **Replace** with Matrix protocol support (matrix-nio)
- ✅ **Add** llama.cpp provider for local LLM support
- ✅ **Keep** all core features: chronicler, agentic tools, multiple modes, proactive interjecting
- ✅ **Rename** package from `irssi_llmagent` to `matrix_llmagent`

### What We're Keeping
- ✅ Provider system (Anthropic, OpenAI, OpenRouter, Perplexity)
- ✅ Agentic actor with tools (web search, code execution, artifacts)
- ✅ Chronicle system (persistent memory)
- ✅ Quest tracking
- ✅ Chat history management
- ✅ Rate limiting
- ✅ Mode classification (sarcastic/serious/unsafe)
- ✅ Proactive interjecting logic

---

## Phase 1: Repository Setup

### Steps

1. **Rename local directory:**
   ```bash
   cd /home/ominiverdi/github
   mv irssi-llmagent matrix-llmagent
   cd matrix-llmagent
   ```

2. **Remove old git origin:**
   ```bash
   git remote remove origin
   ```

3. **Create new GitHub repository:**
   - Go to https://github.com/new
   - Repository name: `matrix-llmagent`
   - Description: "Agentic LLM chatbot for Matrix with persistent memory, tool use, and local model support"
   - Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we have them)
   - Click "Create repository"

4. **Add new origin and push:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/matrix-llmagent.git
   git remote -v  # Verify
   git push -u origin main
   ```

**Estimated Time:** 30 minutes

---

## Phase 2: Package Renaming

### Files to Update

#### A. Rename Python Package Directory
```bash
mv irssi_llmagent matrix_llmagent
```

#### B. Update `pyproject.toml`
```toml
[project]
name = "matrix-llmagent"  # was: irssi-llmagent
description = "Agentic LLM chatbot for Matrix"  # was: AI chatbot for IRC via irssi-varlink

[project.scripts]
matrix-llmagent = "matrix_llmagent.main:main"  # was: irssi-llmagent = "irssi_llmagent.main:main"

[tool.pyright]
include = ["matrix_llmagent", "tests"]  # was: irssi_llmagent

[tool.hatch.build.targets.wheel]
packages = ["matrix_llmagent"]  # was: irssi_llmagent
```

#### C. Update All Python Imports
Search and replace across all `.py` files:
- `from irssi_llmagent` → `from matrix_llmagent`
- `import irssi_llmagent` → `import matrix_llmagent`

Files to update:
- `matrix_llmagent/main.py`
- `matrix_llmagent/providers/*.py`
- `matrix_llmagent/chronicler/*.py`
- `matrix_llmagent/agentic_actor/*.py`
- `tests/**/*.py`

#### D. Update `README.md`
- Change title to `# matrix-llmagent`
- Update description
- Update installation instructions
- Remove IRC-specific documentation
- Add Matrix-specific setup guide

#### E. Update `AGENTS.md`
- Change all references from `irssi-llmagent` to `matrix-llmagent`
- Update command examples

**Estimated Time:** 2-3 hours

---

## Phase 3: Remove IRC Code

### Directories to Delete
```bash
rm -rf matrix_llmagent/rooms/irc/
rm -rf docs/docker.md  # IRC-specific Docker setup
```

### Files to Delete
```bash
rm matrix_llmagent/rooms/proactive.py  # Will recreate Matrix version
```

### Files to Clean Up

#### `matrix_llmagent/rooms/__init__.py`
- Remove IRC imports
- Keep only base classes if any

#### `matrix_llmagent/main.py`
- Remove IRC-specific imports
- Remove `IRCRoomMonitor` instantiation
- Prepare for Matrix monitor

**Estimated Time:** 1 hour

---

## Phase 4: Add Matrix Support

### A. Add Dependencies

Update `pyproject.toml`:
```toml
[project]
dependencies = [
    "aiohttp>=3.9.0",
    "aiosqlite>=0.19.0",
    "ddgs>=0.1.0",
    "markdownify>=0.11.0",
    "e2b-code-interpreter<2",
    "openai>=1.40.0",
    "matrix-nio>=0.24.0",  # NEW: Matrix client library
]
```

### B. Create Matrix Client (`matrix_llmagent/matrix_client.py`)

**Purpose:** Handle Matrix protocol communication

**Key Features:**
- Connect to Matrix homeserver
- Sync with server
- Send/receive messages
- Handle events
- Room membership management
- Support for Matrix features (reactions, edits, threading, HTML formatting)

**Implementation:**
- Use `matrix-nio` library (AsyncClient)
- Async/await pattern
- Event callbacks for messages
- Proper error handling and reconnection logic

**API:**
```python
class MatrixClient:
    async def connect() -> None
    async def sync() -> None
    async def send_message(room_id: str, message: str) -> None
    async def send_html_message(room_id: str, text: str, html: str) -> None
    async def get_next_event() -> dict
    async def get_user_id() -> str
    async def get_display_name() -> str
    async def join_room(room_id: str) -> None
    async def upload_file(file_path: str) -> str  # For artifacts
```

### C. Create Matrix Monitor (`matrix_llmagent/matrix_monitor.py`)

**Purpose:** Replace IRCRoomMonitor with Matrix-specific message handling

**Key Features:**
- Event processing loop
- Command parsing (Matrix mentions vs IRC nick prefixes)
- Mode classification
- Proactive interjecting (Matrix-specific room checks)
- Message formatting (support HTML, reactions)
- Rate limiting
- Auto-chronicling

**Based on:** `rooms/irc/monitor.py` but simplified for Matrix

**Key Differences from IRC:**
- Matrix mentions: `@botname:matrix.org` instead of `botnick:`
- Room IDs instead of channels: `!abc123:matrix.org` instead of `#channel`
- Native HTML support
- Reply threading with `m.relates_to`
- Reactions support
- Better permission model

**API:**
```python
class MatrixMonitor:
    async def run() -> None
    async def process_event(event: dict) -> None
    async def process_message(event: dict) -> None
    async def handle_command(room_id: str, sender: str, message: str) -> None
    async def send_message(room_id: str, message: str) -> None
    def is_addressed_to_bot(message: str, sender: str, room_id: str) -> bool
    async def classify_mode(context: list) -> str
    async def should_interject_proactively(context: list) -> tuple[bool, str, bool]
```

### D. Update Main Application (`matrix_llmagent/main.py`)

**Changes:**
- Remove IRC-specific code
- Rename `IRSSILLMAgent` → `MatrixLLMAgent`
- Instantiate `MatrixMonitor` instead of `IRCRoomMonitor`
- Update CLI mode for Matrix testing
- Update configuration loading for Matrix

**New Structure:**
```python
class MatrixLLMAgent:
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.model_router = ModelRouter(self.config)
        self.history = ChatHistory(...)
        self.chronicle = Chronicle(...)
        self.matrix_monitor = MatrixMonitor(self)
        self.quests = QuestOperator(self)
    
    async def run(self) -> None:
        await self.history.initialize()
        await self.chronicle.initialize()
        await self.quests.scan_and_trigger_open_quests()
        try:
            await self.matrix_monitor.run()
        finally:
            await self.history.close()
```

### E. Update Configuration Schema

**New `config.json.example`:**
```json
{
  "providers": { ... },
  "router": { ... },
  "tools": { ... },
  "actor": { ... },
  "chronicler": { ... },
  
  "matrix": {
    "homeserver": "https://matrix.org",
    "user_id": "@botname:matrix.org",
    "access_token": "your-matrix-access-token",
    "device_id": "MATRIX_LLMAGENT",
    
    "command": {
      "history_size": 30,
      "rate_limit": 30,
      "rate_period": 900,
      "debounce": 1.5,
      "ignore_users": [],
      "default_mode": "serious",
      "room_modes": {
        "!roomid:matrix.org": "classifier"
      },
      "modes": {
        "sarcastic": { ... },
        "serious": { ... },
        "unsafe": { ... }
      },
      "mode_classifier": { ... }
    },
    
    "proactive": {
      "history_size": 10,
      "interjecting": ["!roomid1:matrix.org"],
      "interjecting_test": ["!test:matrix.org"],
      "interject_threshold": 9,
      "rate_limit": 10,
      "rate_period": 60,
      "debounce_seconds": 15.0,
      "models": { ... },
      "prompts": { ... }
    }
  }
}
```

### F. Update History Management

**Changes to `history.py`:**
- Adapt for Matrix room IDs (already uses generic `server_tag` and `channel_name`)
- Use: `server_tag="matrix"`, `channel_name="!roomid:matrix.org"`
- Add Matrix-specific context formatting if needed

**Minimal changes required** - architecture already supports this!

### G. Create Proactive Debouncer for Matrix

**New `matrix_llmagent/proactive.py`:**
- Port from `rooms/proactive.py`
- Adapt for Matrix room IDs
- Same debouncing logic

**Estimated Time:** 10-15 hours

---

## Phase 5: Add llama.cpp Support

### A. Create llama.cpp Provider

**New File:** `matrix_llmagent/providers/llamacpp.py`

**Implementation:**
```python
class LlamaCppClient(BaseAPIClient):
    """llama.cpp API client using OpenAI-compatible API."""
    
    def __init__(self, config: dict[str, Any]):
        providers = config.get("providers", {})
        cfg = providers.get("llamacpp", {})
        super().__init__(cfg)
        
        # Use AsyncOpenAI client pointed at llama.cpp server
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key="not-needed",  # llama.cpp doesn't require auth
            base_url=self.config.get("base_url", "http://localhost:8080/v1")
        )
    
    async def call_raw(self, context, system_prompt, model, ...):
        # Similar to OpenAIClient but simpler
        # No reasoning models, simpler tool handling
        ...
    
    def _extract_raw_text(self, response: dict) -> str:
        # OpenAI-compatible format
        ...
    
    def has_tool_calls(self, response: dict) -> bool:
        # Basic tool support via llama.cpp grammar
        ...
```

**Key Points:**
- llama.cpp server runs at `http://localhost:8080` by default
- Uses OpenAI-compatible API format
- Limited tool/function calling support (uses grammar-based approach)
- No vision support unless using multimodal model
- Context length limits depend on loaded model

### B. Update ModelRouter

**File:** `matrix_llmagent/providers/__init__.py`

Add to `_ensure_client()`:
```python
elif provider == "llamacpp":
    from .llamacpp import LlamaCppClient
    client = LlamaCppClient(self.config)
```

### C. Update Configuration

Add to `config.json.example`:
```json
"providers": {
  "llamacpp": {
    "base_url": "http://localhost:8080/v1",
    "key": "not-needed",
    "max_tokens": 2048
  }
}
```

### D. Usage Examples

Users can now use:
```json
"modes": {
  "serious": {
    "model": "llamacpp:llama-3.1-8b-instruct"
  }
}
```

**Estimated Time:** 2-4 hours

---

## Phase 6: Update Tests

### Test Files to Update

#### A. Rename Test Imports
All test files need updated imports:
- `tests/providers/test_*.py`
- `tests/chronicler/test_*.py`
- `tests/agentic_actor/test_*.py`
- `tests/test_main.py`
- `tests/test_history.py`

Change:
```python
from irssi_llmagent.xxx import yyy
```
To:
```python
from matrix_llmagent.xxx import yyy
```

#### B. Remove IRC Tests
```bash
rm -rf tests/rooms/irc/
rm tests/rooms/test_proactive.py
```

#### C. Create Matrix Tests

**New files:**
- `tests/matrix/test_client.py` - Matrix client tests
- `tests/matrix/test_monitor.py` - Matrix monitor tests
- `tests/test_proactive_matrix.py` - Proactive interjecting for Matrix

**Mock Matrix Events:**
```python
def create_matrix_message_event(room_id, sender, body):
    return {
        "type": "m.room.message",
        "room_id": room_id,
        "sender": sender,
        "content": {
            "msgtype": "m.text",
            "body": body
        },
        "event_id": "$test123",
        "origin_server_ts": 1234567890
    }
```

#### D. Update Fixtures (`tests/conftest.py`)

Update config fixtures to use Matrix instead of IRC:
```python
@pytest.fixture
def config():
    return {
        "providers": {...},
        "matrix": {
            "homeserver": "https://matrix.org",
            "user_id": "@testbot:matrix.org",
            "access_token": "test_token",
            "command": {...},
            "proactive": {...}
        }
    }
```

**Estimated Time:** 8-12 hours

---

## Phase 7: Documentation Updates

### A. Update README.md

**New Structure:**
```markdown
# matrix-llmagent

Agentic LLM chatbot for Matrix with persistent memory, tool use, and local model support.

## Features
- 🟣 **Matrix Protocol Support** - Native Matrix client
- 🧠 **Local LLMs** - llama.cpp support for local models
- 🤖 **AI Integrations** - Anthropic Claude, OpenAI, OpenRouter, Perplexity
- 🛠️ **Agentic Tools** - Web search, code execution, artifact publishing
- 📚 **Persistent Memory** - Chronicle system with quests
- 🎭 **Multiple Personalities** - Sarcastic, serious, and unsafe modes
- 🎯 **Proactive Interjecting** - Joins conversations intelligently

## Installation

1. Install dependencies: `uv sync --dev`
2. Copy `config.json.example` to `config.json`
3. Configure Matrix credentials and API keys
4. Run: `uv run matrix-llmagent`

## Matrix Setup

### Option 1: Using Access Token
1. Log in to Matrix account
2. Get access token from Element: Settings → Help & About → Access Token
3. Add to config.json:
   ```json
   {
     "matrix": {
       "homeserver": "https://matrix.org",
       "user_id": "@yourbot:matrix.org",
       "access_token": "your_token_here"
     }
   }
   ```

### Option 2: Using Password (Auto-login)
See documentation for password-based authentication.

## Local LLM Setup (llama.cpp)

1. Install llama.cpp server
2. Start server: `./llama-server -m model.gguf --port 8080`
3. Configure in config.json:
   ```json
   {
     "providers": {
       "llamacpp": {
         "base_url": "http://localhost:8080/v1"
       }
     }
   }
   ```
4. Use in modes: `"model": "llamacpp:model-name"`

## Commands

In Matrix rooms:
- `@botname: message` - Automatic mode
- `@botname: !h` - Show help
- `@botname: !s query` - Serious mode
- `@botname: !d message` - Sarcastic mode
- `@botname: !a task` - Agentic mode with tools
- `@botname: !p query` - Perplexity research

## Credits

This project is a fork and evolution of [irssi-llmagent](https://github.com/pasky/irssi-llmagent)
by Petr Baudiš, adapted for Matrix protocol and extended with local LLM support.

Original project: https://github.com/pasky/irssi-llmagent
```

### B. Update AGENTS.md

- Change project name throughout
- Update build/test commands
- Update architecture description
- Remove IRC-specific notes
- Add Matrix-specific development notes

### C. Create MIGRATION.md (For Users of Original)

Document how to migrate from irssi-llmagent to matrix-llmagent (if anyone asks).

### D. Create docs/MATRIX_SETUP.md

Detailed guide for:
- Creating Matrix bot account
- Getting access tokens
- Inviting bot to rooms
- Setting permissions
- Troubleshooting

### E. Create docs/LLAMACPP_SETUP.md

Guide for:
- Installing llama.cpp
- Downloading models
- Starting llama.cpp server
- Configuring in matrix-llmagent
- Model recommendations
- Performance tuning

**Estimated Time:** 3-5 hours

---

## Phase 8: Testing & Polish

### A. Manual Testing Checklist

- [ ] Matrix connection and sync
- [ ] Sending/receiving messages
- [ ] Command parsing (mentions)
- [ ] Mode classification
- [ ] All command modes (sarcastic, serious, unsafe)
- [ ] Proactive interjecting
- [ ] Auto-chronicling
- [ ] Quest tracking
- [ ] All tool executions (search, code, artifacts)
- [ ] llama.cpp provider
- [ ] Rate limiting
- [ ] HTML message formatting
- [ ] Error handling and reconnection

### B. Automated Testing

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run pyright
```

All tests must pass!

### C. Integration Testing

- Test with real Matrix server
- Test with llama.cpp server
- Test with various models
- Test in multiple rooms
- Test proactive interjecting in real conversations

### D. Performance Testing

- Message throughput
- Memory usage with large histories
- Chronicle performance
- llama.cpp latency vs cloud providers

**Estimated Time:** 8-16 hours

---

## Total Estimated Timeline

| Phase | Task | Hours |
|-------|------|-------|
| 1 | Repository Setup | 0.5 |
| 2 | Package Renaming | 2-3 |
| 3 | Remove IRC Code | 1 |
| 4 | Add Matrix Support | 10-15 |
| 5 | Add llama.cpp Support | 2-4 |
| 6 | Update Tests | 8-12 |
| 7 | Documentation | 3-5 |
| 8 | Testing & Polish | 8-16 |
| **TOTAL** | | **34.5-56.5 hours** |

**Realistic Timeline:** 1 week of focused full-time work, or 2-3 weeks part-time

---

## Implementation Order

### Week 1: Foundation
1. ✅ Repository setup (Day 1)
2. ✅ Package renaming (Day 1-2)
3. ✅ Remove IRC code (Day 2)
4. ✅ Add Matrix client basics (Day 2-3)
5. ✅ Add Matrix monitor basics (Day 3-4)
6. ✅ Basic send/receive working (Day 4)

### Week 2: Features
7. ✅ Command handling (Day 5)
8. ✅ Mode classification (Day 5)
9. ✅ All command modes (Day 6)
10. ✅ Proactive interjecting (Day 6-7)
11. ✅ llama.cpp provider (Day 7)

### Week 3: Polish (if needed)
12. ✅ Update all tests (Day 8-9)
13. ✅ Documentation (Day 9-10)
14. ✅ Integration testing (Day 10-11)
15. ✅ Bug fixes and polish (Day 11-12)

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Bot connects to Matrix
- [ ] Receives and sends messages
- [ ] Responds to @mentions
- [ ] All three modes work (sarcastic/serious/unsafe)
- [ ] Basic tool use (web search)
- [ ] Chronicle system works
- [ ] llama.cpp provider works
- [ ] Tests pass

### Full Feature Parity
- [ ] All original features working in Matrix
- [ ] Proactive interjecting
- [ ] Quest system
- [ ] All tools (search, code exec, artifacts)
- [ ] HTML formatting
- [ ] Comprehensive tests
- [ ] Complete documentation

### Stretch Goals
- [ ] Matrix-specific features (reactions, threads, edits)
- [ ] File upload for artifacts
- [ ] Admin commands via Matrix
- [ ] Multi-room support
- [ ] Improved llama.cpp tool support

---

## Key Technical Decisions

### Why matrix-nio?
- **Mature library** - Most popular Python Matrix client
- **Async support** - Fits our architecture
- **Active development** - Well maintained
- **Good documentation** - Easy to learn
- **E2E encryption support** - Future-proof

### Why Remove IRC vs Multi-Platform?
- **Simplicity** - Single protocol is easier to maintain
- **Focus** - Better Matrix integration vs split attention
- **User base** - Matrix is growing, IRC is stable/declining
- **Features** - Matrix has richer features to leverage
- **You can always fork** - If someone wants IRC back, it's all in git history

### Why Keep Chronicle/Quest System?
- **Core value** - Persistent memory is key differentiator
- **Platform agnostic** - Works same in Matrix as IRC
- **Proven design** - Already works well
- **User investment** - Existing users have chronicles

---

## Risk Mitigation

### Risk 1: Matrix API Changes
- **Mitigation:** Use stable matrix-nio library that abstracts protocol
- **Fallback:** Pin matrix-nio version if breaking changes occur

### Risk 2: llama.cpp Compatibility
- **Mitigation:** Use standard OpenAI-compatible API
- **Fallback:** Clearly document which models/versions tested

### Risk 3: Feature Gaps in Matrix
- **Mitigation:** Implement workarounds (e.g., HTML for formatting)
- **Fallback:** Document known limitations

### Risk 4: Performance Issues
- **Mitigation:** Async architecture, rate limiting already in place
- **Fallback:** Add caching, optimize database queries

### Risk 5: Lost Users from IRC
- **Mitigation:** Clear README about fork, point to original project
- **Acceptance:** This is a different project, IRC users keep original

---

## Post-Launch Tasks

### Immediate (Week 4)
- [ ] Monitor for bug reports
- [ ] Gather user feedback
- [ ] Quick fixes for critical issues
- [ ] Performance monitoring

### Short-term (Month 2-3)
- [ ] Add Matrix-specific features (reactions, threads)
- [ ] Improve llama.cpp tool support
- [ ] Add more examples and guides
- [ ] Video tutorial for setup

### Long-term (Month 4+)
- [ ] E2E encryption support
- [ ] Admin dashboard via Matrix
- [ ] Model fine-tuning guides
- [ ] Community contributions
- [ ] Consider additional features based on feedback

---

## Questions to Resolve

Before starting implementation:

1. **GitHub Username:** What's your GitHub username for the new repo?
2. **Matrix Bot Account:** Do you already have a Matrix bot account created?
3. **llama.cpp:** Do you have llama.cpp server running locally to test?
4. **Testing Environment:** Do you have a test Matrix room to use during development?
5. **Initial Model:** Which LLM model will you use for initial testing?
6. **Access Token:** Do you have your Matrix access token ready?

---

## Next Steps

Once this plan is approved:

1. Create PLAN.md ✅ (you are here)
2. Execute Phase 1 (Repository Setup)
3. Create development branch
4. Begin Phase 2 (Package Renaming)
5. Iterate through phases with testing at each step

**Ready to begin?** Start with Phase 1: Repository Setup! 🚀
