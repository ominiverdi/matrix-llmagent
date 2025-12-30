# OSGeo Library Client Features

This document describes the chat features implemented in the Rust CLI client (v0.2.2) that should be ported to the Matrix bot. The goal is feature parity so users get the same experience whether using the CLI or Matrix.

## Implementation Status

| Feature | CLI | Matrix Bot |
|---------|-----|------------|
| Document listing (`docs`) | Done | Done |
| Document details (`doc <slug>`) | Done | Done |
| Document selection by index (`doc N`) | Done | Done |
| Element listing (`figures`/`tables`/`equations`) | Done | Done |
| Element filtering (`figures all` vs page-scoped) | Done | Done |
| Context-aware navigation (`n`/`p`) | Done | Done |
| Page navigation (`page N`) | Done | Done |
| Show elements (`show N`) | Done | Done |
| Clear state (`clear`) | Done | Done |
| Semantic search (`!l <query>`) | Done | Done |
| Conversational Q&A with citations | Done | Done |
| View sources (`!sources`) | Done | Done |

## Overview

The library client provides two interaction modes:
1. **Direct search** - Fast semantic search without LLM processing
2. **Conversational Q&A** - Natural language questions answered by an LLM with citations

Both modes support browsing documents, viewing pages, and exploring visual elements (figures, tables, equations).

---

## 1. Document Browsing

### List Documents
Users can browse all documents in the library with pagination.

**Use case**: User wants to see what's available before searching.

**Behavior**:
- Show documents 5 per page with title, page count, and keywords
- Display document slug (the short identifier used in other commands)
- Show pagination info ("page 1/3")
- Provide hint about navigation commands

### View Document Details
Users can select a document to see full metadata.

**Use case**: User found an interesting document and wants details before diving in.

**Behavior**:
- Show slug, title, source filename, total pages
- Show element counts (how many figures, tables, equations)
- Show full summary (3-5 sentences describing the document)
- Show all keywords
- Show license if available

### Document Selection
Users can select a document by number (from the list) or by slug.

**Use case**: Quick selection without typing the full slug.

**Behavior**:
- `doc 1` selects first document from current list
- `doc usgs_snyder` selects by slug directly
- After selection, that document becomes the "current document" for subsequent commands

---

## 2. Page Navigation

### View a Page
Users can view any page with its summary, keywords, and image preview.

**Use case**: User wants to see what's on a specific page, perhaps after finding it in search results.

**Behavior**:
- Show page number and total pages ("p.35/198")
- Show page summary (2-3 sentences)
- Show page keywords (5-8 terms)
- Display page image (thumbnail/preview appropriate for the platform)
- Set this as the "current page" for navigation

### Page Navigation Commands
Users can move through pages sequentially.

**Use case**: User is reading through a section of interest.

**Behavior**:
- `next` or `n` goes to next page
- `prev` or `p` goes to previous page
- At first/last page, show appropriate message
- Navigation updates the current page context

### Jump to Page
Users can jump directly to any page number.

**Use case**: User knows the page number from a citation or table of contents.

**Behavior**:
- `page 45` jumps to page 45 of current document
- `page usgs_snyder 45` jumps to page 45 of specified document (without changing current document permanently)

### Context-Aware Navigation
The `next`/`prev` commands are context-aware.

**Use case**: Seamless navigation whether browsing documents or pages.

**Behavior**:
- If user just viewed a page, `next`/`prev` navigate pages
- If user just listed documents, `next`/`prev` navigate the document list pagination
- Clear feedback about what's being navigated

---

## 3. Element Browsing

### List Elements by Type
Users can list figures, tables, or equations from the current context.

**Use case**: User wants to see all visual elements without searching.

**Commands**:
- `figures` - list figures
- `tables` - list tables  
- `equations` - list equations

**Context-Aware Behavior**:
- If user just viewed a page: show only elements on that page
- If user selected a document but no page: show first 20 elements from document
- If no context: prompt user to select a document first

### List All Elements
Users can explicitly request all elements from a document.

**Use case**: User wants the complete list, not just current page.

**Behavior**:
- `figures all` shows all figures (up to 50) from current document
- `tables all` shows all tables from current document
- `equations all` shows all equations from current document

### Element Display
Each element in the list shows:
- Number (for selection)
- Label (e.g., "Figure 24", "Table 3.1")
- Page number where it appears
- Brief description or caption (truncated if long)

---

## 4. Viewing Elements and Pages

### Show Element in Chat
Users can display element images directly in the conversation.

**Use case**: Quick preview without leaving the chat.

**Behavior**:
- `show 1` displays image for element #1 from last list
- `show 1,3,5` displays multiple elements
- Image rendered appropriately for platform (inline for Matrix, chafa for terminal)

### Open Element Externally
Users can open elements in a full viewer.

**Use case**: User needs higher resolution or wants to save the image.

**Behavior**:
- `open 1` opens element in system viewer
- For Matrix: could provide direct image link or full-resolution attachment

### Show/Open Page
Same commands work for pages.

**Use case**: View full page image, not just the summary.

**Behavior**:
- `show page 45` displays page 45 image in chat
- `open page 45` opens in external viewer

---

## 5. Search

### Semantic Search
Fast search without LLM processing, returns ranked results.

**Use case**: User knows what they're looking for and wants quick results.

**Behavior**:
- `search <query>` performs semantic search
- Returns mixed results: text chunks and visual elements
- Each result shows:
  - Type indicator (CHUNK #N, FIGURE, TABLE, EQUATION)
  - Document slug (for easy reference)
  - Page number
  - Relevance snippet or description
- Results are numbered for selection with `show`

### Search Within Document
Users can scope search to a specific document.

**Use case**: User is exploring one document and wants to find something in it.

**Behavior**:
- Search uses current document context if one is selected
- Or explicit: search within a document by including slug in query context

---

## 6. Conversational Q&A

### Ask Questions
Users can ask natural language questions and get LLM-generated answers with citations.

**Use case**: User wants explanation, not just search results.

**Behavior**:
- Any text that isn't a command is treated as a question
- LLM generates concise answer (2-4 paragraphs)
- Citations appear as `[1]`, `[2]`, `[3]` in the text
- Sources are saved for the `sources` command

### Response Format
Answers should be:
- Concise and direct (2-4 paragraphs typical)
- Cite sources with simple numbered tags
- Focus on answering the question, not restating it
- Include specific details from sources (page numbers, figure references)

### View Sources
Users can see the sources used in the last answer.

**Use case**: User wants to verify information or explore further.

**Behavior**:
- `sources` lists all sources from last LLM answer
- Each source shows:
  - Citation number `[1]`, `[2]`, etc.
  - Type (CHUNK #N for text, or element type)
  - Document slug
  - Page number
  - Brief excerpt or description
- Hint at bottom: "Use 'page <slug> <N>' for full page"

---

## 7. State Management

The client maintains session state for smooth navigation:

### Current Document
- Set when user runs `doc <slug>` or `doc <N>`
- Used as default for `page N`, `figures`, etc.
- Persists until explicitly changed

### Current Page View
- Set when user views a page (`page N`)
- Stores: document slug, page number, total pages
- Used for `next`/`prev` navigation
- Used to scope `figures`/`tables`/`equations` to current page

### Last Sources
- Stored after each LLM answer
- Used by `sources` command
- Cleared on new question

### Document List Pagination
- Current page of document list
- Used for `next`/`prev` when browsing documents (not pages)

---

## 8. Piped Input Support (Testing)

The client supports receiving commands via piped input for testing scenarios.

**Use case**: Automated testing, demos, regression testing.

**Behavior**:
- Detect when stdin is not a terminal (piped input)
- Echo each command as "You: <command>" before output
- Makes test output readable and verifiable

**Example test scenario**:
```
docs
doc usgs_snyder
page 35
What projection methods are described?
sources
quit
```

This allows testing complete user workflows reproducibly.

---

## 9. Help System

### Organized Help
Help is organized into logical groups for discoverability.

**Groups**:
1. **Browse** - docs, doc, page, next/prev
2. **Elements** - figures, tables, equations
3. **View** - show, open, show page, open page
4. **Search** - search, sources, <question>
5. **Other** - help, quit

### Contextual Hints
Commands provide hints about next actions:
- After `docs`: "'doc N' or 'doc <slug>' for details | 'n'=next, 'p'=prev"
- After `doc`: "Use 'figures', 'tables', or 'equations' to browse"
- After `sources`: "Use 'page <slug> <N>' for full page"

---

## Implementation Priority

All core features are now implemented in both CLI and Matrix bot modes.

### Completed
1. Conversational Q&A with citations
2. View sources from last answer
3. Page viewing with summary and image
4. Page navigation (next/prev/jump)
5. Search command
6. Document listing and selection
7. Element listing (figures/tables/equations)
8. Show element images
9. Context-aware element filtering (current page vs all)
10. Document list pagination
11. Piped input for testing (via `--test-conversation`)

### Not Yet Implemented
- Open in external viewer (low priority for Matrix)

---

## API Endpoints Used

The client communicates with the REST API:

- `GET /health` - Server status
- `GET /documents` - List documents (paginated)
- `GET /documents/{slug}` - Document details
- `GET /documents/{slug}/pages/{n}` - Page with image
- `GET /documents/{slug}/elements` - List elements
- `POST /search` - Semantic search
- `POST /chat` - LLM Q&A with sources

All responses include structured data that should be formatted appropriately for Matrix display.
