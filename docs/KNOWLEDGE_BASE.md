# Knowledge Base Tool Setup

The `knowledge_base` tool allows your bot to search a PostgreSQL database with semantic extractions from wikis, documentation, or any text corpus. This enables domain-specific Q&A that goes beyond general web search.

## Overview

The tool provides three types of search:

1. **Full-text search** on page content (summaries, keywords, titles)
2. **Entity search** with fuzzy matching (people, organizations, projects, etc.)
3. **Relationship queries** from a knowledge graph (who works with whom, project dependencies, etc.)

## Database Schema

The knowledge base expects the following PostgreSQL tables:

### Required Tables

#### `page_extensions`
Stores semantic summaries of pages/documents.

```sql
CREATE TABLE page_extensions (
    id SERIAL PRIMARY KEY,
    page_title VARCHAR(500) NOT NULL,
    url VARCHAR(1000) UNIQUE,
    resume TEXT,                    -- Summary/abstract of the page
    keywords TEXT,                  -- Extracted keywords
    
    -- Full-text search vectors (for fast searching)
    page_title_tsv TSVECTOR,
    resume_tsv TSVECTOR,
    keywords_tsv TSVECTOR
);

-- Create indexes for full-text search
CREATE INDEX idx_page_extensions_title_tsv ON page_extensions USING GIN(page_title_tsv);
CREATE INDEX idx_page_extensions_resume_tsv ON page_extensions USING GIN(resume_tsv);
CREATE INDEX idx_page_extensions_keywords_tsv ON page_extensions USING GIN(keywords_tsv);

-- Trigger to auto-update tsvectors
CREATE OR REPLACE FUNCTION update_page_extensions_tsv() RETURNS TRIGGER AS $$
BEGIN
    NEW.page_title_tsv := to_tsvector('english', COALESCE(NEW.page_title, ''));
    NEW.resume_tsv := to_tsvector('english', COALESCE(NEW.resume, ''));
    NEW.keywords_tsv := to_tsvector('english', COALESCE(NEW.keywords, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER page_extensions_tsv_trigger
    BEFORE INSERT OR UPDATE ON page_extensions
    FOR EACH ROW EXECUTE FUNCTION update_page_extensions_tsv();
```

#### `entities`
Stores extracted named entities.

```sql
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100),       -- person, organization, project, event, location, etc.
    url VARCHAR(1000)               -- Optional link to more info
);

-- Index for fuzzy matching (requires pg_trgm extension)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_entities_name_trgm ON entities USING GIN(entity_name gin_trgm_ops);
CREATE INDEX idx_entities_type ON entities(entity_type);
```

#### `entity_relationships`
Stores relationships between entities (knowledge graph).

```sql
CREATE TABLE entity_relationships (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER REFERENCES entities(id),
    predicate VARCHAR(200),         -- e.g., "works_for", "contributed_to", "is_member_of"
    object_id INTEGER REFERENCES entities(id)
);

CREATE INDEX idx_entity_rel_subject ON entity_relationships(subject_id);
CREATE INDEX idx_entity_rel_object ON entity_relationships(object_id);
```

### Example Data

```sql
-- Insert a page
INSERT INTO page_extensions (page_title, url, resume, keywords) VALUES (
    'QGIS',
    'https://wiki.osgeo.org/wiki/QGIS',
    'QGIS is an Open Source Geographic Information System. Started in May 2002, it runs on most Unix platforms, Windows, and OS X.',
    'gis, open source, mapping, geographic, spatial'
);

-- Insert entities
INSERT INTO entities (entity_name, entity_type, url) VALUES
    ('QGIS', 'project', 'https://qgis.org'),
    ('Anita Graser', 'person', 'https://wiki.osgeo.org/wiki/Anita_Graser'),
    ('OSGeo', 'organization', 'https://osgeo.org');

-- Insert relationships
INSERT INTO entity_relationships (subject_id, predicate, object_id)
SELECT s.id, 'is_member_of', o.id
FROM entities s, entities o
WHERE s.entity_name = 'Anita Graser' AND o.entity_name = 'OSGeo';
```

## Configuration

Add to your `config.json`:

```json
{
  "tools": {
    "knowledge_base": {
      "enabled": true,
      "database_url": "postgresql://user:password@localhost/dbname",
      "name": "My Knowledge Base",
      "description": "Search for information about projects, people, and events.",
      "max_results": 5,
      "max_entities": 10
    }
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable the knowledge base tool |
| `database_url` | string | required | PostgreSQL connection string |
| `name` | string | "Knowledge Base" | Display name shown in results |
| `description` | string | auto | Tool description shown to the LLM |
| `max_results` | integer | 5 | Maximum pages to return |
| `max_entities` | integer | 10 | Maximum entities to return |

### Connection String Formats

```bash
# TCP connection with password
postgresql://user:password@localhost:5432/dbname

# Unix socket (no password, uses peer auth)
postgresql:///dbname

# With SSL
postgresql://user:password@host/dbname?sslmode=require
```

## Optimizing LLM Tool Selection

By default, LLMs may prefer `web_search` over `knowledge_base`. To make the bot prioritize your knowledge base for domain-specific queries, update the tool description and system prompt:

### Option 1: Update Tool Description

```json
{
  "tools": {
    "knowledge_base": {
      "description": "PRIORITY TOOL for [Your Domain] queries. Search the knowledge base for information about projects, people, organizations, and events. Use this FIRST before web_search when the query mentions [domain keywords]."
    }
  }
}
```

### Option 2: Update System Prompt

```json
{
  "matrix": {
    "command": {
      "modes": {
        "serious": {
          "system_prompt": "You are a helpful assistant. You have access to a [Your Domain] knowledge_base tool - use it FIRST for any [domain] questions before trying web search."
        }
      }
    }
  }
}
```

## Building Your Knowledge Base

### From a MediaWiki

If you have a MediaWiki instance (like OSGeo Wiki), you can:

1. Export pages via the MediaWiki API
2. Use an LLM to generate summaries and extract entities
3. Store results in the schema above

Example pipeline:
```
MediaWiki API -> Page Content -> LLM Summarization -> PostgreSQL
                              -> Entity Extraction  -> entities table
                              -> Relation Extraction -> entity_relationships table
```

### From Documents

For document corpora (PDFs, markdown, etc.):

1. Extract text from documents
2. Chunk into manageable sections
3. Generate summaries with an LLM
4. Extract entities and relationships
5. Store in PostgreSQL

### From Existing Databases

If you have structured data, map it to the schema:

- Product catalogs -> entities (type: "product")
- User directories -> entities (type: "person")
- Documentation -> page_extensions

## Troubleshooting

### Connection Issues

```
Error: Failed to connect to Knowledge Base
```

- Check `database_url` is correct
- Ensure PostgreSQL is running
- Verify network/firewall allows connection
- Check user has SELECT permissions on tables

### No Results

If searches return no results:

1. Verify data exists: `SELECT COUNT(*) FROM page_extensions;`
2. Check tsvector columns are populated: `SELECT page_title, page_title_tsv FROM page_extensions LIMIT 1;`
3. Test search manually: `SELECT * FROM page_extensions WHERE resume_tsv @@ to_tsquery('your_term');`

### Slow Queries

- Ensure GIN indexes exist on tsvector columns
- Ensure pg_trgm extension is installed for entity fuzzy search
- Consider VACUUM ANALYZE on tables after bulk inserts

## Example Queries

The tool executes queries similar to:

```sql
-- Full-text page search
SELECT page_title, url, resume
FROM page_extensions
WHERE resume_tsv @@ websearch_to_tsquery('english', 'QGIS')
   OR keywords_tsv @@ websearch_to_tsquery('english', 'QGIS')
ORDER BY ts_rank(resume_tsv, websearch_to_tsquery('english', 'QGIS')) DESC
LIMIT 5;

-- Entity fuzzy search
SELECT entity_name, entity_type, url
FROM entities
WHERE entity_name % 'Anita'  -- trigram similarity
   OR entity_name ILIKE '%Anita%'
ORDER BY similarity(entity_name, 'Anita') DESC
LIMIT 10;

-- Relationship lookup
SELECT s.entity_name AS subject, r.predicate, o.entity_name AS object
FROM entity_relationships r
JOIN entities s ON r.subject_id = s.id
JOIN entities o ON r.object_id = o.id
WHERE s.entity_name = 'Anita Graser' OR o.entity_name = 'Anita Graser';
```

## Security Considerations

- Use a **read-only database user** for the bot
- Never expose database credentials in logs
- Consider network isolation (Unix socket or private network)
- The tool only executes SELECT queries, never modifies data
