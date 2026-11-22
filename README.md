# context-cache-stabilizer

Hermes Agent plugin that stabilizes the LLM request prefix between turns to preserve KV/prompt cache with LM Studio (and any OpenAI-compatible backend).

## Problem

When Hermes assembles the prompt for each turn, it can change:
- **Tool definitions order** — Hermes reorders tools dynamically → 37K+ tokens of KV-cache lost
- **Timestamps in messages** — `2026-11-22T10:15` changes every turn → cache invalidated

## Solution

Registers `llm_request` middleware that intercepts provider kwargs before execution and rewrites them to be stable across turns.

### What it does

| Setting | Default | Effect |
|---|---|---|
| `stabilize_tools_order` | `true` | Sorts tool definitions alphabetically (prevents shuffle-induced cache breaks) |
| `freeze_timestamps` | `true` | Replaces changing timestamps with a fixed value (`2026-11-22 00:00`) |
| `cache_breakpoint_tokens` | `""` | Optional Anthropic-style cache breakpoint tokens (space-separated) |

## Installation

```bash
# Install from local directory
hermes plugins install F:/GPU/context-cache-stabilizer --no-enable

# Enable the plugin
hermes plugins enable context-cache-stabilizer
```

Or clone directly:

```bash
cd ~/.hermes/plugins/
git clone F:/GPU/context-cache-stabilizer.git
hermes plugins enable context-cache-stabilizer
```

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  context-cache-stabilizer:
    stabilize_tools_order: true   # Sort tools alphabetically
    freeze_timestamps: true       # Freeze dynamic timestamps
    cache_breakpoint_tokens: ""   # Optional: "<token1> <token2>" for Anthropic breakpoints
```

## How it works

The plugin registers an `llm_request` middleware hook that runs **before** Hermes calls the provider:

```python
def on_llm_request(**kwargs):
    request = dict(kwargs["request"])  # full provider kwargs
    
    # Sort tools alphabetically
    if "tools" in request:
        tools.sort(key=lambda t: t["function"]["name"])
    
    # Freeze timestamps
    messages = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', '2026-11-22 00:00', request["messages"])
    
    return {"request": request}
```

This preserves the KV-cache prefix across turns because the provider sees an identical prompt structure.

## Limitations

- **Cold resume** — After a Hermes restart, the first request is always outside the cache window (LM Studio doesn't persist KV-cache between sessions)
- **Compression** — The plugin stabilizes *after* compression; if compression itself changes the prefix, that's handled separately by Hermes' context engine plugins

## License

MIT © ElSerda
