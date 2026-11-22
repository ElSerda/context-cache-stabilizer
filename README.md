# context-cache-stabilizer

A Hermes Agent plugin that measures what Hermes sends to the model on every request. It changes nothing: it reads the request, hands it on, and writes down sizes.

It answers questions such as:

- How big is the fixed part (tools, system prompt) and which pieces weigh the most?
- How fast does the conversation history grow, and what fills it: user messages, answers, tool results?
- How much of each request is identical to the previous one, so a prefix cache (Anthropic, OpenAI, llama.cpp / LM Studio) can reuse it?
- How many tokens did the provider really count, and how many came from its cache?

## How it works

- **Middleware `llm_execution`:** sees the request exactly as it goes to the provider, after every `llm_request` middleware (webgraft removing its footers, for example), and hands it on unchanged.
- **Hook `post_api_request`:** records the token counts the provider reported for that request.

Each request adds lines to `<hermes home>/plugin-data/context-cache-stabilizer/requests.jsonl`. No message text is stored: only sizes, tool names and the headings of the system prompt. Past 20 MB the file is renamed to `requests.1.jsonl` and a new one starts.

## Install

```bash
hermes plugins install ElSerda/context-cache-stabilizer
hermes plugins enable context-cache-stabilizer
hermes gateway restart
```

## Read the measurements

```bash
hermes context-cache report                    # the latest conversation
hermes context-cache report --session local-1a2b --last 10
hermes context-cache path                      # where the file is
```

The report shows one row per request:

| Column        | Meaning                                                                                   |
|---------------|-------------------------------------------------------------------------------------------|
| est tok       | request characters divided by 4, an estimate                                              |
| prompt tok    | prompt tokens the provider reported                                                       |
| cached        | of those, tokens read from the provider's cache                                           |
| tools ch      | characters of the tool definitions                                                        |
| system ch     | characters of the system prompt                                                           |
| user / assist / tool ch | characters of the messages of each role                                         |
| prev kept     | share of the previous request found unchanged at the start of this one; below 100%, a prefix cache recomputes everything from the first difference |
| new           | share of this request that was not in the previous one                                    |
| ttfb          | time to the first streamed chunk                                                          |

Then the parts of the latest system prompt, the parameters of the latest request, and its tools, largest first.

The parameters show what Hermes really sent besides the prompt. Generation settings keep their value (`reasoning_effort`, `temperature`, `max_tokens`, `stream`, `tool_choice`, thinking switches such as `chat_template_kwargs.enable_thinking`…); every other parameter appears by name only (`extra_headers: "<dict>"`), so no header, key or metadata value is written. `extra_body`, which the OpenAI SDK merges into the request body, is opened one level.

LM Studio reports no cached tokens. A short ttfb there is the sign that the prompt was mostly read from the KV cache.

## Settings

```yaml
plugins:
  entries:
    context-cache-stabilizer:
      settings:
        enabled: true   # false: nothing is written
```

## Limits

- Auxiliary calls (compression summaries, titles) do not go through `llm_execution` and are not measured.
- "prev kept" compares the request as Hermes serializes it, not the provider's tokens. A chat template can still render the same JSON differently (reasoning of earlier turns, for example).
- The system prompt is split at its headings ("# ", "## ", "<block>"). Text without a heading is counted in the part above it: the last section of an auto-loaded skill can absorb the Hermes text that follows.

## License

MIT © ElSerda
