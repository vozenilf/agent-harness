# agent-harness

A small, tested agent harness with tool calling. It runs on a local model (Ollama), provides
it with tools, asks for permission before tools that change something, runs the agent loop,
keeps a persistent memory and connects to MCP servers (including my previous project:
[notion-mcp](https://github.com/vozenilf/notion-mcp)).

Built step by step from a reference tutorial, using AI. Each step is a git tag:
`phase-0` … `phase-5`.

```
you ──► CLI ──► Harness.ask() ─────────────────────────────► answer
                  │  ▲
     conversation │  │ tool results
                  ▼  │
             model (Ollama, OpenAI-compatible API)
                  │ "call notion__search_pages(...)"
                  ▼
             ToolRegistry ── approval? ──► workspace files · memory.md · MCP servers (stdio)
```

## Features
- **Agent loop** with a step limit: model → tool calls → results → model … → final answer
- **Tools defined once** (`Tool` dataclass): JSON schema for the model + the Python function
- **Errors go back to the model**, never crash the harness: unknown tool, broken JSON
  arguments, wrong arguments, exceptions inside tools, failing MCP servers
- **Workspace file tools** confined to one folder (path traversal guard)
- **Long-term memory** in `memory.md`: one-line facts, duplicates skipped, loaded into the
  system prompt at startup
- **MCP client**: servers from `mcp_servers.json` (same format as Claude Desktop), tools
  prefixed by server name, secrets passed as `${ENV_VARS}`, a broken server is skipped
- **Approval**: tools that change something (`read_only=False`, or an MCP tool without
  `readOnlyHint`) run only after an explicit `y`
- **Any OpenAI-compatible model** via environment variables (Ollama by default, or Claude)

## Run it
Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) with a model:
```bash
ollama pull qwen3.5:4b
uv run agent-harness
```
Commands: `/tools` lists tools and how they run, `/quit` exits.

| Variable | Default | Purpose |
|---|---|---|
| `HARNESS_BASE_URL` | `http://localhost:11434/v1` | any OpenAI-compatible endpoint |
| `HARNESS_MODEL` | `qwen3.5:4b` | model name |
| `HARNESS_API_KEY` | `ollama` | API key (Ollama ignores it) |
| `NOTION_TOKEN` | – | passed to the Notion MCP server via `mcp_servers.json` |

Use Claude instead (through its OpenAI-compatible endpoint, no code change):
```powershell
$env:HARNESS_BASE_URL="https://api.anthropic.com/v1/"
$env:HARNESS_API_KEY="sk-ant-..."
$env:HARNESS_MODEL="claude-haiku-5-5"
uv run agent-harness
```
Secrets can also live in a gitignored `.env` file: `uv run --env-file .env agent-harness`.

## Tests
```bash
uv run pytest -v        # 69 tests
uv run ruff check . && uv run ruff format --check .
```
- The model is replaced by a **fake OpenAI-compatible server** (`httpx2.MockTransport`
  under the real `openai` client), so tests check the exact JSON sent and are instant.
- File and memory tests use real temporary folders (`tmp_path`).
- MCP tests start a **real demo MCP server** as a subprocess and talk to it over stdio.
- 43 unit tests (one piece in isolation), 21 component tests (harness + tools with the
  model faked), 5 integration tests (real MCP server process).
- CI (GitHub Actions) runs lint, format check and tests on every push.

## Design decisions
- **Harness is a class with an injected client**: no global state, one harness per
  conversation, testable with a fake server.
- **pydantic for outside data** (config from env and files), **dataclass** for internal
  records (`Tool`), **plain classes** for objects with behaviour.
- **The model proposes, the harness decides**: limits, path checks and approvals live in
  code, not in the prompt.

## Next steps
Streaming output · context trimming for long conversations · validating tool arguments
against their JSON schema · a model registry with `/model` switching · Docker image ·
remote MCP servers over HTTP with OAuth.