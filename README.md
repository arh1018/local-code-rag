# Building a local, hybrid code-search RAG for a large multi-repo codebase

I wanted better code search across a handful of repos I work in daily — a large Python/Django
monolith, a mid-sized Go service, and a smaller Python service — and to serve it straight
into [Claude Code](https://claude.com/claude-code) as an MCP tool. Grep is exact but blind
to intent; a generic "paste your repo into an LLM" tool was a non-starter because the code
can't leave the machine. So I built a small, local-only RAG system from scratch. Here's
what it does, the technology choices, and the bugs that actually mattered.

## Constraints that shaped everything

1. **No code leaves the machine.** This ruled out any hosted embedding API (OpenAI,
   Voyage, Cohere) and any hosted vector DB. Everything — chunking, embedding, storage,
   retrieval — had to run on a laptop CPU.
2. **It has to stay current with a fast-moving codebase.** A one-time index is useless
   after a week of commits.
3. **It had to be good at code, not prose.** Generic text chunkers (fixed-size windows)
   produce chunks that split a function body from its signature, or merge two unrelated
   functions — bad for both embedding quality and for showing a human a coherent result.

## Architecture

```
walk (file discovery + metadata tagging)
  → chunk (tree-sitter)
  → embed (fastembed, local CPU)
  → write (LanceDB: vectors + BM25/FTS in one embedded DB)

edges → SQLite: ORM model → DB table → raw-SQL/ORM refs in another service,
                 background-task name → the call sites that enqueue it

search: hybrid search (vector + BM25) via Reciprocal Rank Fusion
mcp server: exposes search_code / get_symbol / related / cross_repo over MCP
```

### Chunking: tree-sitter, not fixed windows

Fixed-size text chunking is the default in most RAG tutorials, and it's a bad fit for
code: a 150-line sliding window will happily cut a function in half. I used
[tree-sitter](https://tree-sitter.github.io/tree-sitter/) (via `tree-sitter-language-pack`)
to parse real syntax trees for the two languages involved, and chunk along semantic
boundaries: functions and methods (with their decorators), class headers (fields +
metadata, with real line numbers, body stripped), module-level code, and for the Go side,
types/const/var blocks with their doc comments. Overly long definitions get split into
overlapping, numbered parts rather than silently truncated — search results say "part 2/3
of definition at lines X–Y" so a model knows to go read the rest.

Each chunk also gets a **semantic kind** inferred from context (`model`, `view`, `task`,
`serializer`, `admin`, ...) — e.g. a file conventionally named for background tasks whose
function carries a task decorator is tagged `task`, a class inheriting from something
ORM-model-shaped is tagged `model`. This is what lets a query specify
`semantic_kind="task"` and skip everything else.

**Alternative considered:** LangChain/LlamaIndex's `RecursiveCharacterTextSplitter` with
language-aware separators. Faster to set up, but doesn't give you real AST boundaries or
decorator/signature association — it approximates them with regex-ish separators. Given
I only needed two languages, a tree-sitter grammar per language was a bounded, one-time
cost that paid for itself in result quality.

### Embeddings: fastembed + a code-tuned model, running locally

[`fastembed`](https://github.com/qdrant/fastembed) (ONNX runtime under the hood) running
a code-tuned embedding model (768-dim) — chosen specifically because it runs entirely
offline on CPU once the weights are cached. No API key, no network call per chunk, no
code ever transmitted anywhere.

**Alternatives considered:**
- Hosted embedding APIs — ruled out immediately, violates the "code never leaves the
  machine" constraint.
- `sentence-transformers` directly — fastembed wraps the same ONNX models with a lighter
  dependency footprint and better default batching; no strong reason to go lower-level.
- A general-purpose embedding model instead of a code-tuned one — code embeddings
  noticeably outperform general text embeddings on code-shaped queries (natural-language
  intent vs. matching variable/function names), which is the whole point of a
  code-aware model.

### Vector + keyword store: LanceDB

[LanceDB](https://lancedb.github.io/lancedb/) is an embedded, serverless vector database
(like SQLite, but for vectors) with a built-in Tantivy-backed full-text index. That let me
avoid running Postgres+pgvector or a separate Elasticsearch/Qdrant process just to get
BM25 — one file-backed table does both vector ANN search and keyword search.

**Alternative considered:** Qdrant or Chroma for vectors + a separate BM25 library
(`rank_bm25`) for keyword search, combined manually. Works, but it's two moving parts
(and in Qdrant's case, a server process) instead of one embedded file. LanceDB's
trade-off is a less mature ecosystem than Qdrant/pgvector, which was an acceptable cost
for a single-user, single-machine index.

### Hybrid search: Reciprocal Rank Fusion

Pure vector search is bad for exact identifiers (a variable or function name that appears
verbatim in the query gets washed out in semantic similarity). Pure keyword search is bad
for intent (a question phrased in plain language has no exact string match). So every
query runs **both** and merges results with
[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
(`score = Σ 1/(k + rank)`, `k=60` — a standard, parameter-light way to combine ranked
lists without needing to calibrate the two systems' raw scores against each other).

I validated this wasn't just a theoretical improvement: a small eval harness
(recall@5/@10 + MRR over a hand-written question set) showed hybrid beating both
keyword-only and vector-only search on every metric (recall@5 0.80, recall@10 1.00, MRR
0.59). Two of the eval questions turned out to be *wrong* on first pass (a path that
didn't exist, and a genuine naming collision between two similarly-named things in the
codebase) — worth mentioning because it's a reminder that your eval set needs as much
scrutiny as your retrieval code; a bad eval question silently caps your measured recall
below what the system can actually do.

### Cross-repo edges: not everything should be embedding-based

Some questions aren't "find text similar to this" — they're "what connects to what."
Example: one service defines an ORM model; another service (different language) reads
the same table via raw SQL using only the table's string name, with no shared schema or
import to follow. Embeddings are the wrong tool here — there's no semantic similarity
between a class definition and a raw `SELECT * FROM table_name` string. Instead, a
dedicated edge-extraction step does targeted parsing:

- ORM model → DB table name (default naming convention, or explicit overrides)
- background-task decorator → task name string
- the other service: regex extraction of "enqueue this named task" calls and
  `FROM/UPDATE/INTO/JOIN <table>` references

All of this lands in a small SQLite table, and a `related()` MCP tool joins on the shared
string (table name or task name) to answer "what in service B touches this model /
triggers this task" — which an embedding-based tool structurally can't do reliably,
because a bare string literal doesn't look anything like a natural-language query.

### Serving: MCP, not a chat wrapper

The whole point was to make this available *inside* Claude Code without a separate UI, so
it's exposed as an [MCP](https://modelcontextprotocol.io/) server (built on FastMCP) with
four tools: `search_code` (hybrid search, filterable by repo/app/semantic_kind),
`get_symbol` (fetch a class/function/type by name, with method line ranges for classes),
`related`/`cross_repo` (the edge-graph lookup above), registered once via
`claude mcp add --scope user ...`.

## Keeping the index current: git hooks, not CI

This was worth thinking through explicitly rather than defaulting to "add a CI job."
Hosted CI runners are *also* infrastructure outside this machine — pushing a commit would
have the runner check out the code and, to update a shared index, send embeddings
somewhere reachable by other developers. That's the exact thing the project was built to
avoid. So instead:

- An incremental-update step diffs `git` between the last indexed commit and `HEAD`, and
  only re-chunks/re-embeds files that actually changed — seconds, not the couple of hours
  a full rebuild takes on this corpus.
- `post-commit` / `post-merge` / `post-checkout` git hooks fire that update in the
  background after every commit, pull, or branch switch, serialized through a lock file
  so a burst of hooks (e.g. during a rebase) runs one after another instead of racing.
- The MCP server picks up index changes without a restart (it re-checks the table's mtime
  every couple of seconds).

A self-hosted CI runner on internal infrastructure would resolve the "no external
exposure" concern and would be the right move *if* this index were ever shared
automatically across a team — but for a single-developer tool, git hooks are strictly
simpler and have zero extra infrastructure to maintain.

## Bugs that actually mattered

A few things that looked fine until they didn't:

- **Glob matching with `fnmatch`** doesn't understand path semantics for `**` — it let
  `.venv`, `.pytest_cache`, etc. leak into the corpus (tens of thousands of junk files
  indexed before I noticed). Fixed with a proper glob→regex translator that understands
  `**/` as "zero or more path segments."
- **A 726KB single "chunk"** from one test fixture file with a minified-JSON mock on a
  single very long line — chunking was line-count-based, and this file had almost no
  lines. Fixed by capping both per-line and per-chunk character counts independently of
  line count, and routing *every* chunk-producing code path (including class headers and
  one language's type blocks, which had silently bypassed the splitter) through the same
  limit.
- **Embedding cost went superlinear with text length** past ~1500 characters (an
  8000-char batch item cost 6.6s, not the ~26x-of-300-chars you'd expect from linear
  scaling) — a symptom of mixed-length batches forcing padding to the longest sequence.
  Fixed by sorting chunks by length before batching, and capping the text actually sent
  to the embedder at 1200 chars (the full code is still stored and returned — only the
  embedding input is truncated).
- **A dependency's breaking API rename** (a major version bump renamed the core server
  class) silently broke the server import. Pinned to the last compatible major version.
- **`uv run --project X` doesn't change the subprocess's working directory** — so when
  Claude Code launched the MCP server from its own cwd, the module import failed with
  "module not found." `--directory` does what `--project` sounds like it should.


## What I'd change if doing it over

- Index schema migrations as a one-line summary rather than excluding them entirely —
  right now they're skipped wholesale, which is a real gap for any "when was this column
  added" question.
- The FTS (keyword) index is fully rebuilt on every incremental update rather than
  updated in place. Fine at this corpus size (low seconds), but worth revisiting if the
  corpus grows enough that it becomes the slow part of a post-commit hook.
- If this ever needs to be shared *live* across a team rather than handed out as a
  one-time bundle, the git-hooks approach stops being enough — that's the point where a
  self-hosted CI runner plus some way to publish/sync the resulting index becomes worth
  the added infrastructure.

**Stack recap:** tree-sitter (chunking) · fastembed + a code-tuned embedding model (local
CPU embeddings) · LanceDB (vectors + BM25 in one embedded store) · SQLite (cross-repo
edges) · Reciprocal Rank Fusion (hybrid ranking) · FastMCP (serving) · git hooks
(incremental updates) · uv (dependency management).
