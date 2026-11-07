# Project instructions

## Keep `.env` synchronized

Whenever `.env.example` is changed, update `.env` in the same change so every added, renamed, or removed setting is reflected locally. Preserve existing local values and secrets; add missing settings with an appropriate local/default value instead of replacing configured values.

## Keep `docs/feature.md` up to date

`docs/feature.md` lists what still has to be fixed for production and for scale. Whenever your work on this repo reveals something that belongs there, add it in the same session, without being asked. This applies to every agent and subagent, whatever the task was.

**Add an item when** you find a real problem in the current code that would:

- break, slow down, or get expensive as users, documents, or servers grow
- make it unsafe or unreliable to run in production

**Don't add** polish, style, diagram or documentation notes, feature ideas, or limitations people can easily live with. Keep the list short. Don't go hunting for items unless the task asks for it: record what you come across while doing your work.

**How to add one:**

1. Read `docs/feature.md` first. If the problem is already listed, update that item with anything new instead of adding a second one.
2. Put it in the pipeline stage where the fix would be made (Stage 1 Ingestion through Stage 6 Infrastructure), under the group that fits. If a later stage needs data stored during ingestion, also add it to **Store these now** in Stage 1.
3. Match the format of the existing items: a bold one-line statement of the problem, why it matters, and the fix if it's known.
4. End with how it was confirmed: *Reproduced*, *Measured* (with the numbers), or *From the code*. Only claim what you actually did.

**Remove an item** in the same change that fixes it. If you fix only part of it, reword it to say what's left.

Only edit `docs/feature.md` for this. Don't fix the problem as well unless your task covers it, and leave committing to the user. When you finish, tell the user what you added or removed.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` when the Graphify CLI is installed; otherwise skip this optional cache update.
