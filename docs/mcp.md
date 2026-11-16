# MCP

The optional Model Context Protocol server exposes the knowledge base to compatible AI clients without giving them write access.

## Enable it

Set `RAG_MCP_ENABLED=true`. In a deployed environment, set `RAG_MCP_ALLOWED_HOSTS` to the public hostnames accepted for `/mcp`; do not leave host validation open behind a reverse proxy.

## Tools

The server provides read-only tools for knowledge-base search, grounded question answering, passage lookup, section lookup, document listing, and store information. The live tool schema at `/mcp` is authoritative because it supplies exact parameter names and types.

## Authentication and limits

`RAG_API_KEY`, when configured, protects MCP as it protects the HTTP API. MCP requests use the service's normal retrieval and generation path, so they can incur LLM cost. Treat each client as a trusted integration until per-client keys and spending limits are added.
