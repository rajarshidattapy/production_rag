# Security

## Baseline controls

- Protect the service with `RAG_API_KEY` and HTTPS in production.
- Store provider keys in environment variables or secret files; never commit `.env`.
- Use secure session cookies under TLS and configure restrictive CORS origins.
- Keep `RAG_ALLOW_STORE_RESET=false` outside controlled administrative workflows.
- Restrict MCP hosts with `RAG_MCP_ALLOWED_HOSTS`.

## Content safety

Before indexing, the service screens hidden control text, prompt-injection signals, credentials, and configurable PII. Retrieved sources are presented as quoted data, never executable instructions. These controls reduce risk; they are not a guarantee against novel attacks.

## Operational safety

Avoid sensitive values in logs, Prometheus labels, and trace attributes. Restrict access to `/metrics`, backups, data volumes, and the vector store. Rotate credentials after exposure and test restoration from backups.
