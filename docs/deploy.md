# Deployment

## Local Compose

```bash
docker compose -f src/deploy/docker-compose.yml up --build
```

The deployment definition runs the API, ingestion worker, RabbitMQ, Redis, and supporting services. Provide secrets through environment variables or mounted secret files; never bake `.env` into an image.

## Production checklist

- Terminate TLS at a reverse proxy and do not expose an unencrypted API with session cookies.
- Set `RAG_SESSION_COOKIE_SECURE=true` under HTTPS.
- Use durable volumes for data, Chroma, SQLite, and message-broker state.
- Configure `RAG_API_KEY`, host validation for MCP, and a restrictive CORS policy.
- Run the API and worker separately; monitor queue depth and worker health.
- Scrape Prometheus and export OTLP telemetry before relying on the service operationally.

See [production readiness](production_ready.md) for acceptance checks and [security](security.md) for hardening.
