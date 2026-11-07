"""Static checks for the optional multi-replica Prometheus/Grafana stack."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVABILITY = ROOT / "deploy" / "observability"


def test_observability_services_are_optional_and_version_pinned() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "image: prom/prometheus:v3.16.0" in compose
    assert "image: grafana/grafana:13.3.0" in compose
    assert compose.count('profiles: ["observability"]') == 2
    assert "prom/prometheus:latest" not in compose
    assert "grafana/grafana:latest" not in compose
    api_section = compose[compose.index("  api:") : compose.index("  prometheus:")]
    assert "container_name:" not in api_section
    assert "RAG_API_PORT_RANGE:-8000-8009" in api_section


def test_prometheus_discovers_each_api_replica() -> None:
    config = (OBSERVABILITY / "prometheus.yml").read_text(encoding="utf-8")

    assert "dns_sd_configs:" in config
    assert "names: [api]" in config
    assert "port: 8000" in config
    assert "job_name: rag-api" in config


def test_grafana_dashboard_aggregates_and_splits_replicas() -> None:
    dashboard_path = OBSERVABILITY / "grafana" / "dashboards" / "rag-overview.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    expressions = {
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    }

    assert dashboard["uid"] == "rag-api-overview"
    assert any('sum(up{job="rag-api"})' in expression for expression in expressions)
    assert any("sum by (instance)" in expression for expression in expressions)
    assert any("histogram_quantile(0.95" in expression for expression in expressions)


def test_grafana_provisions_prometheus_and_the_dashboard() -> None:
    datasource = (
        OBSERVABILITY / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    ).read_text(encoding="utf-8")
    provider = (OBSERVABILITY / "grafana" / "provisioning" / "dashboards" / "rag.yml").read_text(
        encoding="utf-8"
    )

    assert "uid: prometheus" in datasource
    assert "url: http://prometheus:9090" in datasource
    assert "path: /var/lib/grafana/dashboards" in provider
