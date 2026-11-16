# Deployment

This directory holds the Docker deployment manifests.

Run the local stack from the repository root:

```powershell
docker compose -f src/deploy/docker-compose.yml up --build
```

The Compose file uses the repository root as its build context so the image can
include `src`, `scripts`, `requirements.txt`, `setup.py`, and `index.html`.
Its bind mounts therefore point two levels up to the root-level `data` and
optional `secrets` directories.

`.dockerignore` remains at the repository root because Docker only applies it
from the build-context root. GitHub Actions workflows remain in `.github` as
required by GitHub.
