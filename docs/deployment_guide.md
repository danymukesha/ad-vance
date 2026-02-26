# AD-VANCE Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [AD Workbench Deployment](#ad-workbench-deployment)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.11+
- Docker 24.0+
- Docker Compose 2.20+
- Git
- 16GB RAM minimum

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/ad-vance/ad-vance.git
cd ad-vance
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Ollama (for local LLMs)

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull models
ollama pull llama3
ollama pull mistral
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Run Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Docker Deployment

### 1. Build Images

```bash
cd docker
docker-compose build
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Verify Services

```bash
docker-compose ps
```

Expected output:
```
NAME                    STATUS
ad-vance-orchestrator   Up
ad-vance-data           Up  
ad-vance-kg             Up
ad-vance-hypothesis     Up
ad-vance-validation     Up
ad-vance-ollama         Up
ad-vance-neo4j          Up
```

### 4. Access Services

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Knowledge Graph | http://localhost:7474 |
| Dashboard | http://localhost:3000 |

---

## AD Workbench Deployment

### 1. Prepare Docker Image

```bash
docker build -t ad-vance:latest -f docker/Dockerfile .
```

### 2. Push to Registry

```bash
docker tag ad-vance:latest registry.adworkbench.org/ad-vance:latest
docker push registry.adworkbench.org/ad-vance:latest
```

### 3. Configure Workbench

Create `ad-vance.yaml`:

```yaml
image: registry.adworkbench.org/ad-vance:latest
replicas: 3
resources:
  cpu: "8"
  memory: "32Gi"
  gpu: "1"
ports:
  - 8000:8000
  - 7474:7474
  - 7687:7687
environment:
  NEO4J_PASSWORD: ${NEO4J_PASSWORD}
  OLLAMA_HOST: http://ollama:11434
```

### 4. Deploy

```bash
kubectl apply -f ad-vance.yaml
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama API URL | http://localhost:11434 |
| `NEO4J_URI` | Neo4j connection | bolt://localhost:7687 |
| `NEO4J_USER` | Neo4j username | neo4j |
| `NEO4J_PASSWORD` | Neo4j password | (set in config) |
| `LOG_LEVEL` | Logging level | INFO |

### Configuration Files

- `config/defaults.yaml` - Default settings
- `config/logging.yaml` - Logging configuration

---

## Verification

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "healthy"}`

### 2. API Documentation

Open: http://localhost:8000/docs

### 3. Knowledge Graph

Open: http://localhost:7474/browser/

### 4. Run Integration Test

```bash
python -m pytest tests/integration.py -v
```

---

## Troubleshooting

### Ollama Not Responding

```bash
# Check Ollama logs
docker logs ad-vance-ollama

# Restart Ollama
docker restart ad-vance-ollama
```

### Neo4j Connection Issues

```bash
# Check Neo4j logs
docker logs ad-vance-neo4j

# Verify credentials
# Default: neo4j/advancedb
```

### Out of Memory

```bash
# Increase Docker memory
# Docker Desktop > Settings > Resources > Memory > 16GB
```

---

## Next Steps

1. [ ] Connect to real ADNI/AMP-AD APIs
2. [ ] Configure MLflow tracking
3. [ ] Set up monitoring (Prometheus/Grafana)
4. [ ] Configure SSL/TLS
5. [ ] Set up automated backups

---

## Support

- GitHub Issues: https://github.com/ad-vance/ad-vance/issues
- Documentation: https://docs.ad-vance.org
- Email: team@ad-vance.org
