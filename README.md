# AD-VANCE: Agentic AI for Alzheimer's Disease Research

## Overview

AD-VANCE is an autonomous, agentic AI system designed to accelerate Alzheimer's disease research through intelligent data integration, knowledge graph construction, and automated hypothesis generation.

## Features

- **Multi-Agent Architecture**: Six specialized agents working in concert
- **Data Integration**: Harmonizes heterogeneous AD datasets (ADNI, AMP-AD, ROSMAP, UK Biobank)
- **Knowledge Graph**: Dynamic, self-updating Alzheimer-specific knowledge graph
- **Hypothesis Generation**: Novel biomarker combinations and drug repurposing candidates
- **Rigorous Validation**: Cross-validation, survival analysis, causal inference
- **Full Reproducibility**: Lineage tracking, decision logging, transparency reports

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/danymukesha/ad-vance.git
cd ad-vance

# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Setup Ollama for local LLMs
ollama serve &
ollama pull llama3
ollama pull mistral
```

### Basic Usage

```python
from agents.orchestrator.agent import ResearchOrchestrator

# Initialize the system
orchestrator = ResearchOrchestrator()

# Run a research query
result = orchestrator.execute_goal(
    "Identify novel biomarkers for early Alzheimer's detection using multi-omic data"
)

# Get generated hypotheses
for hypothesis in result.hypotheses:
    print(f"Hypothesis: {hypothesis.description}")
    print(f"Confidence: {hypothesis.confidence}")
    print(f"Evidence: {hypothesis.evidence_paths}")
```

### Docker Deployment

```bash
# Build and run with Docker
cd docker
docker-compose up -d

# Access the dashboard
# Open http://localhost:3000
```

## Architecture

```
┌─────────────────────────────────────────┐
│      Research Orchestrator Agent        │
│  (Goal Decomposition, Task Planning)    │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────────┐
│  Data   │ │Knowledge│ │ Hypothesis  │
│Integration│ │  Graph  │ │ Generation  │
└─────────┘ └─────────┘ └─────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Validation &    │
         │ Simulation      │
         └─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Reproducibility │
         │ & Audit         │
         └─────────────────┘
```

## Scientific Focus

**Primary**: Multi-omic biomarker discovery for early AD detection

This focus was chosen because:
- Early detection significantly improves treatment outcomes
- Multi-omic integration remains a critical unmet need
- Public datasets provide robust training data

## Validation

The system benchmarks against:
- Traditional ML (Random Forest, XGBoost, SVM)
- Single-omic models
- Published clinical criteria

Metrics: ROC-AUC, Precision-Recall, C-index, Hazard Ratios

## Documentation

- [Technical Whitepaper](docs/whitepaper.md)
- [Scientific Manuscript](docs/manuscript.md)
- [Executive Summary](docs/executive_summary.md)
- [Deployment Guide](docs/deployment_guide.md)

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@article{ADVANCE2026,
  title={AD-VANCE: An Agentic AI System for Alzheimer's Disease Research},
  author={AD-VANCE Team},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

- Alzheimer's Disease Neuroimaging Initiative (ADNI)
- AMP-AD Knowledge Portal
- ROSMAP Consortium
- UK Biobank
- All open-source contributors
