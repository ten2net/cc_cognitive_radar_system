# Cognitive Radar System

This project implements a cognitive radar system using reinforcement learning with UV package management.

## Features

- Advanced radar simulation with RadarsimPy
- Reinforcement learning training with RSL-RL
- Gymnasium-based training environment
- UV for dependency management

## Installation

1. Install UV:
   ```bash
   pip install uv
   ```

2. Create virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -e .
   ```

## Usage

Train the radar agent:
```bash
uv run train-radar --config config.yaml
```

Evaluate trained model:
```bash
uv run eval-radar --model trained_model.pt
```

## Project Structure

    .
    ├── src
    │   └── cognitive_radar      # Core package
    ├── tests                    # Unit tests
    ├── docs                     # Documentation
    ├── notebooks                # Jupyter notebooks
    ├── scripts                  # Utility scripts
    ├── pyproject.toml           # UV configuration
    └── README.md                # Project documentation
