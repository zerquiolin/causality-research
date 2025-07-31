# 🧠 Causality Game: Benchmarking Causal Inference Agents

Causality Game is a benchmarking framework designed to evaluate and compare the performance, strategies, and capabilities of causal inference agents. It provides a standardized simulation environment where agents interact with structural causal models (SCMs) through an environment, gather data, and attempt to solve various causal inference tasks.

The framework enables:

- Controlled and repeatable experiments
- Fair agent comparison
- Customizable game instances, agents, and evaluation metrics
- Multiple mission types for different causal inference tasks

# 🚀 Getting Started

## 📦 Installation

Install in editable mode (recommended for development):

```bash
git clone <your-repo-url>
cd causalitygame
pip install -e .
```

## 🧩 Core Components

To test an agent using the Causality Game, several components must be set up:

### 1. Structural Causal Models (SCMs)

📍 `causalitygame/scm`

SCMs are the foundational components that define the causal relationships between variables. The framework supports various types of SCMs:

- **Equation-based SCMs**: Symbolic equations defining causal relationships
- **Database-driven SCMs**: SCMs constructed from real-world datasets
- **Bayesian Network SCMs**: SCMs based on Bayesian networks
- **Physics-based SCMs**: Domain-specific SCMs modeling physical laws

> **Note**: There is no limitation on the creation of new SCMs.

Key features:

- Support for both numerical and categorical variables
- Customizable noise distributions
- Intervention capabilities for controllable variables
- Domain-specific node implementations

### 2. Missions

📍 `causalitygame/missions`

Missions define the specific causal inference task that agents need to solve:

- **DAG Inference Mission**: Learn the underlying causal structure (DAG)
- **Average Treatment Effect (ATE) Mission**: Estimate the average treatment effect
- **Conditional Average Treatment Effect (CATE) Mission**: Estimate treatment effects conditional on covariates
- **Treatment Effect (TE) Mission**: General treatment effect estimation

> **Note**: There is no limitation on the creation of new missions.

### 3. Game Instances

📍 `causalitygame/game_engine.GameInstance`

Game instances encapsulate all necessary components for a benchmark run:

- A specific SCM
- A mission with defined metrics
- Game parameters (max rounds, random state)
- Serialization capabilities for reproducibility

### 4. Agents

📍 `causalitygame/agents`

Agents are the players that interact with the environment to solve missions:

Built-in agents:

- **RandomAgent**: Makes random decisions about experiments and stopping
- **ExhaustiveAgent**: Systematically explores all possible interventions

Custom agents can be implemented by extending [BaseAgent](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/agents/abstract.py#L7-L48) in [causalitygame/agents/abstract.py](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/agents/abstract.py).

> **Note**: There is no limitation on the creation of new agents.

### 5. Evaluation Metrics

📍 `causalitygame/evaluators`

Metrics evaluate both agent behavior and deliverables:

**Behavior Metrics** (evaluate agent conduct):

- [ExperimentsBehaviorMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/behavior/ExperimentsBehaviorMetric.py#L0-L0): Number of experiments performed
- [RoundsBehaviorMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/behavior/RoundsBehaviorMetric.py#L0-L0): Number of rounds taken
- [TreatmentsBehaviorMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/behavior/TreatmentsBehaviorMetric.py#L0-L0): Number of treatments applied

**Deliverable Metrics** (evaluate solution quality):

- [AbsoluteErrorDeliverableMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/deliverable/AbsoluteErrorDeliverableMetric.py#L0-L0): Absolute error in estimations
- [EdgeAccuracyDeliverableMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/deliverable/EdgeAccuracyDeliverableMetric.py#L0-L0): Accuracy of learned graph edges
- [F1DeliverableMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/deliverable/F1DeliverableMetric.py#L0-L0): F1 score for graph recovery
- [MeanSquaredErrorDeliverableMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/deliverable/MeanSquaredErrorDeliverableMetric.py#L0-L0): MSE of estimations
- [SHDDeliverableMetric](file:///Users/sergioamortegui/Desktop/Business/Research/Causality/Code/causalitygame/evaluators/deliverable/SHDDeliverableMetric.py#L0-L0): Structural Hamming Distance for graph recovery

> **Note**: There is no limitation on the creation of new metrics.

### 6. Game Runner

📍 `causalitygame/game_engine.Game`

The central class that orchestrates the benchmark:

- Loads game instances
- Runs each agent in identical environments
- Applies evaluation metrics
- Produces results and visualizations

## 🧪 Example: Running the Benchmark

### Define Agents

```python
import causalitygame as cg
from causalitygame.agents.random import RandomAgent
from causalitygame.agents.exhaustive import ExhaustiveAgent

# Define the agents
agents = [
    (f"Random Agent {i}", RandomAgent(seed=911 + i, samples_range=(1, 3)))
    for i in range(1, 3)
]
# Add an exhaustive agent
agents.append(("Exhaustive Agent", ExhaustiveAgent(num_obs=1)))
```

### Define Game Instance

```python
# Game Instance
game_instance_path = "causalitygame/data/game_instances/cate/foundations_instance.json"
```

### Define Metrics (Optional)

```python
from causalitygame.evaluators.behavior import (
    ExperimentsBehaviorMetric, TreatmentsBehaviorMetric, RoundsBehaviorMetric
)
from causalitygame.evaluators.deliverable import (
    SHDDeliverableMetric, F1DeliverableMetric, EdgeAccuracyDeliverableMetric
)

behavior_metrics = [
    ExperimentsBehaviorMetric(),
    TreatmentsBehaviorMetric(),
    RoundsBehaviorMetric(),
]

deliverable_metrics = [
    SHDDeliverableMetric(),
    F1DeliverableMetric(),
    EdgeAccuracyDeliverableMetric(),
]
```

### Define Custom Hooks (Optional)

```python
def on_game_start():
    ...


def on_agent_game_start(agent_name):
    ...


def on_round_start(agent_name, round, state, actions, samples):
    ...


def on_action_chosen(agent_name, state, action, action_object):
    ...


def on_action_evaluated(agent_name, state, action, action_object, result):
    ...


def on_round_end(agent_name, round, state, action, action_object, samples, result):
    ...


def on_agent_game_end(agent_name):
    ...


def on_game_end():
    ...
```

### Run Game

```python
# Game
import pandas as pd
import causalitygame as cg

# Plotting
import matplotlib.pyplot as plt

# Agents
from causalitygame.agents.random import RandomAgent
from causalitygame.agents.exhaustive import ExhaustiveAgent


# Define the agents
agents = [
    (f"Random Agent {i}", RandomAgent(seed=911 + i, samples_range=(1, 3)))
    for i in range(1, 3)
]
# Add an exhaustive agent
agents.append(("Exhaustive Agent", ExhaustiveAgent(num_obs=1)))
# Game Instance
game_instance_path = "causalitygame/data/game_instances/cate/foundations_instance.json"


# Data for plotting
data = {}

datasets = {}


# Hooks
def on_game_start():
    ...


def on_agent_game_start(agent_name):
    ...


def on_round_start(agent_name, round, state, actions, samples):
    ...


def on_action_chosen(agent_name, state, action, action_object):
    ...


def on_action_evaluated(agent_name, state, action, action_object, result):
    ...


def on_round_end(agent_name, round, state, action, action_object, samples, result):
    ...


def on_agent_game_end(agent_name):
    ...


def on_game_end():
    ...


# Create a game
game = cg.Game(
    agents=agents,
    game_spec=game_instance_path,
    behavior_metrics=behavior_metrics, # Optional 
    deliverable_metrics=deliverable_metrics, # Optional 
    hooks={ # Optional 
        "on_game_start": on_game_start,
        "on_agent_game_start": on_agent_game_start,
        "on_round_start": on_round_start,
        "on_action_chosen": on_action_chosen,
        "on_action_evaluated": on_action_evaluated,
        "on_round_end": on_round_end,
        "on_agent_game_end": on_agent_game_end,
        "on_game_end": on_game_end,
    },
    seed=911, # Optional seed for reproducibility
)
# Run the game
runs = game.run()

# Print the results
game.plot()
```

### Simple Example

```python
import causalitygame as cg
from causalitygame.agents.random import RandomAgent
from causalitygame.agents.exhaustive import ExhaustiveAgent

# Define the agents
agents = [(f"Random Agent {i}", RandomAgent(seed=911 + i)) for i in range(1, 3)]
# Add an exhaustive agent
agents.append(("Exhaustive Agent", ExhaustiveAgent()))

# Game Instance
game_instance_path = "causalitygame/data/game_instances/dag_inference/ideal_gas_instance.json"

# Create a game
game = cg.Game(agents=agents, game_spec=game_instance_path)

# Run the game
runs = game.run()

# Print the results
game.plot()
```

### 📊 Output

After running `game.run()`:

- The result is a dictionary with agent names as keys.
- Each entry contains:
  - "history": Full action log (experiments, datasets)
  - "raw": Raw metric values
  - "behavior_score": Weighted behavior score
  - "deliverable_score": Weighted deliverable score

### 🖼️ Visualize Performance

```python
game.plot()
```

Creates a scatter plot where:

- X-axis: behavior score
- Y-axis: deliverable score
- Each point = one agent

This makes it easy to compare agent strategies and outcomes.

## 📁 Project Structure

```
causalitygame/
│
├── agents/                 # Agent implementations
│   ├── abstract.py         # Base agent class
│   ├── random.py           # Random agent implementation
│   └── exhaustive.py       # Exhaustive agent implementation
│
├── data/                   # Data resources
│   ├── datasets/           # Real-world datasets
│   ├── game_instances/     # Predefined game instances
│   └── scms/               # Structural causal models
│
├── evaluators/             # Evaluation metrics
│   ├── behavior/           # Behavior metrics
│   ├── deliverable/        # Deliverable metrics
│   ├── abstract.py         # Base metric classes
│   └── Evaluator.py        # Evaluation orchestrator
│
├── game_engine/            # Core game components
│   ├── Game.py             # Main game runner
│   ├── GameInstance.py     # Game instance definitions
│   ├── Environment.py      # Game environment
│   └── __init__.py
│
├── generators/             # Component generators
│   ├── outcome/            # Outcome generators
│   ├── abstract.py         # Base generator classes
│   ├── dag_generator.py    # DAG generators
│   └── scm_generator.py    # SCM generators
│
├── lib/                    # Utilities and helpers
│   ├── constants/          # Constant definitions
│   ├── helpers/            # Helper functions
│   ├── scripts/            # External algorithm implementations
│   └── utils/              # Utility functions
│
├── missions/               # Mission definitions
│   ├── abstract.py         # Base mission class
│   ├── DAGInferenceMission.py
│   ├── AverageTreatmentEffectMission.py
│   ├── ConditionalAverageTreatmentEffectMission.py
│   └── TreatmentEffectMission.py
│
├── scm/                    # Structural Causal Models
│   ├── dags/               # DAG implementations
│   ├── impl/               # SCM implementations
│   ├── nodes/              # Node definitions
│   ├── noises/             # Noise distributions
│   ├── abstract.py         # Base SCM classes
│   └── db.py               # Database-driven SCM
│
├── translators/            # Format translators
│   └── bif_translator.py   # BIF format translator
│
└── __init__.py
```

## 📚 Key Features

1. **Modular Design**: Each component is loosely coupled, allowing easy extension and customization
2. **Reproducibility**: Game instances can be serialized and shared for consistent benchmarking
3. **Multiple Mission Types**: Support for various causal inference tasks beyond just DAG learning
4. **Extensible Architecture**: Easy to add new agents, metrics, SCMs, and missions
5. **Real-world Datasets**: Integration with standard causal inference datasets
6. **Physics-based SCMs**: Domain-specific SCMs modeling physical laws
7. **Comprehensive Evaluation**: Both behavior and deliverable metrics for holistic assessment

## 🛠️ Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

Run tests with:

```bash
pytest
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
