```
src/
│
├── benchmark/             # 💡 High-level orchestration & benchmark runner
│   ├── runner.py          # Orchestrates full benchmark lifecycle
│   ├── config/            # Config loading, validation (YAML/JSON schemas)
│   │   └── schema.py
│   └── registry.py        # For dynamic component loading
│
├── environments/          # 🌍 Environments (base + implementations)
│   ├── base.py
│   ├── metrics.py         # Tracking, evaluations, scoring
│   └── impl/              # Concrete envs
│       └── my_env.py
│
├── scm/                   # ⚙️ Structural Causal Models
│   ├── base.py            # Abstract SCM logic
│   ├── dag.py             # DAG structure + utils
│   ├── nodes.py           # SCM nodes (atomic units)
│   └── impl/              # Different SCM implementations
│       └── my_scm.py
│
├── agents/                # 🧠 Agents that interact with the env
│   ├── base.py
│   └── impl/              # Concrete agent types
│       └── greedy_agent.py
│
├── generators/            # 🏗️ DAG/SCM/Env generators (can be heavy)
│   ├── base.py            # Optional base class if needed
│   ├── dag_generator.py
│   ├── scm_generator.py
│   └── env_generator.py
│
├── evaluation/            # 📊 Metrics, analysis, plots
│   ├── evaluator.py       # Evaluation logic
│   └── visualizer.py      # Optional: plots/stats (matplotlib, seaborn)
│
├── lib/                   # 🛠️ Shared utilities
│   ├── constants.py
│   ├── utils/             # General-purpose helper functions
│   │   └── file_utils.py
│   └── models/            # Abstract models (shared interfaces)
│       └── abstract/
│           ├── agent.py
│           ├── scm.py
│           └── environment.py
│
tests/                     # ✅ Unit and integration tests
│   ├── environments/
│   ├── scm/
│   ├── agents/
│   └── ...
```

# 🧠 Causality Game: Benchmarking Causal Inference Agents

Causality Game is a benchmarking framework designed to evaluate and compare the performance, strategies, and capabilities of causal discovery agents. It provides a standardized simulation environment where agents interact with structural causal models (SCMs), gather data, and attempt to learn underlying causal graphs.

The game enables:
 • Controlled and repeatable experiments
 • Fair agent comparison
 • Customizable game instances, agents, and evaluation metrics

# 🚀 Getting Started

## 📦 Installation

Install in editable mode (recommended for development):

git clone <your-repo-url>
cd causalitygame
pip install -e .

## 🧩 Main Components

To test an agent using the Causality Game, several components must be set up:

### 1. DAG Generator

📍 causalitygame.generators.dag_generator

Use this module to generate directed acyclic graphs (DAGs) that define the structure of your SCM.
 • Supports pre-built DAG generation functions (e.g., binary tree, ER graphs).
 • You can implement your own by extending the base DAG class from causalitygame.scm.base.

### 2. SCM Generator

📍 causalitygame.generators.scm_generator

Generates a Structural Causal Model (SCM) over a given DAG:
 • Handles both numerical and categorical variables.
 • Creates symbolic equations and sampling noise.
 • Includes support for defining CDFs for categorical variables.

Custom SCM implementations should extend:
 • SCM class (for the full model)
 • SCMNode class (for individual variables) from causalitygame.scm.base.

### 3. Game Instance Generator

📍 causalitygame.game.GameInstance

Wraps together the DAG, SCM, and random state into a full game-ready configuration.
 • Use .save(filename) to serialize a game instance.
 • Reload with GameInstance.from_dict(...) or via joblib.load(...).

This ensures each agent plays under identical conditions.

### 4. Agents (Players)

📍 causalitygame.agents.base and causalitygame.agents.impl

Agents are the players in the game. They decide:
 • When to stop exploring
 • What interventions or experiments to perform
 • How to construct their estimated causal graph

You can:
 • Create a custom agent by inheriting from BaseAgent in agents.base.
 • Use built-in agents like RandomAgent and ExhaustiveAgent from agents.impl.

### 5. Metrics

📍 causalitygame.evaluators.impl and causalitygame.evaluators.base

Metrics are used to evaluate an agent’s behavior and results.

Two types:
 • Behavior Metrics: e.g. number of experiments, number of samples used.
 • Deliverable Metrics: e.g. SHD, F1-score, edge accuracy.

To define your own, extend the base classes in:
 • causalitygame.evaluators.base.BehaviorMetric
 • causalitygame.evaluators.base.DeliverableMetric

### 6. Game Runner

📍 causalitygame.game.Game

The central class that:
 • Loads a game instance
 • Runs each agent in identical environments
 • Applies evaluation metrics
 • Produces output reports and plots

Constructor parameters:
 • agents: list of (name, agent_instance)
 • game_spec: path to a saved game instance (.pkl)
 • behavior_metrics: list of instantiated behavior metric objects
 • deliverable_metrics: list of instantiated deliverable metric objects
 • seed: optional configuration

## 🧪 Example: Running the Benchmark

### Agents

```python
from causalitygame.agents.impl.RandomAgent import RandomAgent
from causalitygame.agents.impl.ExhaustiveAgent import ExhaustiveAgent
```

### Metrics

```python
from causalitygame.evaluators.impl.behavior import (
    ExperimentsBehaviorMetric, TreatmentsBehaviorMetric, RoundsBehaviorMetric
)
from causalitygame.evaluators.impl.deliverable import (
    SHDDeliverableMetric, F1DeliverableMetric, EdgeAccuracyDeliverableMetric
)
```

### Game

```python
from causalitygame.game.Game import Game
import numpy as np

base_seed = 42
agents = []

for i in range(1, 6):
    rs = np.random.RandomState(base_seed + i)
    agent = (
        f"random {i}",
        RandomAgent(
            stop_probability=rs.beta(a=0.5, b=10),
            experiments_range=(1, max(rs.poisson(10), 2)),
            samples_range=(rs.randint(500, 800), rs.randint(800, 1000)),
            seed=base_seed + i,
        ),
    )
    agents.append(agent)

agents.append(("exhaustive", ExhaustiveAgent()))

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

game = Game(
    agents=agents,
    game_spec="instances/game_instance.pkl",  # path to your saved GameInstance
    behavior_metrics=behavior_metrics,
    deliverable_metrics=deliverable_metrics,
    seed=911,
)


results = game.run()
print(results)
game.plot()
```

### 📊 Output

After running game.run():

- The result is a dictionary with agent names as keys.
- Each entry contains:
- "history": Full action log (experiments, datasets)
- "raw": Raw metric values
- "behavior_score": Weighted behavior score
- "deliverable_score": Weighted deliverable score

🖼️ Visualize Performance

game.plot()

Creates a scatter plot where:
 • X-axis: behavior score
 • Y-axis: deliverable score
 • Each point = one agent

This makes it easy to compare agent strategies and outcomes.

## 📚 Summary

Component Description:

- dag_generator: Generate or customize DAG structures.
- scm_generator: Create symbolic SCM equations over DAGs.
- GameInstance: Bundle DAG + SCM for reproducible game.
- agents.impl: Ready-made agents (Random, Exhaustive).
- agents.base: Interface for custom agent design.
- evaluators.impl: Built-in behavior & deliverable metrics.
- evaluators.base: For building your own metrics.
- Game: Runs the full benchmark and evaluation.

Let me know if you’d like this turned into a README.md or integrated into docstrings across the repo.
