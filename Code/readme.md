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
