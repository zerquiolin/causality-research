from benchmark.config import load_config
from benchmark.registry import ENV_REGISTRY, AGENT_REGISTRY, SCM_REGISTRY


def run_benchmark(config_path: str):
    config = load_config(config_path)

    # Load components from config
    env_class = ENV_REGISTRY[config["environment"]["type"]]
    agent_class = AGENT_REGISTRY[config["agent"]["type"]]
    scm_class = SCM_REGISTRY[config["scm"]["type"]]

    # Initialize components
    scm = scm_class(**config["scm"].get("params", {}))
    env = env_class(scm=scm, **config["environment"].get("params", {}))
    agent = agent_class(**config["agent"].get("params", {}))

    # Run interaction loop
    print("Running benchmark...")
    env.reset()
    done = False
    while not done:
        observation = env.observe()
        action = agent.act(observation)
        reward, done = env.step(action)
        agent.learn(observation, action, reward)

    # Evaluation/reporting placeholder
    print("Benchmark finished.")
    env.report_metrics()
