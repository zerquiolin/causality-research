from causalitygame.scm.dag import DAG
from causalitygame.agents.impl.ExhaustiveAgent import ExhaustiveAgent
from causalitygame.environment.impl.binary_environment import gen_binary_environment

agent = ExhaustiveAgent()
env = gen_binary_environment(agent=agent, seed=42)
print("Environment generated successfully.")

state, history = env.run_game()

print("Game state:", state)
print("Game history:", history)

real_dag = env.game_instance.scm.dag
dag = DAG(history.iloc[-1]["action_object"])

real_dag.plot()
dag.plot()
