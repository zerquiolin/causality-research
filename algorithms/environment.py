class Environment:
    def __init__(self, scm):
        # Initialize the state space, action space, and transition function from scm
        self.states = scm["states"]
        self.actions = scm["actions"]
        self.transition_function = scm["transition"]

    def transition(self, current_state, action):
        # Apply the transition function to get the next state
        return self.transition_function.get((current_state, action), current_state)


def run(scm, s0, phi, p, pi):
    # Step 1: Initialize the environment
    env = Environment(scm)

    # Step 2: Start with the initial state
    state = s0
    run_history = []

    # Step 3: Run the simulation until phi says the match is over
    while not phi(state, run_history):
        # Ask the player for their action
        action = pi(state, run_history)

        # Record the current state and action in the run history
        run_history.append((state, action))

        # Transition to the next state
        state = env.transition(state, action)

    # Step 4: Evaluate the run using the scoring function
    score = p(run_history)

    # Return the run history and the score
    return run_history, score


# Example usage:

# Define an SCM
scm = {
    "states": ["start", "mid", "end"],
    "actions": ["go", "stop"],
    "transition": {
        ("start", "go"): "mid",
        ("mid", "go"): "end",
        ("mid", "stop"): "start",
    },
}


# Define termination condition
def phi(state, run_history):
    return state == "end"


# Define scoring function
def p(run_history):
    return len(run_history)  # Example: Score is the number of steps taken


# Define player logic
def pi(state, run_history):
    if state == "start":
        return "go"
    elif state == "mid":
        return "go"
    else:
        return "stop"


# Run the simulation
run_history, score = run(scm, "start", phi, p, pi)

# Output the results
print("Run History:", run_history)
print("Score:", score)
