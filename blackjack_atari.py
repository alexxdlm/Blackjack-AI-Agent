import gymnasium as gym
import ale_py
import numpy as np
from collections import defaultdict

from ale_py import ALEInterface, roms

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("breakout"))
ale.reset_game()

reward = ale.act(0)  # noop
screen_obs = ale.getScreenRGB()

env = gym.make("ALE/Blackjack-v5", render_mode = "human")

class BlackJackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):


        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # balance exploration with experience through greedy epsilon approach
        
        # with probability epsilon we act randomly

        if np.random.random() < self.epsilon: 
            return self.env.action_space.sample()

        #with probabilty 1-epsilon we act following past experience(s)

        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self, 
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):

        """ This is where the q-values are updated, following the bellman equation"""

        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



# Now we can officially train our agent

learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 10        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
it = 1 
# create the environment and the agent

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackJackAgent(env=env,
learning_rate = learning_rate,
initial_epsilon = start_epsilon,
epsilon_decay = epsilon_decay,
final_epsilon = final_epsilon, 
)

from tqdm import tqdm  # Progress bar

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    # print(f"We are in episode {it}")
    obs, info = env.reset()
    done = False
    

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # input("Press any key to continue ...")
        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()
    it += 1
    print(f"Reward was {reward}")
    print(agent.q_values)

test_agent(agent, env)