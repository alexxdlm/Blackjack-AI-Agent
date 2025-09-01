from collections import defaultdict
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.envs.toy_text.blackjack import BlackjackEnv


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.999,
    ):

        self.env = env
        
        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []



    def get_action(self, state: tuple) -> int:
        
        # If split is allowed
        if state[3]: 
    
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()

            else:
                return int(np.argmax(self.q_values[state]))
        
       # if surrender and dd are allowed
        elif state[4]: 
            if np.random.random() < self.epsilon:
                return np.random.choice([0,1,3,4])

            else:
                q_values = self.q_values[state].copy()
                q_values[2]= -np.inf
                return int(np.argmax(q_values))        

        else:
            if np.random.random() < self.epsilon:
                return np.random.choice([0,1])
            else:
                q_values = self.q_values[state].copy()
                q_values[2], q_values[3], q_values[4] = -np.inf, -np.inf, -np.inf

            return int(np.argmax(q_values))



    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple,
    ):

        future_q_value = (not terminated) * np.max(self.q_values[next_state])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[state][action]


        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    # changed the decay_epsilon function to a non linear approach

    def decay_epsilon(self, t):

        self.epsilon = max(self.final_epsilon, (self.epsilon * self.epsilon_decay)/(self.epsilon_decay+t))



learning_rate = 0.0001 
start_epsilon = 1.0         
epsilon_decay = 3_000_000  
final_epsilon = 0.001      
n_episodes = 5_000_000

# Create environment and agent following gymnasium's instructions

env = gym.make("Blackjack-v1", sab=False) 
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

from tqdm import tqdm  # Progress bar

################################
# This is where the agent learns
################################

def train_agent(n_episodes: int):
    for episode in tqdm(range(n_episodes)):
        # Start a new hand
        state, _ = env.reset()
        done = False
        it = 0

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)

            action = agent.get_action(state)
            it += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, terminated, next_state)
            done = terminated or truncated
            state = next_state

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon(episode)





# the following section is the one for plots, didn't touch this
"""from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    #Compute moving average to smooth noisy data.
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()

"""

##############################################
# This is where we test how the agent performs
##############################################


def test_agent(agent, env, num_episodes):
    total_rewards = []

    # Disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    # the approach is the same as the learning part, except we don't update the q table, we just reference it 
    # to choose the best action
    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        it = 0
        while not done:
            action = agent.get_action(state)
            it+=1
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


# Train agent
train_agent(1_000)
# Test your agent 
test_agent(agent, env, 100)
#print(agent.q_values)

data = []


####################
# this section writes the q values to a csv, for analytical purposes


for (state, q_value) in agent.q_values.items():
    data.append({
        'player_sum': state[0],
        'dealer_card': state[1],
        'usable_ace': state[2],
        'allow_split': state[3],
        'allow_dd': state[4],
        'q0': q_value[0],
        'q1': q_value[1],
        'q2': q_value[2],
        'q3': q_value[3],
        'q4': q_value[4],
})


# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'q_table_pandas.csv'
df.to_csv(csv_file_path, index=False)






















