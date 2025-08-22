from collections import defaultdict
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.envs.toy_text import blackjack


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


# changed the get_action logic, since it was only apt to split and stick operations

    def get_action(self, obs, allow_split) -> int:
        if allow_split:
    
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
            else:
                return int(np.argmax(self.q_values[obs]))
        else:
            if np.random.random() < self.epsilon:
                return np.random.choice([0,1])
            else:
                q_values = self.q_values[obs].copy()
                q_values[2], q_values[3], q_values[4] = -np.inf, -np.inf, -np.inf

            return int(np.argmax(q_values))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
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

    # changed the decay_epsilon function to a non linear approach
    def decay_epsilon(self, t):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, (self.epsilon * self.epsilon_decay)/ (self.epsilon_decay+t))



learning_rate = 0.001   # How fast to learn (higher = faster but less stable)
n_episodes = 2_000_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = 2_000_000  # Reduce exploration over time
final_epsilon = 0.05         # Always keep some exploration

# Create environment and agent
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

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False
    it = 0

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        allow_split = obs[0][0] == obs[0][1] and it == 0
        action = agent.get_action(obs[1:4], allow_split=allow_split)
        next_obs, reward, terminated, truncated, info = env.step(action)
        """  print(next_obs)
        print(next_obs[1:4])"""
        it += 1

        # stick, hit, dd and surrender are handled by this section

        if action != 2: 
            # Take action and observe result

            # Learn from this experience
            #obs[0] = sorted(obs[0])
            agent.update(obs[1:4], action, reward, terminated, next_obs[1:4])

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # this is the part that handles the more complex split operation 
           
        else:
            obs1, obs2 = next_obs
            done1, done2 = False, False
            reward1, reward2 = 0,0

            # deal with the first hand
            while not done1:
                action1 = agent.get_action(obs1[1:4], allow_split=False)
                next_obs1, hand1, terminated1, _, info = env.step(action1)
             #   obs1[0] = sorted(obs1[0])
                agent.update(obs1[1:4], action1, reward1, terminated1, next_obs1[1:4])
                obs1 = next_obs1
                reward1+=hand1
                done1 = terminated1

            #then deal with the second hand
            while not done2:
                action2 = agent.get_action(obs2[1:4], allow_split=False)
                next_obs2, hand2, terminated2, _, info = env.step(action2)
              #  obs2[0] = sorted(obs2[0])
                agent.update(obs2[1:4], action2, reward2, terminated2, next_obs2[1:4])
                obs2 = next_obs2
                reward2+=hand2
                done2 = terminated2
            
            total_split_reward = reward1 + reward2
            #obs[0] = sorted(obs[0])
            agent.update(obs[1:4], action, total_split_reward, terminated=True, next_obs=obs[1:4])
            done=True

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon(episode)


# the following section is the one for plots, didn't touch this
from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
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
plt.show()


##############################################
# This is where we test how the agent performs
##############################################


def test_agent(agent, env, num_episodes=200000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation
    wins, losses, push = 0,0,0

    # the approach is the same as the learning part, except we don't update the q table, we just reference it 
    # to choose the best action
    for _ in tqdm(range(num_episodes)):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        it = 0
        while not done:
            allow_split = obs[0][0] == obs[0][1] and it == 0
            action = agent.get_action(obs[1:4], allow_split)
            it+=1
            next_obs, reward, terminated, truncated, info = env.step(action)
            if action != 2:
                episode_reward += reward
                done = terminated or truncated
                obs = next_obs
            else:
                obs1, obs2 = next_obs
                obs1, obs2 = next_obs

                done1, done2 = False, False
                reward1, reward2 = 0,0

                #FIRST HAND
                while not done1:
                    action1 = agent.get_action(obs1[1:4], allow_split=False)

                    next_obs1, hand1, terminated1, _, info = env.step(action1)
                    obs1 = next_obs1
                    reward1+=hand1
                    done1 = terminated1

                #SECOND HAND
                while not done2:
                    action2 = agent.get_action(obs2[1:4], allow_split=False)
                    
                    next_obs2, hand2, terminated2, _, info = env.step(action2)
                    obs2 = next_obs2
                    reward2+=hand2
                    done2 = terminated2
            
                episode_reward = reward1+reward2
                done=True


        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Test your agent 
test_agent(agent, env)
#print(agent.q_values)

data = []


####################
# this section writes the q values to a csv, for analytical purposes



for (state, q_value) in agent.q_values.items():
    player_sum, dealer_card, usable_ace = state
    data.append({
        'player_sum': player_sum,
        'dealer_card': dealer_card,
        'usable_ace': usable_ace,
        'q_value': q_value
})


# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'q_table_pandas.csv'
df.to_csv(csv_file_path, index=False)






















