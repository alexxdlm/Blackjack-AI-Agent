import gymnasium as gym
import numpy as np
from collections import defaultdict
import tqdm
import pandas as pd

# Blackjack example
# Training an agent 




# Infinite deck scenario (no card counting), agent is the only player against the dealer

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

    def get_action(self, obs, allow_split: bool) -> int:
        # balance exploration with experience through greedy epsilon approach
        
        # with probability epsilon we act randomly

        if np.random.random() < self.epsilon: 
            if allow_split:
                #return self.env.action_space.sample()
                return np.random.choice([0,1])
            else:
                return np.random.choice([0,1])
        else: # Exploitation
            q_values = self.q_values[obs].copy()
            
            #if not allow_split:
            #q_values[2] = -np.inf
                
            return int(np.argmax(q_values))

    def update(
        self, 
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
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

learning_rate = 0.1      # How fast to learn (higher = faster but less stable)
n_episodes = 10_000      # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
ep = 1 
# create the environment and the agent

env = gym.make("Blackjack-v1", sab=False, natural=False)

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
    #print(f"We are in episode {ep}")
    obs, info = env.reset()
    done = False
    it = 0
    #print(f"The hand is {info["hand1"]}")
    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        #print(f"card1 is {info['hand1'][0]}, card2 is {info['hand1'][1]}")
        #allow_split = it == 0 and info["hand1"][0] == info["hand1"][1]
        allow_split = False
        action = agent.get_action(obs, allow_split)
            
        it += 1
        #input("Press any key to continue ...")
        # Take action and observe result

        next_obs, reward, terminated, truncated = env.step(action)[0:4]
        # Learn from this experience
        # If we split next_obs will be an array of 2 states and terminated has to be False

        #if action != 2:
        agent.update(obs, action, reward, terminated, next_obs)

        #TODO set the reward for splitting equal to zero
        # Move to next state
        done = terminated or truncated
        obs = next_obs
        
        """else:
            obs1, obs2 = next_obs 
            hand1_done = False
            hand2_done = False
            # agent.update(obs, action, reward, False, obs1)
            # agent.update(obs, action, reward, False, obs2)
            terminated1, terminated2 = False, False

            while not hand1_done:
                action1 = agent.get_action(obs1, allow_split=False)
                it +=1
                next_obs1, hand1_reward, terminated1, truncated1, info1 = env.step(action1)
                agent.update(obs1, action1, hand1_reward, terminated1 or truncated1, next_obs1)                
                obs1 = next_obs1
                hand1_done = terminated1 or truncated1
            
            while not hand2_done:
                action2 = agent.get_action(obs2, allow_split=False)
                it+=1
                next_obs2, hand2_reward, terminated2, truncated2, info2 = env.step(action2)
                agent.update(obs2, action2, hand2_reward, terminated2 or truncated2, next_obs2)
                obs2 = next_obs2
                hand2_done = terminated2 or truncated2
            
            total_split_reward = hand1_reward + hand2_reward
            
            agent.update(obs, action, total_split_reward, terminated=True, next_obs=obs) # doesn't really matter what we put as next_obs since terminated=True"""

            #done = terminated1 and terminated2



    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()
    ep += 1
    #print(f"Reward was {reward}")
    

from matplotlib import pyplot as plt

print("\n")

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
plt.savefig("training_results.png")

# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in tqdm(range(num_episodes)):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        it = 0
        while not done:
            allow_split = it == 0 and info["hand1"][0] == info["hand1"][1]

            action = agent.get_action(obs, allow_split)
            it += 1
            next_obs, reward, terminated, truncated = env.step(action)[0:4]

            if action == 2:

                obs1, obs2 = next_obs 
                hand1_done = False
                hand2_done = False
                # agent.update(obs, action, reward, False, obs1)
                # agent.update(obs, action, reward, False, obs2)
                terminated1, terminated2 = False, False

                while not hand1_done:
                    action = agent.get_action(obs1, allow_split=False)
                    it += 1
                    next_obs1, hand1_reward, terminated1, truncated1, info1 = env.step(action)
                    obs1 = next_obs1
                    hand1_done = terminated1 or truncated1
                
                while not hand2_done:
                    action = agent.get_action(obs2, allow_split=False)
                    it += 1
                    next_obs2, hand2_reward, terminated2, truncated2, info2 = env.step(action)
                    obs2 = next_obs2
                    hand2_done = terminated2 or truncated2
                

                done = terminated1 and terminated2
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                it +=1

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

import signal
# Test your agent
"""try:
    test_agent(agent, env)
except TimeoutError as e:
    print(e)
    print(action, info)
finally:
    signal.alarm(0)"""


#print(agent.q_values)
test_agent(agent, env)

data = []

# print(agent.q_values.items())
try:
    for (state, q_value) in agent.q_values.items():
        card1, player_sum, dealer_card, usable_ace = state
        data.append({
            'cards': card1,
            'player_sum': player_sum,
            'dealer_card': dealer_card,
            'usable_ace': usable_ace,
            'q_value': q_value
    })
except ValueError as e:
    print(f'state is {state}, {q_value}')

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'q_table_pandas.csv'
df.to_csv(csv_file_path, index=False)