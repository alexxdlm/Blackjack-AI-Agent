#DEEP Q-LEARNING
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from blackjack import BlackjackEnv, score, is_bust
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# hyperparameters
batch_size = 256
discount_factor = 0.999
initial_eps = 1
final_eps = 0.05
decay_factor = 20_000
learning_rate = 0.0001

n_layers = 11
k = 13
n_episodes = 800_000
start_period = 0
decay_period = n_episodes - 150000 - start_period
# how often to take gradient descent steps
C = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = BlackjackEnv()
input_layer = env.len_state # TODO maybe action_space.n?
output_layer = env.len_actions

# build network
# this is assigning the number of nodes each hidden layer will have
layers_size = {-1: input_layer}
factor = (output_layer/k/input_layer)**(1/(n_layers - 1))

for layer in range(n_layers):
	layers_size[layer] = int(np.rint(k*input_layer * factor**(layer)))


# this is assigning each layer the activation function
modules = []
for i in layers_size.keys():
	if i == -1:
		continue
	modules.append(nn.Linear(layers_size[i-1],layers_size[i]))
	if i < n_layers - 1:
		modules.append(nn.BatchNorm1d(layers_size[i]))
		modules.append(nn.ReLU())



class ReplayBuffer():
	#In here we store past experiences, in the form [state, action, next_state, reward]
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, transition):
		if len(self.buffer) < self.capacity:
			self.buffer.append(transition)
		else:
			self.buffer[self.position] = transition
			self.position = (self.position + 1) % self.capacity

	def sample(self, batchsize): #TODO ?
		return random.sample(self.buffer, batchsize)

	def __len__(self):
		return len(self.buffer)
	
	

class DQN(nn.Module):
	def __init__(self, modules):
		super(DQN, self).__init__()
		for layer, module in enumerate(modules):
			self.add_module("layer_" + str(layer), module)

	def forward(self, x):
		for layer in self.children():
			x = layer(x)
		return x

# initialize model
model = DQN(modules).to(device)


optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=learning_rate)
memory = ReplayBuffer(20000)
model.eval()
losses = []
rewards = []

counter = 0

# This returns a mask of illegal actions that can't be used in learning/testing
def get_action_mask(state, device):
	#1:legal, 0:illegal
	mask = np.ones(5, dtype=np.uint8)

	#if splitting is legal
	if not state[3]:
		mask[2] = 0

	#if allow_dd is true
	if not state[4]:
		mask[3], mask[4] = 0, 0

	return mask


# Depending on the value of epsilon (whether we are exploring or exploiting),
# this returns either a random action or the best action (out of the available ones) 
def get_action(state, device):
	eps = final_eps + max((initial_eps - final_eps) * (1 - np.exp((counter - start_period - decay_period)/decay_factor)), 0)
	with torch.no_grad():
		scores = model(state)
		
	mask = get_action_mask(state[0], device)
	mask = torch.tensor(mask, dtype=torch.bool, device=device)
	masked_scores = scores.squeeze().clone()
	masked_scores[mask == 0] = -np.inf  # large negative number to disable invalid moves
	best_action = masked_scores.argmax().item()
	#Exploration or exploitation
	if eps < np.random.rand():
		return torch.tensor([best_action], dtype = torch.long, device=device)
	
	else:
		valid_actions = torch.where(mask == 1)[0].tolist()
	return torch.tensor([random.choice(valid_actions) if valid_actions else best_action],
                       dtype=torch.long, device=device)


#This is where the learning happens,
# a batch_size of experience is uniformly sampled 
# and used as target value, to compare with the NN's predictions
def optimize_model():
	global losses
	if len(memory) < batch_size:
		return

	model.train()
	sample = memory.sample(batch_size)

	batch = list(zip(*sample))
	state_batch = torch.cat(batch[0]).to(device)
	action_batch = torch.cat(batch[1]).to(device)
	reward_batch = torch.cat(batch[3]).to(device)

	# Predicts q values for each state in the state batch and selects the q value that corresponds to the
	# action that is actually taken
	Q_sa = model(state_batch).gather(1, action_batch.view(-1,1)).squeeze()

	#Initiliaze the value of each state with 0
	V_s = torch.zeros(batch_size, dtype=torch.float, device=device)
	not_terminal = torch.tensor(tuple(map(lambda s: s is not None, batch[2])), device=device)

	#If there are non terminal states
	if not_terminal.sum() > 0: # 
		model.eval()
		with torch.no_grad():
			not_terminal_states = torch.cat([s for s in batch[2] if s is not None]).to(device)
			
			# computes actions that are illegal (1=legal, 0=illegal) and inverts to use masked_fill,
			# then make it into a boolean tensor ready for use in masked_fill
			masks = torch.tensor(np.array([1 - get_action_mask(s, device=device) for s in not_terminal_states]), dtype=torch.bool, device=device)
			
			# where not terminal is true we compute the q values and replace them with -np.inf if masks==true, meaning the action is illegal
			# and should be masked. Finally we pick the max q value (the best action)
			V_s[not_terminal] = (model(not_terminal_states).masked_fill_(masks, -np.inf)).max(1)[0]
			

		model.train()
	observed_sa = reward_batch + (V_s * discount_factor)
	# Huber loss
	loss = F.smooth_l1_loss(Q_sa, observed_sa)
	losses.append(loss.item())

	optimizer.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
	optimizer.step()
	model.eval()



# training

it = 0


def _train():
	global rewards
	it, counter = 0, 0
	for episode in tqdm(range(n_episodes)):

		state, info = env.reset()
		state = torch.tensor([state], dtype=torch.float, device=device)
		while True:
			action = get_action(state, device)
			next_state, reward, terminated, _, _ = env.step(int(action[0]))
			reward = torch.tensor([reward], dtype=torch.float, device=device)

			if terminated:
				next_state = None
				memory.push([state, action, next_state, reward])
				state = next_state

			else:
				next_state = torch.tensor([next_state], dtype=torch.float, device=device)
				memory.push([state, action, next_state, reward])
				state = next_state

			if counter % C == 0:
				optimize_model()
			counter += 1
			it+=1

			rewards.append(reward[0])
			
			if terminated:
				break

	name = "models/blackjack_DQN_" + str(n_episodes)[0:3] + ".pt"
	torch.save(model.state_dict(), name)
	print(f"saved model as {name}")
	#show_losses(losses, rewards)


# This is used to plot the loss and the rewards, with different windows depending on 
# the number of episodes we're training for
def show_losses(losses: list, rewards: list):
	window = 2000
	smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
	plt.plot(smoothed)
	plt.xlabel('Training step')
	plt.ylabel('Loss')
	plt.title('DQN Training Loss')
	plt.savefig("losses.png")
	plt.close()
	smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
	plt.plot(smoothed)
	plt.xlabel('Training step')
	plt.ylabel('Reward')
	plt.title('DQN Rewards')
	plt.savefig("rewards.png")
	plt.close()



# Win rate between 42-45% is considered a good blackjack performance without card counting

def test_dqn_agent(model, env, num_episodes, device="cpu"):
	rewards = []
	wins, losses, pushes = 0, 0, 0
	for _ in tqdm(range(num_episodes)):
		obs, _ = env.reset()
		state = torch.tensor([obs], dtype=torch.float, device=device)
		done = False
		while not done:
			with torch.no_grad():
				q_values = model(state)

				# Mask the actions that aren't allowed
				mask = torch.tensor(get_action_mask(state[0], device=device), dtype=torch.bool, device=device)  # 1=legal, 0=illegal
				masked_q = q_values.masked_fill(mask == 0, -np.inf)
				action = masked_q.argmax(dim=1).item()
				next_obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			if not done:
				state = torch.tensor([next_obs], dtype=torch.float, device=device)

        # Tracking of statistics
		player_score = score(env.player)
		dealer_score = score(env.dealer)
		rewards.append(reward)
		if is_bust(env.player):
			losses += 1
		elif is_bust(env.dealer):
			wins += 1
		elif player_score > dealer_score:
			wins += 1
		elif player_score < dealer_score:
			losses += 1
		else:
			pushes+=1

			
		
		total_hands = wins + losses + pushes
		win_rate = wins / total_hands if total_hands > 0 else 0.0
	print(f"Over {num_episodes} episodes:")
	print(f"Wins: {wins} ({wins/num_episodes:.1%})")
	print(f"Losses: {losses} ({losses/num_episodes:.1%})")
	print(f"Pushes: {pushes} ({pushes/num_episodes:.1%})")
	print(f"Win rate (wins / total_hands): {win_rate:.1%}")
	print(f'Average return was: {np.mean(np.array(rewards))}')

model.eval()

# convert .pt -> .csv

def save_to_csv(model):
	player_sums = range(4, 22)      # typical blackjack values (4–21)
	dealer_cards = range(1, 11+1)     # 1 = Ace, 2–10
	usable_aces = [0, 1]
	allow_split = [0, 1]
	allow_dd = [0, 1]

	rows = []

	for ps in player_sums:
		for dc in dealer_cards:
			for ua in usable_aces:
				for sp in allow_split:
					for dd in allow_dd:
						state = torch.tensor([[ps, dc, ua, sp, dd]], dtype=torch.float, device=device)
						with torch.no_grad():
							q_values = model(state).squeeze().cpu().numpy()

						rows.append([ps, dc, ua, sp, dd] + q_values.tolist())

	df = pd.DataFrame(
		rows,
		columns=["player_sum", "dealer_card", "usable_ace", "allow_split", "allow_dd", "q0", "q1", "q2", "q3", "q4"]
	)

	# --- Save to CSV ---
	df.to_csv("models/result_night_800.csv", index=False)
	print("Saved to result_night_800.csv")



model = DQN(modules).to(device)
env = BlackjackEnv()

if __name__ == "__main__":
	try:
		# If a model is already loaded, use it and test it
		model.load_state_dict(torch.load("models/result_night_800.pt", map_location=torch.device('cpu')))
		print("loaded saved model")
		model.eval()
		test_dqn_agent(model, env, num_episodes=20_000)
	except FileNotFoundError:
		# If no model is present, train one and test it 
		print("No saved model")
		_train()
		print("Model was trained")
		print("\n")
		model.load_state_dict(torch.load("models/blackjack_DQN_" + str(n_episodes)[0:3] + ".pt",  map_location=torch.device('cpu')))
		save_to_csv(model)
		print("Now testing...")
		test_dqn_agent(model, env, num_episodes=20_000)

