import torch
import csv
import pandas as pd
from DQN import DQN, modules
device = "cpu"

def save_to_csv(model):
	model = DQN(modules)
	model.load_state_dict(torch.load("result_night_800.pt", map_location="cpu"))
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
	df.to_csv("result_night_800.csv", index=False)
	print("Saved to result_night_800.csv")


save_to_csv("result_night_800.pt")

