import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pdb


#This is the environment we're working with, based on what gymnasium provided but with substantial additions

def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return int(1 in hand and sum(hand) + 10 <= 21)


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):

    def __init__(self, render_mode: str | None = None, natural=False):
        
        # Action space was modified to include all blackjack actions
        self.action_space = spaces.Discrete(5) # 0: Stand, 1: Hit, 2: Split, 3: Surrender, 4: Double Down
     
        #Observation space was modified to include flags, which tell the agent whether it can take certain actions or not
        #Beware the name "allow_dd" is misleading, allow_dd tells the agent if it can pick both dd AND surrender, since both are
        #available under the same circumstances
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2))
            # player_sum,         dealer_card,         usable_ace,         allow_split,        allow_dd
        )
        self.len_state = 5
        self.len_actions = 5
        self.natural = natural
        self.dealer = []
        self.hand_terminated = False
        self.allow_dd = False
        self.allow_split = False
        self.reward = 0
        self.it = 0

    # Returns the state
    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0],
                 usable_ace(self.player), self.allow_split, self.allow_dd)
        
    # Resets the environment, deals initial cards and sets initial flags
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ):
        super().reset(seed=seed)
        self.reward = 0
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        self.allow_split, self.allow_dd = (self.player[0]==self.player[1]), True

        _, _, dealer_card_value, _, _ = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        return self._get_obs(), {}
    

    # Computes a mask to avoid selecting illegal actions
    def mask_for(self, state: list) -> list:
        #1:legal, 0:illegal
        # (player_sum, dealer_card, usable_ace, allow_split, allow_dd)

        mask = np.ones(self.action_space.n, dtype=np.uint8)
        # Disable split if not allowed
    
        if not state[3]:
            mask[2] = 0  # index for split
        # Disable double down if not allowed
        if not state[4]:
            mask[3] = 0
            mask[4] = 0  # index for double down
        return mask
        

    # Draws a single card
    def draw_card(self, np_random):
        return int(np_random.choice(deck))

    # Draws two cards
    def draw_hand(self, np_random):
        return [draw_card(np_random), draw_card(np_random)]
    
    # Implements hit
    def hit(self):
        self.it+=1
        self.player.append(draw_card(self.np_random))
        self.allow_split, self.allow_dd = False, False
        if is_bust(self.player):
            terminated = True
            self.hand_terminated = True
            self.reward = -1.0
        else:
            terminated = False
            self.reward = 0.0
        return terminated, self.reward
    
    #Implements stand
    def stand(self):
        self.it+=1
        terminated = True
        self.hand_terminated = True
        while sum_hand(self.dealer) < 17:
            self.dealer.append(draw_card(self.np_random))
        self.reward = cmp(score(self.player), score(self.dealer))
        if is_natural(self.player) and not is_natural(self.dealer):
            # Player automatically wins, natural blackjack pays 3:2 (150%)
            self.reward = 1.5
        return terminated, self.reward
    
    # Implements surrender
    def surrender(self):
        self.it+=1
        #self.allow_dd = True
        self.allow_split, self.allow_dd = False, False
        terminated, self.hand_terminated = True, True
        self.reward = -0.5

        return terminated, self.reward
    
    # Implements double down
    def double_down(self):
        self.it+=1
        #self.allow_dd = True
        self.allow_split, self.allow_dd = False, False
        self.player.append(draw_card(self.np_random))
        if is_bust(self.player):
            self.reward = -1.0
        else:
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            self.reward = cmp(score(self.player), score(self.dealer))

        terminated, self.hand_terminated = True, True

        return terminated, 2 * self.reward

    # Takes the action that is selected by the agent
    def step(self, action):
        self.hand_terminated = False
        assert self.action_space.contains(action)

        #STAND
        if action == 0: 
            terminated, self.reward = self.stand()
        
        #HIT
        elif action == 1:  # hit: add a card to players hand and return
            terminated, self.reward = self.hit()
        
        #SPLIT
        elif action == 2:
            # reset the hand to only one card, double the bet and keep playing as usual
            self.player.pop(1)
            result = self.hit()
            terminated, self.reward = result[0], result[1]
        
        #SURRENDER
        elif action == 3:
            terminated, self.reward = self.surrender()

        #DOUBLE DOWN
        elif action == 4:
            terminated, self.reward = self.double_down()

        return self._get_obs(), self.reward, terminated, False, {}
    