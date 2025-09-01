import cv2
from computervision import card_classification, card_detection
import pandas as pd
import numpy as np
from bjack import BlackjackEnv

cards_v = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
            '8': 8, '9': 9, '10': 10, '11': 10, '12': 10, '13': 10}

actions = {0: "STICK", 1: "HIT", 2: "SPLIT", 3: "SURRENDER", 4: "DOUBLE DOWN"}

env = BlackjackEnv()

def hand_value_and_ace(cards, cards_v):
    if not cards:
        return 0, False
    
    aces = 0
    tot = 0

    for rank, suit in cards:
        if rank == "1":
            tot += 11
            aces += 1
        else:
            tot += cards_v[rank]

    while aces > 0 and tot > 21:
        tot -= 10
        aces -= 1
    if aces:
        aces = 1
    
    return tot, aces


def can_split(cards):
    return len(cards) == 2 and cards[0][0] == cards[1][0]

q_table = pd.read_csv("blackjack_DQN_800.csv")
cols = ['player_sum', 'dealer_card', 'usable_ace', 'allow_split', 'allow_dd']
q_table = q_table.set_index(cols)


def main():
    # We left one image to test the project works. To run it using a video comment out the line 51 (frame = cv2.imread("computervision/images/Games/test3.jpg"))
    # and uncomment all the lines under //// sign

    # //// uncomment line below if using video
    #cap = cv2.VideoCapture()
    frame = cv2.imread("computervision/images/Games/test3.jpg")

    while True:
        dealers_cards = []
        players_cards = []
        # //// uncomment 4 lines below if using video
        # ret, frame = cap.read()
        # frame = resized_frame = cv2.resize(frame, (3000, 4000), interpolation=cv2.INTER_LINEAR)
        # if not ret:
        #      break

        # split frame in top and bottom half
        h, w, _ = frame.shape
        # draw a horizontal line to separate dealer and player
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 2)
        cv2.putText(frame, f"Dealer's half", (10, h - round(0.97 * h)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
        cv2.putText(frame, f"Player's half", (10, h//2 + round(h * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
        cv2.putText(frame, f"Press q to exit", (10, h - round(0.94 * h)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

        # process frame
        ths = card_detection.preprocess(frame)
        cards_contours, n_of_cards = card_detection.find_cards(ths)
        if len(cards_contours) > 0:
            for card_contour in cards_contours:
                rk, su, x, y = card_classification.preprocess_card(card_contour, frame)
                rank, suit, rank_diff, suit_diff, up_img = card_classification.match_cards(rk, su, "/Users/alessandroorsi/blackjack01/computervision/images/templates", frame, x, y)
                if y < h // 2:
                    dealers_cards.append((rank, suit))
                else:
                    players_cards.append((rank, suit))
                frame = up_img

        players_sum, usable_ace = hand_value_and_ace(players_cards[:2], cards_v)
        dealers_sum, _ = hand_value_and_ace(dealers_cards[:1], cards_v)

        current_state = (players_sum, dealers_sum, usable_ace, 1 * can_split(players_cards[:2]), 1 * (len(players_cards[:2]) == 2))
        q_cols = ['q0', 'q1', 'q2', 'q3', 'q4']
        q_state = q_table.loc[current_state, q_cols]

        masked_q_values = np.where(np.array(env.mask_for(current_state)), q_state.values, -np.inf)
        best_action = actions[np.argmax(masked_q_values)]

        cv2.putText(frame, f"Best Move:{best_action}", (10, h // 2 + round(0.06 * h)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

        cv2.imshow("Detected Cards", frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print("Dealer's Cards:", dealers_cards)
            # print("Player's Cards:", players_cards)
            break
    # //// uncomment line below if using video
    #cap.release()
    cv2.destroyAllWindows()

main()