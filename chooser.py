from cards import *

from typing import List, Optional

from game import Choice, StateActionPair
import torch


class Chooser(object):
    def __init__(self, chooser_func):
        self.chooser_func = chooser_func
        self.state_action_pairs: List[StateActionPair] = []
        self.game_outcome: Optional[GameOutcome] = None
        self.action_probability_tensors: List[torch.Tensor] = []
        self.valid_action_distribution_entropies: List[torch.Tensor] = []

    def make_choice(self, game_state: GameState, choices: List[Choice], player_index: int) -> int:
        selected_action = self.chooser_func(self, game_state, choices, player_index)

        state_action_pair = StateActionPair(state=game_state,
                                            possible_actions=choices,
                                            selected_action=selected_action)
        self.state_action_pairs.append(state_action_pair)

        return selected_action
