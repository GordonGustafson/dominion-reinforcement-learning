from cards import *

from typing import List, Optional

class Chooser(object):
    def __init__(self, chooser_func):
        self._chooser_func = chooser_func
        self._state_action_pairs: List[StateActionPair] = []
        self._game_outcome: Optional[GameOutcome] = None

    def make_choice(self, game_state: GameState, choices: List[Choice]) -> int:
        selected_action = self._chooser_func(game_state, choices)

        state_action_pair = StateActionPair(state=GameState,
                                            possible_actions=choices,
                                            selected_action=selected_action)
        self._state_action_pairs.append(state_action_pair)

        return selected_action
