from cards import *

from typing import NamedTuple, List, Optional, Union

# TODO: make this a proper Python enum
class GAME_OUTCOME:
    WIN = "WIN"
    LOSS = "LOSS"

GameOutcome = Union[GAME_OUTCOME.WIN, GAME_OUTCOME.LOSS]

StateActionPair = NamedTuple("StateActionPair", [
    ("state", GameState),
    ("possible_actions", List[Choice]),
    ("selected_action", int),
])

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

    def record_win(self):
        self._game_outcome = GAME_OUTCOME.WIN

    def record_loss(self):
        self._game_outcome = GAME_OUTCOME.LOSS
