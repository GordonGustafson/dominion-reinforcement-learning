from cards import *
from chooser import Chooser

import strategies
import featurizer


if __name__ == '__main__':
    player_names = ["player 1", "player 2"]
    chooser_funcs = [strategies.big_money_provinces_only,
                     strategies.big_money_provinces_only]
    choosers = [Chooser(f) for f in chooser_funcs]
    game_flow(player_names, choosers)

    print(featurizer.game_history_to_df(choosers[0]._state_action_pairs,
                                        choosers[0]._game_outcome))
