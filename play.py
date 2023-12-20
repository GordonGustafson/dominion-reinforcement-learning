from cards import *
from chooser import Chooser

import strategies


if __name__ == '__main__':
    player_names = ["player 1", "player 2"]
    chooser_funcs = [strategies.big_money_until_province_then_all_victory,
                     strategies.big_money_until_province_then_all_victory]
    choosers = [Chooser(f) for f in chooser_funcs]
    game_flow(player_names, choosers)
