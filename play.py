from cards import *

import strategies

def start_game(choosers: List):
    player_names = [chooser.__name__ for chooser in choosers]
    game_flow(player_names, choosers)

if __name__ == '__main__':
    start_game([strategies.big_money_until_province_then_all_victory,
                strategies.big_money_provinces_only])
