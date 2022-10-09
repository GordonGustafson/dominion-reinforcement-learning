from cards import *

import strategies

if __name__ == '__main__':
    game_flow(num_players=2, choosers=[strategies.big_money_until_province_then_all_victory,
                                       strategies.user_chooser])
