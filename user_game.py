from chooser import Chooser

import strategies
import play

games_df, win_rates = play.play_n_games(
    player_names=["user_chooser", "big_money_provinces_then_all_victory"],
    choosers=[Chooser(strategies.user_chooser),
              Chooser(strategies.big_money_until_province_then_all_victory)],
    n=1
)