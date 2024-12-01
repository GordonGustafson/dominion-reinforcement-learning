from cards import *
from chooser import Chooser

import strategies
import featurizer

import pandas as pd
from sklearn.linear_model import LinearRegression


def play_game_and_get_dataframe(chooser_funcs) -> pd.DataFrame:
    player_names = ["player 1", "player 2"]
    choosers = [Chooser(f) for f in chooser_funcs]
    game_flow(player_names, choosers)
    return featurizer.game_history_to_df(choosers[0]._state_action_pairs,
                                         choosers[0]._game_outcome)

def play_n_games_and_get_dataframe(chooser_funcs, n: int) -> pd.DataFrame:
    dfs = [play_game_and_get_dataframe(chooser_funcs) for _ in range(n)]
    return pd.concat(dfs, axis="index", ignore_index=True)

if __name__ == '__main__':
    chooser_funcs = [strategies.random_strategy,
                     strategies.random_strategy]
    games_df = play_n_games_and_get_dataframe(chooser_funcs, n=1000)

    print(games_df)
    X = games_df[["player_1_vp_lead", "num_provinces_remaining", "average_treasure_value_player_1", "average_treasure_value_player_2"]]
    y = games_df["reward"]
    reg = LinearRegression().fit(X, y)
    print(reg.coef_)
    print(reg.intercept_)
    # array([1., 2.])
    # reg.intercept_
    # np.float64(3.0...)
    # reg.predict(np.array([[3, 5]]))
