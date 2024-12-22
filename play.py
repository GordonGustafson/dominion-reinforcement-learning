from cards import *
from chooser import Chooser
from pytorch.dataloader import DominionDataset

import strategies
import featurizer

import pandas as pd
from sklearn.linear_model import LinearRegression



def play_game_and_get_dataframe(chooser_funcs) -> pd.DataFrame:
    player_names = ["player 1", "player 2"]
    choosers = [Chooser(f) for f in chooser_funcs]
    game_flow(player_names, choosers)
    player_dfs = [featurizer.game_history_to_df(chooser._state_action_pairs,
                                                chooser._game_outcome,
                                                player_index)
                  for player_index, chooser in enumerate(choosers)]

    return pd.concat(player_dfs, axis="index", ignore_index=True)

def play_n_games_and_get_dataframe(chooser_funcs, n: int) -> pd.DataFrame:
    dfs = [play_game_and_get_dataframe(chooser_funcs) for _ in range(n)]
    return pd.concat(dfs, axis="index", ignore_index=True)

def train_linear_model():
    random_chooser_funcs = [strategies.random_strategy,
                            strategies.random_strategy]
    games_df = play_n_games_and_get_dataframe(random_chooser_funcs, n=20)

    print(games_df)
    X = games_df.drop(columns=["reward"])
    y = games_df["reward"]
    model = LinearRegression().fit(X, y)
    print(model.coef_)
    print(model.intercept_)

    trained_chooser_func = strategies.scikit_learn_model_strategy(model)
    trained_chooser_funcs = [trained_chooser_func] * 2

    trained_games_df = play_n_games_and_get_dataframe(trained_chooser_funcs, n=2)


if __name__ == '__main__':
    # train_linear_model()

    random_chooser_funcs = [strategies.random_strategy,
                            strategies.random_strategy]
    games_df = play_n_games_and_get_dataframe(random_chooser_funcs, n=1)
    dataset = DominionDataset(games_df)
    print(dataset[0])

