from cards import *
from chooser import Chooser
from pytorch.dataloader import DominionDataset, collate_fn
from pytorch.model import DominionModel

import torch
from torch.utils.data import DataLoader

import lightning as L

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
    games_df = play_n_games_and_get_dataframe(random_chooser_funcs, n=100)
    dataset = DominionDataset(games_df)
    train_dataloader = DataLoader(dataset=dataset, batch_size=1024, collate_fn=collate_fn, shuffle=True)

    model = DominionModel(model=torch.nn.Linear(4, 1))
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    print(model.model.weight)
    print(model.model.bias)

