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


def game_outcome_to_number_of_wins(game_outcome: GameOutcome) -> float:
    if game_outcome == GAME_OUTCOME.WIN:
        return 1.0
    elif game_outcome == GAME_OUTCOME.LOSS:
        return 0.0
    elif game_outcome == GAME_OUTCOME.DRAW:
        return 0.5
    else:
        assert False

def play_game(player_names, chooser_funcs) -> tuple[pd.DataFrame, dict[str, float]]:
    choosers = [Chooser(f) for f in chooser_funcs]
    game_flow(player_names, choosers)
    list_of_game_dfs = [featurizer.game_history_to_df(chooser._state_action_pairs,
                                                      chooser._game_outcome,
                                                      player_index)
                        for player_index, chooser in enumerate(choosers)]
    game_df = pd.concat(list_of_game_dfs, axis="index", ignore_index=True)
    player_name_to_number_of_wins = {
        player_name: game_outcome_to_number_of_wins(chooser._game_outcome)
        for player_name, chooser
        in zip(player_names, choosers)
    }

    return game_df, player_name_to_number_of_wins

def play_n_games(player_names, chooser_funcs, n: int) -> tuple[pd.DataFrame, dict[str, float]]:
    list_of_tuples = [play_game(player_names, chooser_funcs) for _ in range(n)]
    game_dfs, player_name_to_number_of_wins_dicts = zip(*list_of_tuples)
    game_df = pd.concat(game_dfs, axis="index", ignore_index=True)

    player_name_to_number_of_wins_dict = {
        key: sum(d[key] for d in player_name_to_number_of_wins_dicts)
        for key in player_name_to_number_of_wins_dicts[0]
    }

    return game_df, player_name_to_number_of_wins_dict

def train_scikit_learn_linear_model():
    random_chooser_funcs = [strategies.random_strategy,
                            strategies.random_strategy]
    games_df, _ = play_n_games(["random_chooser_1", "random_chooser_2"], random_chooser_funcs, n=20)

    print(games_df)
    X = games_df.drop(columns=["reward"])
    y = games_df["reward"]
    model = LinearRegression().fit(X, y)
    print(model.coef_)
    print(model.intercept_)

    trained_chooser_func = strategies.scikit_learn_model_strategy(model)
    trained_chooser_funcs = [trained_chooser_func] * 2

    trained_games_df, _ = play_n_games(["linear_model_1", "linear_model_2"], trained_chooser_funcs, n=2)

def train_pytorch_model(games_df: pd.DataFrame, model, num_epochs: int) -> None:
    dataset = DominionDataset(games_df)
    train_dataloader = DataLoader(dataset=dataset, batch_size=1024, collate_fn=collate_fn, shuffle=True)

    wrapped_model = DominionModel(model=model)
    trainer = L.Trainer(max_epochs=num_epochs)
    trainer.fit(model=wrapped_model, train_dataloaders=train_dataloader)

if __name__ == '__main__':
    epsilons = [1.0, 2**-1, 2**-2, 2**-3, 2**-4, 2**-5]
    num_epochs_per_data_collection_round = 20

    num_input_features = 4
    hidden_layer_width = 4
    num_model_outputs = 1
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_input_features, affine=False),

        torch.nn.Linear(num_input_features, hidden_layer_width),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(hidden_layer_width, affine=True),

        torch.nn.Linear(hidden_layer_width, num_model_outputs, bias=False)
    )

    for epsilon in epsilons:
        model.eval()
        strategy = strategies.wrap_with_epsilon_greedy(strategies.pytorch_model_strategy(model), epsilon=epsilon)
        games_df, player_name_to_number_of_wins = play_n_games(["model_1", "model_2"], [strategy] * 2, n=100)
        model.train()
        train_pytorch_model(games_df, model, num_epochs=num_epochs_per_data_collection_round)

    model.eval()
    model_games, win_rates = play_n_games(["model_chooser", "big_money_provinces_only"],
                                         [strategies.pytorch_model_strategy(model), strategies.big_money_provinces_only],
                                         n=100)
    print(win_rates)
    for parameter in model.parameters():
        print(parameter.data)
