from pytorch.dataloader import DominionDataset, collate_fn
from pytorch.model import DominionModel

import torch
from torch.utils.data import DataLoader

import lightning as L

import strategies
import play

import pandas as pd


def train_pytorch_model(games_df: pd.DataFrame, model, num_epochs: int, batch_size: int) -> None:
    dataset = DominionDataset(games_df)
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    wrapped_model = DominionModel(model=model)
    trainer = L.Trainer(max_epochs=num_epochs)
    trainer.fit(model=wrapped_model, train_dataloaders=train_dataloader)

    
def train_value_function_approximation_model():
    batch_size = 1024
    epsilons = [1.0, 2**-1, 2**-2, 2**-3, 2**-4]
    num_games_per_data_collection_round = 800
    num_epochs_per_data_collection_round = 10

    print(f"batch_size: {batch_size}")

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
        strategy = strategies.wrap_with_epsilon_greedy(strategies.pytorch_state_scoring_model_strategy(model), epsilon=epsilon)
        games_df, player_name_to_number_of_wins = play.play_n_games(["model_1", "model_2"], [strategy] * 2, n=num_games_per_data_collection_round)
        model.train()
        train_pytorch_model(games_df, model, num_epochs=num_epochs_per_data_collection_round, batch_size=batch_size)

    model.eval()
    model_games, win_rates = play.play_n_games(["model_chooser", "big_money_provinces_only"],
                                               [strategies.pytorch_state_scoring_model_strategy(model), strategies.big_money_provinces_only],
                                               n=100)
    print(win_rates)
    for parameter in model.parameters():
        print(parameter.data)
