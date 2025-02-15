from chooser import Chooser
from pytorch.dataloader import DominionDataset, collate_fn

import torch
from torch.utils.data import DataLoader

import lightning as L

import strategies
import play

import pandas as pd


class DominionModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, batch_idx):
        features, reward = batch
        raw_scores = self.model.forward(features)
        return torch.nn.functional.sigmoid(raw_scores)

    def training_step(self, batch, batch_idx):
        predicted_rewards = self.forward(batch, batch_idx).squeeze(1)
        features, rewards = batch
        # print(f"predicted_rewards: {predicted_rewards}, rewards: {rewards}")
        for parameter in self.model.parameters():
            print(parameter.data)
        train_loss = torch.nn.functional.binary_cross_entropy(predicted_rewards, target=rewards)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(),
                                 lr=1e-1,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.04)

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
        strategy = strategies.wrap_with_epsilon_greedy(strategies.pytorch_max_state_score_strategy(model), epsilon=epsilon)
        choosers = [Chooser(s) for s in [strategy] * 2]
        games_df, player_name_to_number_of_wins = play.play_n_games(["model_1", "model_2"], choosers, n=num_games_per_data_collection_round)
        model.train()
        train_pytorch_model(games_df, model, num_epochs=num_epochs_per_data_collection_round, batch_size=batch_size)

    model.eval()
    model_games, win_rates = play.play_n_games(["model_chooser", "big_money_provinces_only"],
                                               [Chooser(strategies.pytorch_max_state_score_strategy(model)),
                                                Chooser(strategies.big_money_provinces_only)],
                                               n=100)
    print(win_rates)
    for parameter in model.parameters():
        print(parameter.data)
