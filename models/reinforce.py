from typing import Iterator, Callable
import math

from actions import NUM_ACTIONS
from chooser import Chooser
from featurizer import game_outcome_to_reward
from pytorch.dataloader import DominionDataset, collate_fn, tensorify_inputs, tensorify_reward

import torch
from torch.utils.data import DataLoader

import lightning as L

import strategies
import play

from torch.utils.data import IterableDataset



class DatasetFromCallable(IterableDataset):
    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        yield self.generate_batch()

def concat_zero_dimensional_tensors(zero_dimensional_tensors: list[torch.Tensor]) -> torch.Tensor:
    one_dimensional_tensors = [t.reshape(1) for t in zero_dimensional_tensors]
    return torch.cat(one_dimensional_tensors)

class PolicyGradientModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def generate_batch(self):
        choosers = [Chooser(f) for f in [strategies.pytorch_sampled_action_strategy(self.model)] * 2]
        model_games, _ = play.play_n_games(
            player_names=["model_1", "model_2"],
            choosers=choosers,
            n=1)
        selected_action_probabilities = concat_zero_dimensional_tensors(choosers[0].action_probability_tensors + choosers[1].action_probability_tensors)
        rewards = torch.tensor([game_outcome_to_reward(choosers[0]._game_outcome)] * len(choosers[0].action_probability_tensors)
                               + [game_outcome_to_reward(choosers[1]._game_outcome)] * len(choosers[1].action_probability_tensors))
        return selected_action_probabilities, rewards


    def train_dataloader(self):
        dataset = DatasetFromCallable(self.generate_batch)
        # DON'T SET THIS! Set the argument to play_n_games in generate_batch instead.
        # I'm not sure why setting the batch size doesn't seem to work.
        return DataLoader(dataset=dataset, batch_size=1)

    def forward(self, batch, batch_idx):
        features, reward = batch
        action_scores = self.model.forward(features)
        return action_scores

    def training_step(self, batch, batch_idx):
        for parameter in self.model.parameters():
            print(parameter.data)
        selected_action_probabilities, rewards = batch  # shape: (N, D), (N, 1)
        train_loss_items = - torch.log(selected_action_probabilities) * (rewards - 0.5)
        return train_loss_items.sum()

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(),
                                 # math.exp(-2) too large, math.exp(-6) too small
                                 lr=math.exp(-4),
                                 betas=(0.9, 0.999),
                                 weight_decay=0.00)

def train_reinforce_model():
    num_input_features = 5
    hidden_layer_width = 4
    num_model_outputs = NUM_ACTIONS
    model = torch.nn.Sequential(
        # torch.nn.BatchNorm1d(num_input_features, affine=False),

        torch.nn.Linear(num_input_features, hidden_layer_width),
        torch.nn.ReLU(),
        # torch.nn.BatchNorm1d(hidden_layer_width, affine=True),

        torch.nn.Linear(hidden_layer_width, num_model_outputs, bias=True)
    )
    wrapped_model = PolicyGradientModel(model=model)
    trainer = L.Trainer(max_epochs=800)
    trainer.fit(model=wrapped_model)

    model.eval()
    model_games, win_rates = play.play_n_games(
        player_names=["model_chooser", "big_money_provinces_only"],
        choosers=[Chooser(strategies.pytorch_max_action_score_strategy(model)),
                  Chooser(strategies.big_money_provinces_only)],
        n=50)
    print(win_rates)
