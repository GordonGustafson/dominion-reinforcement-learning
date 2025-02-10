from typing import Iterator, Callable
import math

from actions import NUM_ACTIONS
from chooser import Chooser
from featurizer import game_outcome_to_reward

import torch
from torch.utils.data import DataLoader

import lightning as L

import strategies
import play

from torch.utils.data import IterableDataset

import pandas as pd
import featurizer
from pytorch.bias_only import LearnableConstant

from pytorch.dataloader import tensorify_inputs
from pytorch.running_statistics_norm import RunningStatisticsNorm1d

MAX_EPOCHS=1600


class DatasetFromCallable(IterableDataset):
    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        yield self.generate_batch()

def concat_zero_dimensional_tensors(zero_dimensional_tensors: list[torch.Tensor]) -> torch.Tensor:
    one_dimensional_tensors = [t.reshape(1) for t in zero_dimensional_tensors]
    return torch.cat(one_dimensional_tensors)

class PolicyGradientModel(L.LightningModule):
    def __init__(self, policy_model: torch.nn.Module, state_value_model: torch.nn.Module | None):
        super().__init__()
        self.policy_model = policy_model
        self.state_value_model = state_value_model
        self.automatic_optimization = False

    def generate_batch(self):
        choosers = [Chooser(f) for f in [strategies.pytorch_sampled_action_strategy(self.policy_model)] * 2]
        _, _ = play.play_n_games(
            player_names=["model_1", "model_2"],
            choosers=choosers,
            n=1)
        list_of_game_dfs = [featurizer.game_history_to_df(chooser.state_action_pairs,
                                                          chooser.game_outcome,
                                                          player_index)
                            for player_index, chooser in enumerate(choosers)]
        game_df = pd.concat(list_of_game_dfs, axis="index", ignore_index=True)
        state_features = tensorify_inputs(game_df)
        selected_action_probabilities = concat_zero_dimensional_tensors(choosers[0].action_probability_tensors + choosers[1].action_probability_tensors)
        rewards = torch.tensor([game_outcome_to_reward(choosers[0].game_outcome)] * len(choosers[0].action_probability_tensors)
                               + [game_outcome_to_reward(choosers[1].game_outcome)] * len(choosers[1].action_probability_tensors))
        return state_features, selected_action_probabilities, rewards


    def train_dataloader(self):
        dataset = DatasetFromCallable(self.generate_batch)
        # DON'T SET batch_size HERE! Set the argument to play_n_games in generate_batch instead.
        # I'm not sure why setting the batch size doesn't seem to work.
        return DataLoader(dataset=dataset, batch_size=1)

    def policy_model_loss(self, batch):
        state_features, selected_action_probabilities, rewards = batch  # shape: (N, D), (N, 1)
        # A batch size dimension of 1 gets preprended to each tensor by the train_dataloader since it thinks
        # generate_batch returns a single training input. We remove that dimension here.
        state_features = state_features.flatten(0, 1)
        selected_action_probabilities = selected_action_probabilities.flatten(0, 1)
        rewards = rewards.flatten(0, 1)
        if self.state_value_model is not None:
            baseline_rewards = self.state_value_model.forward(state_features)
        else:
            baseline_rewards = 0.5

        train_loss_items = - torch.log(selected_action_probabilities) * (rewards - baseline_rewards)
        return train_loss_items.sum()

    def state_value_model_loss(self, batch):
        state_features, _, rewards = batch  # shape: (N, D), (N, 1)
        # A batch size dimension of 1 gets preprended to each tensor by the train_dataloader since it thinks
        # generate_batch returns a single training input. We remove that dimension here.
        state_features = state_features.flatten(0, 1)
        rewards = rewards.flatten(0, 1)

        state_value_scores = self.state_value_model.forward(state_features).squeeze()
        state_value_predictions = torch.nn.functional.sigmoid(state_value_scores)
        # print(f"reward predictions vs rewards: {torch.transpose(torch.stack([state_value_predictions, rewards]), 0, 1)}")
        train_loss = torch.nn.functional.binary_cross_entropy(state_value_predictions, target=rewards)
        return train_loss.sum()

    def training_step(self, batch, batch_idx):
        for parameter in self.policy_model.parameters():
            print(parameter.data)

        optimizers = self.optimizers()
        policy_model_optimizer = optimizers if self.state_value_model is None else optimizers[0]

        policy_model_optimizer.zero_grad()
        loss = self.policy_model_loss(batch)
        self.manual_backward(loss)
        policy_model_optimizer.step()

        if self.state_value_model is not None:
            state_value_model_optimizer = optimizers[1]
            state_value_model_optimizer.zero_grad()
            loss = self.state_value_model_loss(batch)
            self.manual_backward(loss)
            state_value_model_optimizer.step()

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        policy_model_optimizer = torch.optim.AdamW(self.policy_model.parameters(),
                                      # math.exp(-2) too large, math.exp(-6) too small
                                      lr=math.exp(-4),
                                      betas=(0.9, 0.999),
                                      weight_decay=0)
        optimizers = [policy_model_optimizer]
        if self.state_value_model is not None:
            state_value_model_optimizer = torch.optim.AdamW(self.state_value_model.parameters(),
                                          lr=math.exp(-5),
                                          betas=(0.9, 0.999),
                                          weight_decay=0)
            optimizers.append(state_value_model_optimizer)
        return optimizers
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        #                                                    max_lr=math.exp(-3),
        #                                                    total_steps=MAX_EPOCHS,
        #                                                    anneal_strategy='cos',
        #                                                    cycle_momentum=True,
        #                                                    base_momentum=0.85,
        #                                                    max_momentum=0.95,
        #                                                    div_factor=math.exp(1),
        #                                                    final_div_factor=math.exp(1))
        return [policy_model_optimizer, state_value_model_optimizer]

def get_policy_model():
    num_input_features = 1
    num_model_outputs = NUM_ACTIONS
    linear_layer = torch.nn.Linear(num_input_features, num_model_outputs, bias=True)
    torch.nn.init.xavier_uniform_(linear_layer.weight, gain=1.0)
    torch.nn.init.zeros_(linear_layer.bias)
    return torch.nn.Sequential(
        RunningStatisticsNorm1d(num_input_features, momentum=0.0001, affine=False),
        linear_layer
    )

    # hidden_layer_width = 8
    # return torch.nn.Sequential(
    #     # torch.nn.BatchNorm1d(num_input_features, affine=False),
    #     # torch.nn.Linear(num_input_features, hidden_layer_width, bias=True),
    #     # torch.nn.ReLU(),
    #     # torch.nn.Linear(hidden_layer_width, hidden_layer_width, bias=True),
    #     # torch.nn.ReLU(),
    #     # torch.nn.BatchNorm1d(hidden_layer_width, affine=True),

    #     # torch.nn.Linear(hidden_layer_width, num_model_outputs, bias=True)
    # )



def get_state_value_model():
    return None
    # num_input_features = 7
    # hidden_layer_width = 4
    # num_model_outputs = 1
    # model = torch.nn.Sequential(
    #     torch.nn.BatchNorm1d(num_input_features, affine=False),

    #     torch.nn.Linear(num_input_features, hidden_layer_width),
    #     torch.nn.ReLU(),
    #     torch.nn.BatchNorm1d(hidden_layer_width, affine=True),

    #     torch.nn.Linear(hidden_layer_width, num_model_outputs, bias=True)
    # )

    # return model

def train_reinforce_model():
    policy_model = get_policy_model()
    state_value_model = get_state_value_model()
    wrapped_model = PolicyGradientModel(policy_model=policy_model, state_value_model=state_value_model)
    trainer = L.Trainer(max_epochs=MAX_EPOCHS)
    trainer.fit(model=wrapped_model)

    policy_model.eval()
    games_df, win_rates = play.play_n_games(
        player_names=["model_chooser", "big_money_provinces_only"],
        choosers=[Chooser(strategies.pytorch_max_action_score_strategy(policy_model)),
                  Chooser(strategies.big_money_provinces_only)],
        n=50)
    print(win_rates)

    print("Actions taken by policy_model in sample games:")
    for i in range(10):
        choosers = [Chooser(strategies.pytorch_max_action_score_strategy(policy_model)),
                    Chooser(strategies.big_money_provinces_only)]
        game_df, _ = play.play_n_games(
            player_names=["model_chooser", "big_money_provinces_only"],
            choosers=choosers,
            n=1)
        for state_action_pair in choosers[0].state_action_pairs:
            selected_choice = state_action_pair.possible_actions[state_action_pair.selected_action]
            print(selected_choice.action.get_description())
