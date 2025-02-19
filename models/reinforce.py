from typing import Iterator, Callable
import math

from pathlib import Path
import matplotlib.pyplot as plt

from actions import NUM_ACTIONS, GainMostExpensiveCardAvailable, GainCardInsteadOfMoreExpensiveCard
from cards import card_name_to_card, CARD_LIST
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

from pytorch.dataloader import tensorify_inputs, NUM_INPUT_FEATURES
from pytorch.running_statistics_norm import RunningStatisticsNorm1d
from pytorch.sum_modules import SumModules

MAX_EPOCHS=6400
VALIDATION_GAMES=50
VP_REWARD_MULTIPLIER = 0.00
ACTION_TO_REWARD = {}
for card in CARD_LIST:
    _card_reward = VP_REWARD_MULTIPLIER * (card.vp_effects[0].value if len(card.vp_effects) > 0 else 0)
    ACTION_TO_REWARD[GainMostExpensiveCardAvailable(card)] = _card_reward
    ACTION_TO_REWARD[GainCardInsteadOfMoreExpensiveCard(card)] = _card_reward


class DatasetFromCallable(IterableDataset):
    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        yield self.generate_batch()

def concat_zero_dimensional_tensors(zero_dimensional_tensors: list[torch.Tensor]) -> torch.Tensor:
    one_dimensional_tensors = [t.reshape(1) for t in zero_dimensional_tensors]
    return torch.cat(one_dimensional_tensors)

class PolicyGradientModel(L.LightningModule):
    def __init__(self, policy_model: torch.nn.Module,
                 state_value_model: torch.nn.Module | None,
                 entropy_loss_weight: float,
                 output_path: Path):
        super().__init__()
        self.policy_model = policy_model
        self.state_value_model = state_value_model
        self.entropy_loss_weight = entropy_loss_weight
        self.output_path = output_path
        self.output_path.mkdir(exist_ok=True)

        self.automatic_optimization = False
        self.val_epochs = []
        self.win_rate_metrics = []


    def generate_batch(self):
        chooser_function = strategies.combination_of_gaining_strategy_and_playing_strategy(
            gaining_strategy=strategies.wrap_with_epsilon_greedy(strategies.pytorch_sampled_action_strategy(self.policy_model,
                                                                                                            temperature=math.exp(5)),
                                                                 epsilon=0.0),
            playing_strategy=strategies.play_plus_actions_first)
        choosers = [Chooser(f) for f in [chooser_function] * 2]
        _, _ = play.play_n_games(
            player_names=["model_1", "model_2"],
            choosers=choosers,
            n=1,
            action_to_reward=ACTION_TO_REWARD)
        # for tensor in choosers[0].valid_action_probabilities:
        #     print(tensor.max())
        #all_model_1_valid_action_probabilities = torch.stack(choosers[0].valid_action_probabilities, dim=0)
        #print(f"max action probabilities: {all_model_1_valid_action_probabilities.max(dim=1)}")
        list_of_game_dfs = [featurizer.game_history_to_df(chooser.state_action_pairs,
                                                          chooser.game_outcome,
                                                          player_index,
                                                          action_to_reward=ACTION_TO_REWARD)
                            for player_index, chooser in enumerate(choosers)]
        game_df = pd.concat(list_of_game_dfs, axis="index", ignore_index=True)
        state_features = tensorify_inputs(game_df)
        selected_action_probabilities = concat_zero_dimensional_tensors(choosers[0].action_probability_tensors +
                                                                        choosers[1].action_probability_tensors)
        valid_action_distribution_entropies = concat_zero_dimensional_tensors(choosers[0].valid_action_distribution_entropies +
                                                                              choosers[1].valid_action_distribution_entropies)
        rewards = torch.tensor([game_outcome_to_reward(choosers[0].game_outcome)] * len(choosers[0].action_probability_tensors)
                               + [game_outcome_to_reward(choosers[1].game_outcome)] * len(choosers[1].action_probability_tensors))
        return state_features, selected_action_probabilities, valid_action_distribution_entropies, rewards


    def train_dataloader(self):
        dataset = DatasetFromCallable(self.generate_batch)
        # DON'T SET batch_size HERE! Set the argument to play_n_games in generate_batch instead.
        # I'm not sure why setting the batch size doesn't seem to work.
        return DataLoader(dataset=dataset, batch_size=1)

    def policy_model_loss(self, batch):
        state_features, selected_action_probabilities, valid_action_distribution_entropies, rewards = batch  # shape: (N, D), (N, 1)
        # A batch size dimension of 1 gets preprended to each tensor by the train_dataloader since it thinks
        # generate_batch returns a single training input. We remove that dimension here.
        state_features = state_features.flatten(0, 1)
        selected_action_probabilities = selected_action_probabilities.flatten(0, 1)
        rewards = rewards.flatten(0, 1)
        if self.state_value_model is not None:
            baseline_rewards = self.state_value_model.forward(state_features)
        else:
            baseline_rewards = 0.5

        log_selected_action_probabilities = torch.log(selected_action_probabilities)
        train_loss_items = - log_selected_action_probabilities * (rewards - baseline_rewards)
        total_train_loss = train_loss_items.sum()
        total_entropy_loss = - self.entropy_loss_weight * valid_action_distribution_entropies.sum()
        print(f"total_train_loss: {total_train_loss}")
        print(f"weighted total_entropy_loss: {total_entropy_loss}")
        return total_train_loss + total_entropy_loss

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

    def val_dataloader(self):
        # Dataloader returning nothing, since the real logic happens in validation_step.
        dataset = DatasetFromCallable(lambda: [0])
        return DataLoader(dataset=dataset, batch_size=1)

    def validation_step(self, batch, batch_idx):
        self.policy_model.eval()
        _, win_rates = play.play_n_games(
            player_names=["model_chooser", "big_money_provinces_only"],
            choosers=[
                Chooser(strategies.combination_of_gaining_strategy_and_playing_strategy(
                    gaining_strategy=strategies.pytorch_max_action_score_strategy(self.policy_model),
                    playing_strategy=strategies.play_plus_actions_first)),
                Chooser(strategies.big_money_provinces_only)],
            n=VALIDATION_GAMES,
            action_to_reward=ACTION_TO_REWARD)

        self.val_epochs.append(self.current_epoch)
        self.win_rate_metrics.append(win_rates["model_chooser"])
        win_percentage = win_rates["model_chooser"] / VALIDATION_GAMES
        filename = f"epoch={self.current_epoch}-win_percentage={win_percentage}.ckpt"
        torch.save(self.policy_model.state_dict(), self.output_path / filename)

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
        policy_model_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(policy_model_optimizer,
                                                                        max_lr=math.exp(-4),
                                                                        total_steps=MAX_EPOCHS,
                                                                        pct_start=0.3,
                                                                        anneal_strategy='cos',
                                                                        cycle_momentum=False,
                                                                        base_momentum=0.9,
                                                                        max_momentum=0.9,
                                                                        div_factor=1,
                                                                        final_div_factor=math.exp(1))

        lr_schedulers = [policy_model_lr_scheduler]
        # lr_schedulers = []
        return optimizers, lr_schedulers

def get_policy_model():
    hidden_layer_width = 16
    num_model_outputs = NUM_ACTIONS
    final_linear_layer = torch.nn.Linear(NUM_INPUT_FEATURES, num_model_outputs, bias=True)
    torch.nn.init.xavier_uniform_(final_linear_layer.weight, gain=1.0)
    torch.nn.init.zeros_(final_linear_layer.bias)

    # return torch.nn.Sequential(
    #     RunningStatisticsNorm1d(NUM_INPUT_FEATURES, momentum=0.0001, affine=False),
    #     final_linear_layer,
    # )
    return torch.nn.Sequential(
        RunningStatisticsNorm1d(NUM_INPUT_FEATURES, momentum=0.0001, affine=True),
        SumModules([
            final_linear_layer,
            torch.nn.Sequential(
                torch.nn.Linear(NUM_INPUT_FEATURES, hidden_layer_width, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_layer_width, num_model_outputs, bias=False),
            ),
        ])
    )



def get_state_value_model():
    return None
    # NUM_INPUT_FEATURES = 7
    # hidden_layer_width = 4
    # num_model_outputs = 1
    # model = torch.nn.Sequential(
    #     torch.nn.BatchNorm1d(NUM_INPUT_FEATURES, affine=False),

    #     torch.nn.Linear(NUM_INPUT_FEATURES, hidden_layer_width),
    #     torch.nn.ReLU(),
    #     torch.nn.BatchNorm1d(hidden_layer_width, affine=True),

    #     torch.nn.Linear(hidden_layer_width, num_model_outputs, bias=True)
    # )

    # return model

def plot(x, y):
    plt.plot(x, y, marker='o', linestyle='-', color='r')
    plt.grid(True)
    plt.show()

def train_reinforce_model(output_path: Path):
    policy_model = get_policy_model()
    state_value_model = get_state_value_model()
    wrapped_model = PolicyGradientModel(policy_model=policy_model,
                                        state_value_model=state_value_model,
                                        entropy_loss_weight=0.0,
                                        output_path=output_path)
    trainer = L.Trainer(max_epochs=MAX_EPOCHS, check_val_every_n_epoch=200)
    trainer.fit(model=wrapped_model)

    print(f"policy_model.win_rate_metrics: {MAX_EPOCHS} games: {wrapped_model.win_rate_metrics} peak: {max(wrapped_model.win_rate_metrics)}")
    plot(x=wrapped_model.val_epochs, y=wrapped_model.win_rate_metrics)

    print("Actions taken by policy_model in sample games:")
    for i in range(10):
        choosers=[
            Chooser(strategies.combination_of_gaining_strategy_and_playing_strategy(
                gaining_strategy=strategies.pytorch_max_action_score_strategy(policy_model),
                playing_strategy=strategies.play_plus_actions_first)),
            Chooser(strategies.big_money_provinces_only)]
        game_df, _ = play.play_n_games(
            player_names=["model_chooser", "big_money_provinces_only"],
            choosers=choosers,
            n=1,
            action_to_reward=ACTION_TO_REWARD)
        for state_action_pair in choosers[0].state_action_pairs:
            selected_choice = state_action_pair.possible_actions[state_action_pair.selected_action]
            print(selected_choice.action.get_description())

