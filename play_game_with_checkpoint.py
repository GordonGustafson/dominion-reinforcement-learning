from chooser import Chooser
import strategies
import play
from models.reinforce import get_policy_model

import sys
import torch

policy_model_path = sys.argv[1]

policy_model = get_policy_model()
policy_model.load_state_dict(torch.load(policy_model_path, weights_only=True))
policy_model.eval()

model_chooser_function = strategies.combination_of_gaining_strategy_and_playing_strategy(
    gaining_strategy=strategies.pytorch_max_action_score_strategy(policy_model),
    playing_strategy=strategies.play_plus_actions_first)


print("Actions taken by policy_model in sample games:")
for i in range(10):
    choosers = [
        Chooser(strategies.combination_of_gaining_strategy_and_playing_strategy(
            gaining_strategy=strategies.pytorch_max_action_score_strategy(policy_model),
            playing_strategy=strategies.play_plus_actions_first)),
        Chooser(strategies.big_money_provinces_only)]
    game_df, _ = play.play_n_games(
        player_names=["model_chooser", "big_money_provinces_only"],
        choosers=choosers,
        n=1)
    for state_action_pair in choosers[0].state_action_pairs:
        selected_choice = state_action_pair.possible_actions[state_action_pair.selected_action]
        print(selected_choice.action.get_description())


games_df, win_rates = play.play_n_games(
    player_names=["model_chooser", "big_money_provinces_only"],
    choosers=[Chooser(model_chooser_function),
              Chooser(strategies.big_money_provinces_only)],
    n=1000)


print(win_rates)
