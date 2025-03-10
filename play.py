from actions import Action
from cards import GameOutcome
from game import game_flow
from chooser import Chooser

import featurizer

import pandas as pd


def game_outcome_to_number_of_wins(game_outcome: GameOutcome) -> float:
    if game_outcome == GameOutcome.WIN:
        return 1.0
    elif game_outcome == GameOutcome.LOSS:
        return 0.0
    elif game_outcome == GameOutcome.DRAW:
        return 0.5
    else:
        raise ValueError(f"Unknown GameOutcome: {game_outcome}")

def play_game(player_names, choosers, action_to_reward: dict[Action, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    game_flow(player_names, choosers)
    list_of_game_dfs = [featurizer.game_history_to_df(chooser.state_action_pairs,
                                                      chooser.game_outcome,
                                                      player_index,
                                                      action_to_reward)
                        for player_index, chooser in enumerate(choosers)]
    for chooser in choosers:
        # Clear the state action pairs we just consumed
        chooser.state_action_pairs = []

    game_df = pd.concat(list_of_game_dfs, axis="index", ignore_index=True)
    player_name_to_number_of_wins = {
        player_name: game_outcome_to_number_of_wins(chooser.game_outcome)
        for player_name, chooser
        in zip(player_names, choosers)
    }

    return game_df, player_name_to_number_of_wins

def play_n_games(player_names,
                 choosers,
                 n: int,
                 action_to_reward: dict[Action, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    list_of_tuples = [play_game(player_names, choosers, action_to_reward) for _ in range(n)]
    game_dfs, player_name_to_number_of_wins_dicts = zip(*list_of_tuples)
    game_df = pd.concat(game_dfs, axis="index", ignore_index=True)

    player_name_to_number_of_wins_dict = {
        key: sum(d[key] for d in player_name_to_number_of_wins_dicts)
        for key in player_name_to_number_of_wins_dicts[0]
    }

    return game_df, player_name_to_number_of_wins_dict
