from cards import game_flow, GameOutcome, GAME_OUTCOME
from chooser import Chooser

import featurizer

import pandas as pd


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
