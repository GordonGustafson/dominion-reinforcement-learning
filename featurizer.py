from typing import Any

from cards import *

import numpy as np
import pandas as pd

from game import StateActionPair


def game_outcome_to_reward(game_outcome: GameOutcome) -> float:
    if game_outcome == GameOutcome.WIN:
        return 1.0
    elif game_outcome == GameOutcome.LOSS:
        return 0.0
    elif game_outcome == GameOutcome.DRAW:
        return 0.5
    else:
        raise ValueError("Unknown GameOutcome")

def game_history_to_df(state_action_pairs: List[StateActionPair],
                       game_outcome_for_player: GameOutcome,
                       player_index: int):
    selected_game_states = (sap.possible_actions[sap.selected_action].game_state
                            for sap in state_action_pairs)
    game_state_dicts = [game_state_to_dict(gs, player_index=player_index)
                        for gs in selected_game_states]


    reward = game_outcome_to_reward(game_outcome_for_player)
    for d in game_state_dicts:
        # Using gamma=1 for now.
        d["reward"] = reward

    return pd.DataFrame(game_state_dicts, dtype=np.float32)

def game_state_to_dict(game_state: GameState, player_index: int) -> dict[str, Any]:
    if player_index not in [0, 1]:
        raise ValueError("player_index must be 0 or 1.")
    opponent_index = 1 - player_index
    player_vp_lead = (get_total_player_vp(game_state.players[player_index])
                      - get_total_player_vp(game_state.players[opponent_index]))
    num_provinces = num_copies_of_card(game_state.supply, "province")


    non_player_state_dict = {"player_vp_lead": player_vp_lead,
                             "num_provinces_remaining": num_provinces}

    player_dict = player_to_dict(game_state.players[player_index], suffix="_self")
    opponent_dict = player_to_dict(game_state.players[opponent_index], suffix="_opponent")
    return {**non_player_state_dict, **player_dict, **opponent_dict}

def game_state_to_df(game_state: GameState, player_index: int) -> pd.DataFrame:
    return pd.DataFrame([game_state_to_dict(game_state, player_index)], dtype=np.float32)

def player_to_dict(player: Player, suffix: str) -> dict:
    return {"average_treasure_value" + suffix: get_average_treasure_value_per_card(player)}
