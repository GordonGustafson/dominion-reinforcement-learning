from cards import *

import numpy as np
import pandas as pd


def game_outcome_to_reward(game_outcome: GameOutcome) -> float:
    if game_outcome == GAME_OUTCOME.WIN:
        return 1.0
    elif game_outcome == GAME_OUTCOME.LOSS:
        return 0.0
    elif game_outcome == GAME_OUTCOME.DRAW:
        return 0.5
    else:
        assert False

def game_history_to_df(state_action_pairs: List[StateActionPair],
                       game_outcome_for_player: List[GameOutcome],
                       player_index: int):
    selected_game_states = (sap.possible_actions[sap.selected_action].game_state
                            for sap in state_action_pairs)
    game_state_df = pd.concat([game_state_to_df(gs, player_index=player_index)
                               for gs in selected_game_states],
                              axis="index", ignore_index=True)


    reward = game_outcome_to_reward(game_outcome_for_player)
    # Using gamma=1 for now.
    reward_df = pd.DataFrame({"reward": [reward] * len(state_action_pairs)}, dtype=np.float32)

    return pd.concat([game_state_df, reward_df], axis="columns")


def game_state_to_df(game_state: GameState, player_index: int):
    if player_index not in [0, 1]:
        raise ValueError("player_index must be 0 or 1.")
    opponent_index = 1 - player_index
    player_vp_lead = (get_total_player_vp(game_state.players[player_index])
                      - get_total_player_vp(game_state.players[opponent_index]))
    num_provinces = num_copies_of_card(game_state.supply, "province")


    non_player_state_df = pd.DataFrame({"player_vp_lead": [player_vp_lead],
                                        "num_provinces_remaining": [num_provinces]},
                                       dtype=np.float32)

    player_df = player_to_df(game_state.players[player_index]).add_suffix("_self")
    opponent_df = player_to_df(game_state.players[opponent_index]).add_suffix("_opponent")
    return pd.concat([non_player_state_df, player_df, opponent_df], axis="columns")



def player_to_df(player: Player) -> pd.DataFrame:
    # num_cards = len(get_all_player_cards(player))
    return pd.DataFrame({"average_treasure_value": [get_average_treasure_value_per_card(player)]},
                        dtype=np.float32)


def card_counts_to_df(card_counts: CardCounts):
    pass
