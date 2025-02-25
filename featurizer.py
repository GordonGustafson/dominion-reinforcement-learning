from typing import Any

from actions import Action, action_to_action_id
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
                       player_index: int,
                       action_to_reward: dict[Action, float] | None = None):
    if action_to_reward is None:
        action_to_reward = {}
    selected_game_states = (sap.possible_actions[sap.selected_action].game_state
                            for sap in state_action_pairs)
    game_state_dicts = [game_state_to_dict(gs, player_index=player_index)
                        for gs in selected_game_states]


    game_outcome_reward = game_outcome_to_reward(game_outcome_for_player)
    cumulative_action_reward = 0.0
    for sap, d in reversed(list(zip(state_action_pairs, game_state_dicts))):
        selected_action = sap.possible_actions[sap.selected_action].action
        d[f"player_{player_index}_selected_action_id"] = action_to_action_id(selected_action)
        cumulative_action_reward += action_to_reward.get(selected_action, 0.0)
        # Using gamma=1 for now.
        d["reward"] = game_outcome_reward + cumulative_action_reward

    return pd.DataFrame(game_state_dicts, dtype=np.float32)

def game_state_to_dict(game_state: GameState, player_index: int) -> dict[str, Any]:
    if player_index not in [0, 1]:
        raise ValueError("player_index must be 0 or 1.")
    opponent_index = 1 - player_index
    player_vp_lead = (get_total_player_vp(game_state.players[player_index])
                      - get_total_player_vp(game_state.players[opponent_index]))
    num_provinces = num_copies_of_card(game_state.supply, "province")


    non_player_state_dict = {"player_vp_lead": player_vp_lead,
                             "num_provinces_remaining": num_provinces,
                             "max_turns_per_player": game_state.max_turns_per_player,
                             "two_provinces_remaining": 1 if num_provinces == 2 else 0,
                             "one_province_remaining": 1 if num_provinces == 1 else 0,
                             }

    player_dict = player_to_dict(game_state.players[player_index], suffix="_self")
    # opponent_dict = player_to_dict(game_state.players[opponent_index], suffix="_opponent")
    return {**non_player_state_dict, **player_dict}

def game_state_to_df(game_state: GameState, player_index: int) -> pd.DataFrame:
    return pd.DataFrame([game_state_to_dict(game_state, player_index)], dtype=np.float32)

def player_to_dict(player: Player, suffix: str) -> dict:
    all_player_cards = get_all_player_cards(player)
    result = {
        "average_treasure_value" + suffix: get_average_treasure_value_per_card(player),
        "num_vp" + suffix: get_total_player_vp(player),

        "num_victory_cards_owned" + suffix: (
                all_player_cards[card_name_to_card("estate")]
                + all_player_cards[card_name_to_card("duchy")]
                + all_player_cards[card_name_to_card("province")]
                + all_player_cards[card_name_to_card("curse")]),

        "num_actions_owned_with_plus_zero_actions" + suffix: sum(all_player_cards[card] for card in action_cards_with_plus_n_actions(0)),
        "num_actions_owned_with_plus_one_action" + suffix: sum(all_player_cards[card] for card in action_cards_with_plus_n_actions(1)),
        "num_actions_owned_with_plus_two_actions" + suffix: sum(all_player_cards[card] for card in action_cards_with_plus_n_actions(2)),
    }


    for card in CARD_LIST:
        result[f"num_{card.name}_owned" + suffix] = all_player_cards[card]
        # result[f"zero_{card.name}_owned" + suffix] = all_player_cards[card] == 0
        # result[f"one_{card.name}_owned" + suffix] = all_player_cards[card] == 1
        # result[f"two_{card.name}_owned" + suffix] = all_player_cards[card] == 2

    return result
