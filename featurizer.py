from typing import Any

from actions import Action
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
    opponent_dict = player_to_dict(game_state.players[opponent_index], suffix="_opponent")
    return {**non_player_state_dict, **player_dict, **opponent_dict}

def game_state_to_df(game_state: GameState, player_index: int) -> pd.DataFrame:
    return pd.DataFrame([game_state_to_dict(game_state, player_index)], dtype=np.float32)

def player_to_dict(player: Player, suffix: str) -> dict:
    all_player_cards = get_all_player_cards(player)
    return {
        "average_treasure_value" + suffix: get_average_treasure_value_per_card(player),
        "num_vp" + suffix: get_total_player_vp(player),

        "num_victory_cards_owned" + suffix: (
                all_player_cards[card_name_to_card("estate")]
                + all_player_cards[card_name_to_card("duchy")]
                + all_player_cards[card_name_to_card("province")]
                + all_player_cards[card_name_to_card("curse")]),

        "num_copper_owned" + suffix: all_player_cards[card_name_to_card("copper")],
        "num_silver_owned" + suffix: all_player_cards[card_name_to_card("silver")],
        "num_gold_owned" + suffix: all_player_cards[card_name_to_card("gold")],
        "num_smithy_owned" + suffix: all_player_cards[card_name_to_card("smithy")],
        "num_laboratory_owned" + suffix: all_player_cards[card_name_to_card("laboratory")],
        "num_village_owned" + suffix: all_player_cards[card_name_to_card("village")],
        "num_festival_owned" + suffix: all_player_cards[card_name_to_card("festival")],
        "num_market_owned" + suffix: all_player_cards[card_name_to_card("market")],

        "zero_copper_owned" + suffix: all_player_cards[card_name_to_card("copper")] == 0,
        "zero_silver_owned" + suffix: all_player_cards[card_name_to_card("silver")] == 0,
        "zero_gold_owned" + suffix: all_player_cards[card_name_to_card("gold")] == 0,
        "zero_smithy_owned" + suffix: all_player_cards[card_name_to_card("smithy")] == 0,
        "zero_laboratory_owned" + suffix: all_player_cards[card_name_to_card("laboratory")] == 0,
        "zero_village_owned" + suffix: all_player_cards[card_name_to_card("village")] == 0,
        "zero_festival_owned" + suffix: all_player_cards[card_name_to_card("festival")] == 0,
        "zero_market_owned" + suffix: all_player_cards[card_name_to_card("market")] == 0,

        "one_copper_owned" + suffix: all_player_cards[card_name_to_card("copper")] == 1,
        "one_silver_owned" + suffix: all_player_cards[card_name_to_card("silver")] == 1,
        "one_gold_owned" + suffix: all_player_cards[card_name_to_card("gold")] == 1,
        "one_smithy_owned" + suffix: all_player_cards[card_name_to_card("smithy")] == 1,
        "one_laboratory_owned" + suffix: all_player_cards[card_name_to_card("laboratory")] == 1,
        "one_village_owned" + suffix: all_player_cards[card_name_to_card("village")] == 1,
        "one_festival_owned" + suffix: all_player_cards[card_name_to_card("festival")] == 1,
        "one_market_owned" + suffix: all_player_cards[card_name_to_card("market")] == 1,

        "two_copper_owned" + suffix: all_player_cards[card_name_to_card("copper")] == 2,
        "two_silver_owned" + suffix: all_player_cards[card_name_to_card("silver")] == 2,
        "two_gold_owned" + suffix: all_player_cards[card_name_to_card("gold")] == 2,
        "two_smithy_owned" + suffix: all_player_cards[card_name_to_card("smithy")] == 2,
        "two_laboratory_owned" + suffix: all_player_cards[card_name_to_card("laboratory")] == 2,
        "two_village_owned" + suffix: all_player_cards[card_name_to_card("village")] == 2,
        "two_festival_owned" + suffix: all_player_cards[card_name_to_card("festival")] == 2,
        "two_market_owned" + suffix: all_player_cards[card_name_to_card("market")] == 2,
    }
