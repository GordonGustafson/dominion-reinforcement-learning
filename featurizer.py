from cards import *

import numpy as np
import pandas as pd


def game_outcome_to_reward(game_outcome: GameOutcome) -> float:
    if game_outcome == GAME_OUTCOME.WIN:
        return 1.0
    elif game_outcome == GAME_OUTCOME.LOSS:
        return -1.0
    elif game_outcome == GAME_OUTCOME.DRAW:
        return 0.0
    else:
        assert False

def game_history_to_df(state_action_pairs: List[StateActionPair],
                       game_outcome_for_player: List[GameOutcome]):
    selected_game_states = (sap.possible_actions[sap.selected_action].game_state
                            for sap in state_action_pairs)
    game_state_df = pd.concat([game_state_to_df(gs) for gs in selected_game_states], axis="index", ignore_index=True)


    reward = game_outcome_to_reward(game_outcome_for_player)
    # Using gamma=1 for now.
    reward_df = pd.DataFrame({"reward": [reward] * len(state_action_pairs)})

    return pd.concat([game_state_df, reward_df], axis="columns")
    

    # result = np.stack(selected_dfs, axis=0)
    # result = np.concatenate((result, reward_df), axis=1)
    # return result


def game_state_to_df(game_state: GameState):
    # TODO: make this reflect the appropriate player's perspective.
    player_0_vp_lead = (get_total_player_vp(game_state.players[0])
                        - get_total_player_vp(game_state.players[1]))
    num_provinces = num_copies_of_card(game_state.supply, "province")

    non_player_state_df = pd.DataFrame({"player_1_vp_lead": [player_0_vp_lead],
                                            "num_provinces_remaining": [num_provinces]})

    player_dfs = [player_to_df(p).add_suffix(f"_player_{i+1}") for i, p in enumerate(game_state.players)]
    return pd.concat([non_player_state_df, *player_dfs], axis="columns")



def player_to_df(player: Player) -> pd.DataFrame:
    # num_cards = len(get_all_player_cards(player))
    return pd.DataFrame({"average_treasure_value": [get_average_treasure_value_per_card(player)]})
