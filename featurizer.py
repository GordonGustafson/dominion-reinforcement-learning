from cards import *

import numpy as np


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
                       game_outcome_for_player: List[GameOutcome]):
    selected_game_states = (sap.possible_actions[sap.selected_action].game_state
                            for sap in state_action_pairs)
    selected_vectors = [game_state_to_vector(gs) for gs in selected_game_states]
    reward = game_outcome_to_reward(game_outcome_for_player)
    # Using gamma=1 for now.
    reward_vector = np.full(shape=(len(state_action_pairs), 1), fill_value=reward, dtype=np.float32)

    result = np.stack(selected_vectors, axis=0)
    result = np.concatenate((result, reward_vector), axis=1)
    return result


def game_state_to_vector(game_state: GameState):
    # TODO: make this reflect the appropriate player's perspective.
    player_0_vp_lead = (get_total_player_vp(game_state.players[0])
                        - get_total_player_vp(game_state.players[1]))
    num_provinces = num_copies_of_card(game_state.supply, "province")

    non_player_state_vector = np.array([player_0_vp_lead, num_provinces], dtype=np.float32)

    player_vectors = [player_to_vector(p) for p in game_state.players]
    return np.concatenate([non_player_state_vector] + player_vectors, axis=0)

def player_to_vector(player: Player) -> np.ndarray:
    average_treasure_value = get_average_treasure_value_per_card(player)
    # num_cards = len(get_all_player_cards(player))

    return np.array([average_treasure_value])
