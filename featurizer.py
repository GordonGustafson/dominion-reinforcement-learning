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
    player_vectors = [player_to_vector(p) for p in game_state.players]
    return np.concatenate(player_vectors, axis=0)

def player_to_vector(player: Player) -> np.ndarray:
    total_vp = get_total_player_vp(player)
    average_treasure_value = get_average_treasure_value_per_card(player)
    num_cards = len(get_all_player_cards(player))

    return np.array([float(total_vp), average_treasure_value, num_cards])
