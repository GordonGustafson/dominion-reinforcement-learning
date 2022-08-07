import pandas as pd

from typing import NamedTuple, List, Set, Union


CardCounts = List[int]

GameState = NamedTuple("GameState", [
    ("cleanup_phase", bool),
    ("hand", CardCounts),
    ("deck", CardCounts),
    ("discard_pile", CardCounts),

    # ("buy_piles", CardCounts),

    # For later:
    # actions
    # buys
    # num cards in opponents hand: Militia, Council Room
    # opponent card counts?
])

CARD_DEFS = pd.DataFrame(columns=["cost", "victory_points", "money_produced", "name"], data=[
    [0, 0, 1, "copper"],
    [3, 0, 2, "silver"],
    [6, 0, 3, "gold"],
    [2, 1, 0, "estate"],
    [5, 3, 0, "duchy"],
    [8, 6, 0, "province"],
])

def treasure_total(card_counts: CardCounts) -> int:
    return CARD_DEFS.money_produced.dot(card_counts)

def vp_total(card_counts: CardCounts) -> int:
    return CARD_DEFS.victory_points.dot(card_counts)

# TODO: use numpy for CardCounts?
def add_card(card_counts: CardCounts, card_index: int) -> CardCounts:
    card_counts_copy = card_counts[:]
    card_counts_copy[card_index] += 1
    return card_counts_copy

def add_card_by_name(card_counts: CardCounts, card_name: str) -> CardCounts:
    card_index = CARD_DEFS.index[CARD_DEFS['name'] == card_name].item()
    return add_card(card_counts, card_index)

# TODO: make this return a set? Will need to stop using List in GameState
def buy_phase_options(buy_game_state: GameState) -> Set[GameState]:
    total_money_for_turn = treasure_total(buy_game_state.hand)

    # TODO: only allow buying from non-empty buy piles
    buyable_card_indices = CARD_DEFS.index[CARD_DEFS['cost'] <= total_money_for_turn].to_list()

    cleanup_game_state = buy_game_state._replace(cleanup_phase=True)
    # buy nothing case
    results = [cleanup_game_state]
    for buyable_card_index in buyable_card_indices:
        results.append(cleanup_game_state._replace(
            discard_pile=add_card(buy_game_state.discard_pile, buyable_card_index)))

    return results

def game_state_to_features(game_state: GameState):
    pass
    # total_victory_points =
    # total_money_for_turn =

def dict_to_card_counts(card_names_dict):
    return [card_names_dict.get(card_name, 0) for card_name in CARD_DEFS['name'].to_list()]


# Improvements for later:
# - make card counts available to ML agent
# - make buy pile counts available to ML agent

# lookup
# query
# c.index[c['name'] == 'silver'].item()
