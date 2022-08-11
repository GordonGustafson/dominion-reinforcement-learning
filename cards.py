import numpy as np
import pandas as pd

from typing import NamedTuple, Tuple, List


CardCounts = np.ndarray

_GameStateBase = NamedTuple("GameState", [
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

class GameState(_GameStateBase):
    def __eq__(self, other):
        return (self.cleanup_phase == other.cleanup_phase
                and card_counts_equal(self.hand, other.hand)
                and card_counts_equal(self.deck, other.deck)
                and card_counts_equal(self.discard_pile, other.discard_pile))


Action = NamedTuple("Action", [
    ("game_state", GameState),
    ("description", str),
])

CARD_DEFS = pd.DataFrame(columns=["cost", "victory_points", "money_produced", "name"], data=[
    [0, 0, 1, "copper"],
    [3, 0, 2, "silver"],
    [6, 0, 3, "gold"],
    [2, 1, 0, "estate"],
    [5, 3, 0, "duchy"],
    [8, 6, 0, "province"],
])

NUM_CARDS_DEFINED = CARD_DEFS.shape[0]

def card_counts_equal(lhs: CardCounts, rhs: CardCounts) -> bool:
    return np.array_equal(lhs, rhs)

def treasure_total(card_counts: CardCounts) -> int:
    return CARD_DEFS.money_produced.dot(card_counts)

def vp_total(card_counts: CardCounts) -> int:
    return CARD_DEFS.victory_points.dot(card_counts)

def empty_card_counts():
    return np.zeros(shape=(NUM_CARDS_DEFINED,), dtype=np.int32)

def num_cards(card_counts: CardCounts) -> int:
    return np.sum(card_counts)

def add_card_counts(card_counts_lhs: CardCounts, card_counts_rhs: CardCounts) -> CardCounts:
    return card_counts_lhs + card_counts_rhs

def add_card(card_counts: CardCounts, card_index: int) -> CardCounts:
    card_counts_copy = card_counts.copy()
    card_counts_copy[card_index] += 1
    return card_counts_copy

def remove_card(card_counts: CardCounts, card_index: int) -> CardCounts:
    if card_counts[card_index] < 1:
        raise ValueError("Can't remove card index {card_index} from card counts {card_counts}")

    card_counts_copy = card_counts.copy()
    card_counts_copy[card_index] -= 1
    return card_counts_copy

def draw_card(game_state: GameState):
    if num_cards(game_state.deck) == 0 and num_cards(game_state.discard_pile) == 0:
        return game_state

    if num_cards(game_state.deck) == 0:
        game_state = game_state._replace(deck=game_state.discard_pile,
                                         discard_pile=empty_card_counts())

    deck = game_state.deck
    card_drawn = np.random.choice(a=NUM_CARDS_DEFINED, size=1, replace=False, p=deck/num_cards(deck))
    game_state = game_state._replace(hand=add_card(game_state.hand, card_drawn),
                                    deck=remove_card(game_state.deck, card_drawn))

    return game_state

# TODO: make this return a set? Will need to stop using List in GameState
def buy_phase_options(buy_game_state: GameState) -> List[Action]:
    total_money_for_turn = treasure_total(buy_game_state.hand)

    # TODO: only allow buying from non-empty buy piles
    buyable_card_indices = CARD_DEFS.index[CARD_DEFS['cost'] <= total_money_for_turn].to_list()

    cleanup_game_state = buy_game_state._replace(cleanup_phase=True)
    buy_nothing = Action(game_state=cleanup_game_state, description="buy nothing")
    actions = [buy_nothing]
    for buyable_card_index in buyable_card_indices:
        game_state = cleanup_game_state._replace(
            discard_pile=add_card(buy_game_state.discard_pile, buyable_card_index))
        card_name = CARD_DEFS["name"][buyable_card_index]
        actions.append(Action(game_state, f"buy {card_name}"))

    return actions

def do_cleanup_phase_if_set(game_state: GameState) -> GameState:
    if not game_state.cleanup_phase:
        return game_state

    # Discard your hand
    game_state = game_state._replace(
        cleanup_phase=False,
        hand=empty_card_counts(),
        discard_pile=add_card_counts(game_state.hand, game_state.discard_pile),
    )

    # draw 5 cards
    for _ in range(5):
        game_state = draw_card(game_state)

    return game_state

def game_state_to_features(game_state: GameState):
    pass
    # total_victory_points =
    # total_money_for_turn =
