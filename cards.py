import numpy as np
import pandas as pd

from typing import NamedTuple, Tuple, List, Dict


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
        # The default implementation uses ==, which doesn't work for numpy arrays
        return (self.cleanup_phase == other.cleanup_phase
                and card_counts_equal(self.hand, other.hand)
                and card_counts_equal(self.deck, other.deck)
                and card_counts_equal(self.discard_pile, other.discard_pile))

    def __ne__(self, other):
        return not (self == other)

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

def empty_card_counts():
    return np.zeros(shape=(NUM_CARDS_DEFINED,), dtype=np.int32)

def num_cards(card_counts: CardCounts) -> int:
    return np.sum(card_counts)

def card_counts_equal(lhs: CardCounts, rhs: CardCounts) -> bool:
    return np.array_equal(lhs, rhs)

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


def card_name_to_index(card_name: str) -> int:
    return CARD_DEFS.index[CARD_DEFS['name'] == card_name].item()

def num_copies_of_card(card_counts: CardCounts, card_name: str) -> int:
    card_index = card_name_to_index(card_name)
    return card_counts[card_index]

def dict_to_card_counts(card_names_dict: Dict[str, int]) -> CardCounts:
    return np.array([card_names_dict.get(card_name, 0) for card_name in CARD_DEFS['name'].to_list()])

def card_counts_to_dict(card_counts: CardCounts) -> Dict[str, int]:
    return {card_name: num_copies_of_card(card_counts, card_name)
            for card_name in CARD_DEFS['name'].to_list()
            if num_copies_of_card(card_counts, card_name) > 0}


def treasure_total(card_counts: CardCounts) -> int:
    return CARD_DEFS.money_produced.dot(card_counts)

def vp_total(card_counts: CardCounts) -> int:
    return CARD_DEFS.victory_points.dot(card_counts)

def add_card_counts(card_counts_lhs: CardCounts, card_counts_rhs: CardCounts) -> CardCounts:
    return card_counts_lhs + card_counts_rhs


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

def initial_game_state() -> GameState:
    # TODO: account for copper and estates being taken into starting hands
    # return dict_to_card_counts({
    #     "copper": 60,
    #     "silver": 40,
    #     "gold": 30,
    #     "estate": 24,
    #     "duchy": 12,
    #     "province": 8,
    # })

    game_state = GameState(cleanup_phase=False,
                           hand=dict_to_card_counts({}),
                           deck=dict_to_card_counts({"estate": 3, "copper": 7}),
                           discard_pile=dict_to_card_counts({}))

    for _ in range(5):
        game_state = draw_card(game_state)

    return game_state

def is_last_turn(game_state: GameState) -> bool:
    # hack until we add supply piles
    num_provinces = (num_copies_of_card(game_state.hand, "province")
                     + num_copies_of_card(game_state.deck, "province")
                     + num_copies_of_card(game_state.discard_pile, "province"))
    return num_provinces >= 4

def game_flow(option_chooser):
    game_state = initial_game_state()

    while not is_last_turn(game_state):
        possible_actions = buy_phase_options(game_state)

        selected_action_index = option_chooser(game_state, possible_actions)
        selected_action = possible_actions[selected_action_index]
        game_state = selected_action.game_state
        print(selected_action.description)

        game_state = do_cleanup_phase_if_set(game_state)


def user_option_chooser(game_state: GameState, actions: List[Action]) -> int:
    print(f"hand: {card_counts_to_dict(game_state.hand)}")
    for i, action in enumerate(actions):
        print(f"{i}: {action.description}")

    return int(input())
