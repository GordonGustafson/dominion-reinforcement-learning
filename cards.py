import numpy as np
import pandas as pd

from typing import NamedTuple, Tuple, List, Dict


CardCounts = np.ndarray

_PlayerBase = NamedTuple("Player", [
    ("hand", CardCounts),
    ("deck", CardCounts),
    ("discard_pile", CardCounts),
])

class Player(_PlayerBase):
    def __eq__(self, other):
        # The default implementation uses ==, which doesn't work for numpy arrays
        return (card_counts_equal(self.hand, other.hand)
                and card_counts_equal(self.deck, other.deck)
                and card_counts_equal(self.discard_pile, other.discard_pile))

    # I think this is necessary because NamedTuple overrides __ne__ in some fancy way???
    def __ne__(self, other):
        return not (self == other)

_GameStateBase = NamedTuple("GameState", [
    ("players", List[Player]),
    ("current_player_index", int),
    ("supply", CardCounts),
    ("cleanup_phase", bool),
])

class GameState(_GameStateBase):
    def __eq__(self, other):
        # The default implementation uses ==, which doesn't work for numpy arrays
        return (self.players == other.players
                and self.current_player_index == other.current_player_index
                and card_counts_equal(self.supply, other.supply)
                and self.cleanup_phase == other.cleanup_phase)

    def __ne__(self, other):
        return not (self == other)

    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    # These replace_current_player... methods are syntactic sugar for making other code clearer
    def replace_current_player(self, player):
        players_copy = self.players[:]
        players_copy[self.current_player_index] = player
        return self._replace(players=players_copy)

    def replace_current_player_kwargs(self, **kwargs):
        return self.replace_current_player(self.current_player()._replace(**kwargs))

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

def card_name_to_index(card_name: str) -> int:
    return CARD_DEFS.index[CARD_DEFS['name'] == card_name].item()

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

def add_card_by_name(card_counts: CardCounts, card_name: str) -> CardCounts:
    card_index = card_name_to_index(card_name)
    return add_card(card_counts, card_index)

def remove_card_by_name(card_counts: CardCounts, card_name: str) -> CardCounts:
    card_index = card_name_to_index(card_name)
    return remove_card(card_counts, card_index)

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


def total_player_vp(player: Player) -> int:
    return vp_total(player.hand) + vp_total(player.deck) + vp_total(player.discard_pile)

# TODO: make this return a set? Will need to stop using List in GameState
def buy_phase_options(buy_game_state: GameState) -> List[Action]:
    player = buy_game_state.current_player()
    total_money_for_turn = treasure_total(player.hand)

    buyable_card_indices = CARD_DEFS.index[(CARD_DEFS['cost'] <= total_money_for_turn)
                                           & (buy_game_state.supply > 0)].to_list()

    cleanup_game_state = buy_game_state._replace(cleanup_phase=True)
    buy_nothing = Action(game_state=cleanup_game_state, description="buy nothing")
    actions = [buy_nothing]
    for buyable_card_index in buyable_card_indices:
        game_state = (cleanup_game_state
                      ._replace(supply=remove_card(cleanup_game_state.supply, buyable_card_index))
                      .replace_current_player_kwargs(discard_pile=add_card(player.discard_pile,
                                                                           buyable_card_index)))

        card_name = CARD_DEFS["name"][buyable_card_index]
        actions.append(Action(game_state, f"buy {card_name}"))

    return actions

def draw_card(player: Player) -> Player:
    if num_cards(player.deck) == 0 and num_cards(player.discard_pile) == 0:
        return player

    if num_cards(player.deck) == 0:
        player = player._replace(deck=player.discard_pile,
                                 discard_pile=empty_card_counts())

    deck = player.deck
    card_drawn = np.random.choice(a=NUM_CARDS_DEFINED, size=1, replace=False, p=deck/num_cards(deck))
    player = player._replace(hand=add_card(player.hand, card_drawn),
                             deck=remove_card(deck, card_drawn))

    return player

def do_cleanup_phase_if_set(game_state: GameState) -> GameState:
    if not game_state.cleanup_phase:
        return game_state

    # Discard your hand
    game_state = game_state._replace(cleanup_phase=False)
    game_state = game_state.replace_current_player_kwargs(
        hand=empty_card_counts(),
        discard_pile=add_card_counts(game_state.current_player().hand,
                                     game_state.current_player().discard_pile),
    )

    # draw 5 cards
    player = game_state.current_player()
    for _ in range(5):
        player = draw_card(player)
    game_state = game_state.replace_current_player(player)

    # Move to next player
    index = game_state.current_player_index + 1
    num_players = len(game_state.players)
    if index >= num_players:
        index = index % num_players
    game_state = game_state._replace(current_player_index=index)

    return game_state

def initial_player_state() -> Player:
    player = Player(hand=dict_to_card_counts({}),
                    deck=dict_to_card_counts({"estate": 3, "copper": 7}),
                    discard_pile=dict_to_card_counts({}))

    for _ in range(5):
        player = draw_card(player)

    return player

def initial_base_card_counts(num_players: int) -> Dict[str, int]:
    # Rulebook doesn't include rules for 1 player games so we can put whatever
    # we want here.
    if num_players == 1:
        return {"copper": 53,
                "silver": 40,
                "gold": 30,
                "curse": 10,
                "estate": 4,
                "duchy": 4,
                "province": 4}
    if num_players == 2:
        return {"copper": 46,
                "silver": 40,
                "gold": 30,
                "curse": 10,
                "estate": 8,
                "duchy": 8,
                "province": 8}
    elif num_players == 3:
        return {"copper": 39,
                "silver": 40,
                "gold": 30,
                "curse": 20,
                "estate": 12,
                "duchy": 12,
                "province": 12}
    elif num_players == 4:
        return {"copper": 32,
                "silver": 40,
                "gold": 30,
                "curse": 30,
                "estate": 12,
                "duchy": 12,
                "province": 12}
    else:
        assert False, f"Invalid number of players: {num_players}"

def initial_game_state(num_players: int) -> GameState:
    return GameState(players=[initial_player_state() for _ in range(num_players)],
                     current_player_index=0,
                     supply=dict_to_card_counts(initial_base_card_counts(num_players)),
                     cleanup_phase=False)

def game_completed(game_state: GameState) -> bool:
    num_empty_piles = np.sum(game_state.supply == 0)
    return (num_empty_piles >= 3
            or num_copies_of_card(game_state.supply, "province") == 0)

def game_flow(num_players: int, option_choosers: List):
    game_state = initial_game_state(num_players)

    while not game_completed(game_state):
        possible_actions = buy_phase_options(game_state)

        selected_action_index = option_choosers[game_state.current_player_index](game_state, possible_actions)
        selected_action = possible_actions[selected_action_index]
        game_state = selected_action.game_state
        print(selected_action.description)

        game_state = do_cleanup_phase_if_set(game_state)

    for i, player in enumerate(game_state.players):
        print(f"player {i} score: {total_player_vp(player)}")


def user_option_chooser(game_state: GameState, actions: List[Action]) -> int:
    print(f"hand: {card_counts_to_dict(game_state.current_player().hand)}")
    for i, action in enumerate(actions):
        print(f"{i}: {action.description}")

    selected_option = int(input())
    while selected_option >= len(actions) or selected_option < 0:
        print("index out of bounds, try again")
        selected_option = int(input())

    return selected_option
