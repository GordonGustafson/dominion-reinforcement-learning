import random

from multiset import Multiset
from typing import NamedTuple, Tuple, List, Dict


# TODO: make this a proper Python enum
class EFFECT_NAME:
    DRAW_CARDS = "draw_cards"
    PRODUCE_MONEY = "produce_money"
    VP = "vp"

Effect = NamedTuple("Effect", [
    ("name", EFFECT_NAME),
    ("value", int),
])

Card = NamedTuple("Card", [
    ("name", str),
    ("cost", int),
    ("action_effects", Tuple[Effect]),
    ("treasure_effects", Tuple[Effect]),
    ("vp_effects", Tuple[Effect]),
])

def make_card(name, cost, action_effects=(), treasure_effects=(), vp_effects=()):
    return Card(name, cost, action_effects, treasure_effects, vp_effects)

CardCounts = Multiset

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

# TODO: make this a proper Python enum
class TURN_PHASES:
    ACTION = "ACTION"
    BUY = "BUY"
    CLEANUP = "CLEANUP"

_GameStateBase = NamedTuple("GameState", [
    ("players", List[Player]),
    ("current_player_index", int),
    ("supply", CardCounts),
    # TODO: proper type annotation for this
    ("turn_phase", str),
])

class GameState(_GameStateBase):
    def __eq__(self, other):
        # The default implementation uses ==, which doesn't work for numpy arrays
        return (self.players == other.players
                and self.current_player_index == other.current_player_index
                and card_counts_equal(self.supply, other.supply)
                and self.turn_phase == other.turn_phase)

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

# TODO: should we use the same "money_produced" field for both treasures and actions?
CARD_DEFS = {
    "copper":   make_card(name="copper",   cost=0, treasure_effects=(Effect(EFFECT_NAME.PRODUCE_MONEY, 1),)),
    "silver":   make_card(name="silver",   cost=3, treasure_effects=(Effect(EFFECT_NAME.PRODUCE_MONEY, 2),)),
    "gold":     make_card(name="gold",     cost=6, treasure_effects=(Effect(EFFECT_NAME.PRODUCE_MONEY, 3),)),

    "estate":   make_card(name="estate",   cost=2, vp_effects=(Effect(EFFECT_NAME.VP, 1),)),
    "duchy":    make_card(name="duchy",    cost=5, vp_effects=(Effect(EFFECT_NAME.VP, 3),)),
    "province": make_card(name="province", cost=8, vp_effects=(Effect(EFFECT_NAME.VP, 6),)),

    "curse":   make_card(name="curse",   cost=0, vp_effects=(Effect(EFFECT_NAME.VP, -1),)),

    # +cards
    # {"name": "Smithy",       "cost": 4, EFFECT_NAME.DRAW_CARDS: 3,}

    # +actions
    # {"name": "Laboratory",   "cost": 5, "type": "action", EFFECT_NAME.DRAW_CARDS: 2, "actions": 1,}
    # {"name": "Village",      "cost": 3, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 2,}

    # trashing cards
    # {"name": "Chapel",       "cost": 2, "type": "action", "trash_up_to_X_cards_from_your_hand": 4,

    # gaining cards
    # {"name": "Workshop",     "cost": 3, "type": "action", @"gain_a_card_costing_up_to_4": 1
    # {"name": "Remodel",      "cost": 4, "type": "action", @"trash a card from your hand. gain a card costing up to 2 more than it"
    # {"name": "Mine",         "cost": 5, "type": "action", @"you may trash a treasure from your hand. gain a treasure to your hand costing up to $3 more than it"

    # +buys
    # {"name": "Festival",     "cost": 5, "type": "action", "actions": 2, @", +1 buy, +2$"
    # {"name": "Market",       "cost": 5, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, @"+1$ +1 buy"
    # {"name": "Council Room", "cost": 5, "type": "action", EFFECT_NAME.DRAW_CARDS: 4, @"+1 buy, each other player drawns a card"

    # simple draw/discard effects
    # {"name": "Cellar",       "cost": 2, "type": "action", "actions": 1, "discard_any_number_then_draw_that_many": 1,
    # {"name": "Harbinger",    "cost": 3, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, "put_any_card_from_discard_pile_onto_deck": 1,
    # {"name": "Poacher",      "cost": 4, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, @"+1$, discard a card per empty supply pile"

    # interacting with other cards
    # {"name": "Moneylender",  "cost": 4, "type": "action", @"you may trash a copper from your hand for +3$"
    # {"name": "Merchant",     "cost": 3, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, "the_first_time_you_play_a_silver_this_turn_+1_money": 1,

    # VP cards
    # {"name": "Gardens",      "cost": 4, "type": "victory", @"worth 1 vp per 10 cards you have (rounded down)"

    # attacks
    # {"name": "Witch",        "cost": 5, "type": "action", EFFECT_NAME.DRAW_CARDS: 2, @"each other player gains a curse"
    # {"name": "Bandit",       "cost": 5, "type": "action", @"gain a gold. each other player reveals the top 2 cards of their deck, trashes a revealed treasure other than copper, and discards the rest"
    # {"name": "Militia",      "cost": 4, "type": "action", @"+2$ each other player discards down to 3 cards in hand"
    # {"name": "Bureaucrat",   "cost": 4, "type": "action", @"gain a silver onto your deck. each other player reveals a victory card from their hand it puts it onto their deck (or reveals a hand with no victory cards)"
    # {"name": "Moat",         "cost": 2, "type": "action", EFFECT_NAME.DRAW_CARDS: 2, "moat_effect": 1,

    # new types of game state or choices
    # {"name": "Sentry",       "cost": 5, "type": "action", "actions": 1, @"+1 card . Look at the top 2 cards of your deck. Trash and/or discard any number of them, put the rest back on top in any order"
    # {"name": "Throne Room",  "cost": 4, "type": "action", @"you may play an action card from your hand twice"
    # {"name": "Vassal",       "cost": 3, "type": "action", "money_produced": 2, "Discard_the_top_card_of_your_deck_if_it's_an_action_card,_you_may_play_it": 1,
    # {"name": "Artisan",      "cost": 6, "type": "action", @"gain a card to your hand costing up to $5. put a card from your hand onto your deck"
    # {"name": "Library",      "cost": 5, "type": "action", @"draw until you have 7 cards in hand, skipping any action cards you choose to. Set those aside, discarding them afterwards"
}


def empty_card_counts():
    return Multiset()

def num_cards(card_counts: CardCounts) -> int:
    return sum(card_counts.values())

def card_counts_equal(lhs: CardCounts, rhs: CardCounts) -> bool:
    # Multiset implements == properly
    return lhs == rhs

def add_card(card_counts: CardCounts, card: Card) -> CardCounts:
    card_counts_copy = card_counts.copy()
    if card in card_counts_copy:
        card_counts_copy[card] += 1
    else:
        card_counts_copy[card] = 1
    return card_counts_copy

def remove_card(card_counts: CardCounts, card: Card) -> CardCounts:
    if card not in card_counts or card_counts[card] < 1:
        raise ValueError("Can't remove card {card} from card counts {card_counts}")

    card_counts_copy = card_counts.copy()
    card_counts_copy[card] -= 1
    return card_counts_copy

def dict_to_card_counts(card_names_dict: Dict[str, int]) -> CardCounts:
    result = empty_card_counts()
    for card_name, card_occurrences in card_names_dict.items():
        result[CARD_DEFS[card_name]] = card_occurrences
    return result


def treasure_total(card_counts: CardCounts) -> int:
    total = 0
    for card, card_occurrences in card_counts.items():
    # HACK this assumes all treasures only have one treasure effect
        if len(card.treasure_effects) > 0:
            total += card_occurrences * card.treasure_effects[0].value

    return total


def vp_total(card_counts: CardCounts) -> int:
    total = 0
    for card, card_occurrences in card_counts.items():
    # HACK this assumes all victory cards only have one vp effect
        if len(card.vp_effects) > 0:
            total += card_occurrences * card.vp_effects[0].value

    return total

def add_card_counts(card_counts_lhs: CardCounts, card_counts_rhs: CardCounts) -> CardCounts:
    return card_counts_lhs + card_counts_rhs

def total_player_vp(player: Player) -> int:
    return vp_total(player.hand) + vp_total(player.deck) + vp_total(player.discard_pile)


################################################################################
#                                                  Operations Using Card Names #
################################################################################

def card_name_to_card(card_name: str) -> Card:
    return CARD_DEFS[card_name]

def add_card_by_name(card_counts: CardCounts, card_name: str) -> CardCounts:
    card = card_name_to_card(card_name)
    return add_card(card_counts, card)

def remove_card_by_name(card_counts: CardCounts, card_name: str) -> CardCounts:
    card = card_name_to_card(card_name)
    return remove_card(card_counts, card)

def num_copies_of_card(card_counts: CardCounts, card_name: str) -> int:
    card = card_name_to_card(card_name)
    return card_counts[card]

def card_counts_to_dict(card_counts: CardCounts) -> Dict[str, int]:
    return {card.name: num_occurrences
            for card, num_occurrences in card_counts.items()
            if num_occurrences > 0}

################################################################################
#                                                                      Options #
################################################################################

# TODO: make this return a set? Will need to stop using List in GameState
def buy_phase_options(buy_game_state: GameState) -> List[Action]:
    player = buy_game_state.current_player()
    total_money_for_turn = treasure_total(player.hand)

    supply = buy_game_state.supply
    buyable_cards = [card for card in supply.distinct_elements()
                     if supply[card] > 0
                     and card.cost <= total_money_for_turn]

    cleanup_game_state = buy_game_state._replace(turn_phase=TURN_PHASES.CLEANUP)
    buy_nothing = Action(game_state=cleanup_game_state, description="buy nothing")
    actions = [buy_nothing]
    for buyable_card in buyable_cards:
        game_state = (cleanup_game_state
                      ._replace(supply=remove_card(supply, buyable_card))
                      .replace_current_player_kwargs(discard_pile=add_card(player.discard_pile,
                                                                           buyable_card)))

        actions.append(Action(game_state, f"buy {buyable_card.name}"))

    return actions

def draw_card(player: Player) -> Player:
    if num_cards(player.deck) == 0 and num_cards(player.discard_pile) == 0:
        return player

    if num_cards(player.deck) == 0:
        player = player._replace(deck=player.discard_pile,
                                 discard_pile=empty_card_counts())

    deck = player.deck
    unique_cards, frequencies = zip(*deck.items())

    card_drawn = random.choices(unique_cards, weights=frequencies, k=1)[0]
    player = player._replace(hand=add_card(player.hand, card_drawn),
                             deck=remove_card(deck, card_drawn))

    return player

def do_cleanup_phase(game_state: GameState) -> GameState:
    assert game_state.turn_phase == TURN_PHASES.CLEANUP

    # Discard your hand
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

    game_state = game_state._replace(turn_phase=TURN_PHASES.ACTION)

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
                     turn_phase=TURN_PHASES.ACTION)

def game_completed(game_state: GameState) -> bool:
    # distinguish between an card that has been fully bought up and a card that wasn't in the game
    # HACK for now assumes all cards defined are in the game
    original_non_empty_piles = len(CARD_DEFS)
    current_non_empty_piles = len([True for card in CARD_DEFS if game_state.supply.get(card, 0) == 0])
    num_empty_piles = original_non_empty_piles - current_non_empty_piles
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

        game_state = do_cleanup_phase(game_state)

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
