import random

from multiset import Multiset
from typing import NamedTuple, Tuple, List, Dict, Set, Optional, Sequence

from enum import Enum


class EffectName(Enum):
    DRAW_CARDS = "draw_cards"
    PLUS_ACTIONS = "plus_actions"
    PLUS_BUYS = "plus_buys"
    GAIN_A_GOLD = "gain_a_gold"
    MAY_TRASH_A_CARD_FROM_YOUR_HAND = "may_trash_a_card_from_your_hand"
    GAIN_A_CARD_COSTING_UP_TO = "gain_a_card_costing_up_to"
    GAIN_A_TREASURE_TO_HAND_COSTING_UP_TO = "gain_a_treasure_to_hand_costing_up_to"
    TRASH_GAIN_A_CARD_COSTING_UP_TO_X_MORE = "trash_gain_a_card_costing_up_to_x_more"
    MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE = "may_trash_treasure_gain_treasure_to_hand_costing_up_to_x_more"
    DISCARD_ANY_NUMBER_THEN_DRAW_THAT_MANY = "discard_any_number_then_draw_that_many"
    MAY_PUT_ANY_CARD_FROM_DISCARD_PILE_ONTO_DECK = "may_put_any_card_from_discard_pile_onto_deck"
    MAY_TRASH_A_COPPER_TO_PRODUCE_MONEY = "may_trash_a_copper_to_produce_money"
    EACH_OTHER_PLAYER_DRAWS_A_CARD = "each_other_player_draws_a_card"
    EACH_OTHER_PLAYER_GAINS_A_CURSE = "each_other_player_gains_a_curse"
    EACH_OTHER_PLAYER_DISCARDS_DOWN_TO = "each_other_player_discards_down_to"
    EACH_OTHER_PLAYER_BANDIT_EFFECT = "each_other_player_bandit_effect"

    PRODUCE_MONEY = "produce_money"
    VP = "vp"

Effect = NamedTuple("Effect", [
    ("name", EffectName),
    ("value", int | None),
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

Player = NamedTuple("Player", [
    ("hand", CardCounts),
    ("deck", CardCounts),
    ("top_of_deck", Tuple[Card]),
    ("played_actions", CardCounts),
    ("discard_pile", CardCounts),
    ("name", str),
])

def make_player(hand=Multiset(),
                deck=Multiset(),
                top_of_deck=(),
                played_actions=Multiset(),
                discard_pile=Multiset(),
                name="unnamed player"):
    return Player(hand, deck, top_of_deck, played_actions, discard_pile, name)

class TurnPhase(Enum):
    ACTION = "ACTION"
    TREASURE = "TREASURE"
    FIRST_BUY = "FIRST_BUY"
    NON_FIRST_BUY = "NON_FIRST_BUY"
    CLEANUP = "CLEANUP"

_GameStateBase = NamedTuple("GameState", [
    ("players", List[Player]),
    ("first_player_index", int),
    ("current_player_index", int),
    ("max_turns_per_player", int),
    ("supply", CardCounts),
    ("turn_phase", TurnPhase),
    ("actions", int),
    ("buys", int),
    ("total_money", int),
    ("pending_effects", Tuple[Effect])
])

class GameState(_GameStateBase):
    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    def replace_player_by_index(self, index, player):
        players_copy = self.players[:]
        players_copy[index] = player
        return self._replace(players=players_copy)

    def replace_player_by_index_kwargs(self, index, **kwargs):
        return self.replace_player_by_index(index, self.players[index]._replace(**kwargs))

    def replace_current_player(self, player):
        return self.replace_player_by_index(self.current_player_index, player)

    def replace_current_player_kwargs(self, **kwargs):
        return self.replace_current_player(self.current_player()._replace(**kwargs))

    def prepend_effect(self, effect: Effect):
        new_pending_effects = (effect,) + self.pending_effects
        return self._replace(pending_effects=new_pending_effects)

def make_game_state(
        players,
        first_player_index=0,
        current_player_index=0,
        max_turns_per_player=0,
        supply=Multiset(),
        turn_phase=TurnPhase.ACTION,
        actions=1,
        buys=1,
        total_money=0,
        pending_effects=()):
    return GameState(players, first_player_index, current_player_index, max_turns_per_player, supply, turn_phase,
                     actions, buys, total_money, pending_effects)



class GameOutcome:
    WIN = "WIN"
    LOSS = "LOSS"
    DRAW = "DRAW"

# TODO: should we use the same "money_produced" field for both treasures and actions?
CARD_LIST = [
    make_card(name="copper", cost=0, treasure_effects=(Effect(EffectName.PRODUCE_MONEY, 1),)),
    make_card(name="silver", cost=3, treasure_effects=(Effect(EffectName.PRODUCE_MONEY, 2),)),
    make_card(name="gold", cost=6, treasure_effects=(Effect(EffectName.PRODUCE_MONEY, 3),)),

     make_card(name="estate", cost=2, vp_effects=(Effect(EffectName.VP, 1),)),
     make_card(name="duchy", cost=5, vp_effects=(Effect(EffectName.VP, 3),)),
     make_card(name="province", cost=8, vp_effects=(Effect(EffectName.VP, 6),)),

     make_card(name="curse", cost=0, vp_effects=(Effect(EffectName.VP, -1),)),

     # +cards
     make_card(name="smithy", cost=4, action_effects=(Effect(EffectName.DRAW_CARDS, 3),)),
 
     # +actions
     make_card(name="laboratory", cost=5, action_effects=(Effect(EffectName.DRAW_CARDS, 2), Effect(EffectName.PLUS_ACTIONS, 1))),
     make_card(name="village", cost=3, action_effects=(Effect(EffectName.DRAW_CARDS, 1), Effect(EffectName.PLUS_ACTIONS, 2))),

     # +buys
     make_card(name="festival", cost=5, action_effects=(Effect(EffectName.PLUS_ACTIONS, 2),
                                                        Effect(EffectName.PLUS_BUYS, 1),
                                                        Effect(EffectName.PRODUCE_MONEY, 2))),
     make_card(name="market", cost=5, action_effects=(Effect(EffectName.PLUS_ACTIONS, 1),
                                                      Effect(EffectName.PLUS_BUYS, 1),
                                                      Effect(EffectName.PRODUCE_MONEY, 1, ),
                                                      Effect(EffectName.DRAW_CARDS, 1))),
 
#    # trashing cards
     make_card(name="chapel", cost=2, action_effects=(Effect(EffectName.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None),
                                                      Effect(EffectName.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None),
                                                      Effect(EffectName.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None),
                                                      Effect(EffectName.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None))),

#    # gaining cards
     make_card(name="workshop", cost=3, action_effects=(Effect(EffectName.GAIN_A_CARD_COSTING_UP_TO, 4),)),
#    make_card(name="remodel", cost=4, action_effects=(Effect(EffectName.TRASH_GAIN_A_CARD_COSTING_UP_TO_X_MORE, 2),)),
#    make_card(name="mine", cost=5, action_effects=(Effect(EffectName.MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE, 3),)),
#
#    # simple draw/discard effects
#    make_card(name="cellar", cost=2, action_effects=(Effect(EffectName.PLUS_ACTIONS, 1),
#                                                     Effect(EffectName.DISCARD_ANY_NUMBER_THEN_DRAW_THAT_MANY, None))),
#    make_card(name="harbinger", cost=3, action_effects=(Effect(EffectName.PLUS_ACTIONS, 1),
#                                                        Effect(EffectName.DRAW_CARDS, 1),
#                                                        Effect(EffectName.MAY_PUT_ANY_CARD_FROM_DISCARD_PILE_ONTO_DECK, None))),
#    # {"name": "Poacher",      "cost": 4, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, @"+1$, discard a card per empty supply pile"
#
#    # interacting with other cards
#    make_card(name="moneylender", cost=4, action_effects=(Effect(EffectName.MAY_TRASH_A_COPPER_TO_PRODUCE_MONEY, 3),)),
#    # {"name": "Merchant",     "cost": 3, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, "the_first_time_you_play_a_silver_this_turn_+1_money": 1,
#
#    # VP cards
#    # {"name": "Gardens",      "cost": 4, "type": "victory", @"worth 1 vp per 10 cards you have (rounded down)"
#
#    # attacks
     make_card(name="council room", cost=5, action_effects=(Effect(EffectName.DRAW_CARDS, 4),
                                                            Effect(EffectName.PLUS_BUYS, 1),
                                                            Effect(EffectName.EACH_OTHER_PLAYER_DRAWS_A_CARD, None))),
     make_card(name="witch", cost=5, action_effects=(Effect(EffectName.DRAW_CARDS, 2),
                                                     Effect(EffectName.EACH_OTHER_PLAYER_GAINS_A_CURSE, None))),
#    make_card(name="militia", cost=4, action_effects=(Effect(EffectName.PRODUCE_MONEY, 2),
#                                                      Effect(EffectName.EACH_OTHER_PLAYER_DISCARDS_DOWN_TO, 3))),
#    make_card(name="bandit", cost=5, action_effects=(Effect(EffectName.GAIN_A_GOLD, None),
#                                                     Effect(EffectName.EACH_OTHER_PLAYER_BANDIT_EFFECT, None))),
# {"name": "Bureaucrat",   "cost": 4, "type": "action", @"gain a silver onto your deck. each other player reveals a victory card from their hand it puts it onto their deck (or reveals a hand with no victory cards)"
# {"name": "Moat",         "cost": 2, "type": "action", EFFECT_NAME.DRAW_CARDS: 2, "moat_effect": 1,

# new types of game state or choices
# {"name": "Sentry",       "cost": 5, "type": "action", "actions": 1, @"+1 card . Look at the top 2 cards of your deck. Trash and/or discard any number of them, put the rest back on top in any order"
# {"name": "Throne Room",  "cost": 4, "type": "action", @"you may play an action card from your hand twice"
# {"name": "Vassal",       "cost": 3, "type": "action", "money_produced": 2, "Discard_the_top_card_of_your_deck_if_it's_an_action_card,_you_may_play_it": 1,
# {"name": "Artisan",      "cost": 6, "type": "action", @"gain a card to your hand costing up to $5. put a card from your hand onto your deck"
# {"name": "Library",      "cost": 5, "type": "action", @"draw until you have 7 cards in hand, skipping any action cards you choose to. Set those aside, discarding them afterwards"
]

CARD_DICT = {card.name: card for card in CARD_LIST}


CardId = int
def card_id_to_card(card_id: CardId) -> Card:
    return CARD_LIST[card_id]

def card_to_card_id(card: Card) -> CardId:
    return CARD_LIST.index(card)

def action_cards_with_plus_n_actions(n: int) -> List[Card]:
    action_cards = [card for card in CARD_LIST if len(card.action_effects) > 0]

    if n == 0:
        return [card for card in action_cards
                if not any(effect.name == EffectName.PLUS_ACTIONS for effect in card.action_effects)]

    return [card for card in action_cards if Effect(EffectName.PLUS_ACTIONS, n) in card.action_effects]


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

def add_card_counts(card_counts_lhs: CardCounts, card_counts_rhs: CardCounts) -> CardCounts:
    return card_counts_lhs + card_counts_rhs

def is_treasure(card: Card) -> bool:
    return len(card.treasure_effects) > 0

def is_treasure_other_than_a_copper(card: Card) -> bool:
    return is_treasure(card) and card != card_name_to_card("copper")

def money_from_treasures(card_counts: CardCounts) -> int:
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

def get_all_player_cards(player: Player) -> CardCounts:
    all_cards = player.hand + player.deck + player.played_actions + player.discard_pile
    for card in player.top_of_deck:
        all_cards = add_card(all_cards, card)
    return all_cards

def get_total_player_vp(player: Player) -> int:
    return vp_total(get_all_player_cards(player))

def get_total_owned_treasure_value(player: Player) -> float:
    all_player_cards = get_all_player_cards(player)
    return money_from_treasures(all_player_cards)

def get_average_treasure_value_per_card(player: Player) -> float:
    all_player_cards = get_all_player_cards(player)
    player_treasure_total = money_from_treasures(all_player_cards)
    player_card_total = num_cards(all_player_cards)
    return player_treasure_total / player_card_total

def non_current_player_indices(game_state: GameState):
    result = list(range(len(game_state.players)))
    result.remove(game_state.current_player_index)
    return result

################################################################################
#                                                  Operations Using Card Names #
################################################################################

def card_name_to_card(card_name: str) -> Card:
    return CARD_DICT[card_name]

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

def dict_to_card_counts(card_names_dict: Dict[str, int]) -> CardCounts:
    result = empty_card_counts()
    for card_name, card_occurrences in card_names_dict.items():
        result[card_name_to_card(card_name)] = card_occurrences
    return result

def card_sequence_to_card_counts(cards: Sequence[Card]) -> CardCounts:
    card_counts = empty_card_counts()
    for card in cards:
        card_counts = add_card(card_counts, card)
    return card_counts

def unique_cards(card_counts: CardCounts) -> Set[Card]:
    # For some reason .key() didn't work, even though it's described here: https://pythonhosted.org/multiset/
    # maybe try updating my multiset package version?
    # use this hack for now.
    return set(card_counts)


def add_card_to_top_of_deck(player: Player, card: Card) -> Player:
    return player._replace(top_of_deck=(card,) + player.top_of_deck)

def take_top_card_off_of_deck(player: Player) -> Tuple[Player, Optional[Card]]:
    if len(player.top_of_deck) == 0 and num_cards(player.deck) == 0 and num_cards(player.discard_pile) == 0:
        return player, None

    if len(player.top_of_deck) > 0:
        card_taken = player.top_of_deck[0]
        player = player._replace(top_of_deck=player.top_of_deck[1:])
        return player, card_taken

    if len(player.top_of_deck) == 0 and num_cards(player.deck) == 0:
        player = player._replace(deck=player.discard_pile,
                                 discard_pile=empty_card_counts())

    deck = player.deck
    unique_cards, frequencies = zip(*deck.items())

    card_taken = random.choices(unique_cards, weights=frequencies, k=1)[0]
    player = player._replace(deck=remove_card(deck, card_taken))

    return player, card_taken

def draw_card(player: Player) -> Player:
    player, card_drawn = take_top_card_off_of_deck(player)
    if card_drawn is not None:
        return player._replace(hand=add_card(player.hand, card_drawn))
    else:
        return player

def discard_specific_card_current_player(game_state: GameState, card_discarded: Card) -> GameState:
    player = game_state.current_player()
    return game_state.replace_current_player_kwargs(
        hand=remove_card(player.hand, card_discarded),
        discard_pile=add_card(player.discard_pile, card_discarded))

def discard_specific_card_player_index(game_state: GameState, card_discarded: Card, index: int) -> GameState:
    player = game_state.players[index]
    return game_state.replace_player_by_index_kwargs(
        index=index,
        hand=remove_card(player.hand, card_discarded),
        discard_pile=add_card(player.discard_pile, card_discarded))

def move_specific_card_to_played_actions(game_state: GameState, action_played: Card) -> GameState:
    player = game_state.current_player()
    return game_state.replace_current_player_kwargs(
        hand=remove_card(player.hand, action_played),
        played_actions=add_card(player.played_actions, action_played))

def draw_cards_current_player(game_state: GameState, num_cards_to_draw: int) -> GameState:
    player = game_state.current_player()
    for _ in range(num_cards_to_draw):
        player = draw_card(player)
    return game_state.replace_current_player(player)

def gain_card_by_player_index(game_state: GameState, card: Card, index: int) -> GameState:
    if game_state.supply[card] <= 0:
        # print(f"Player index {index} not gaining {card.name} because its supply pile is empty")
        return game_state

    player_at_index = game_state.players[index]
    return (game_state
            ._replace(supply=remove_card(game_state.supply, card))
            .replace_player_by_index_kwargs(index=index,
                                            discard_pile=add_card(player_at_index.discard_pile, card)))

def gain_card_current_player(game_state: GameState, card: Card) -> GameState:
    return gain_card_by_player_index(game_state, card, game_state.current_player_index)

def gain_card_to_hand_current_player(game_state: GameState, card: Card) -> GameState:
    if game_state.supply[card] <= 0:
        # print(f"Current player not gaining {card.name} to hand because its supply pile is empty")
        return game_state

    return (game_state
            ._replace(supply=remove_card(game_state.supply, card))
            .replace_current_player_kwargs(hand=add_card(game_state.current_player().hand,
                                                         card)))

def cards_in_supply_costing_less_than(game_state, max_cost) -> List[Card]:
    supply = game_state.supply
    # The > 0 check is likely not needed now that we're using a Multiset
    return [card for card in supply.distinct_elements()
            if supply[card] > 0
            and card.cost <= max_cost]


def do_cleanup_phase(game_state: GameState) -> GameState:
    assert game_state.turn_phase == TurnPhase.CLEANUP

    # Discard your hand and any actions played
    new_discard_pile = add_card_counts(game_state.current_player().hand,
                                       game_state.current_player().discard_pile)
    new_discard_pile = add_card_counts(game_state.current_player().played_actions,
                                       new_discard_pile)
    game_state = game_state.replace_current_player_kwargs(
        hand=empty_card_counts(),
        played_actions=empty_card_counts(),
        discard_pile=new_discard_pile,
    )

    # draw 5 cards
    game_state = draw_cards_current_player(game_state, 5)

    # Move to next player
    index = game_state.current_player_index + 1
    num_players = len(game_state.players)
    if index >= num_players:
        game_state = game_state._replace(max_turns_per_player=game_state.max_turns_per_player+1)
        index = index % num_players
    game_state = game_state._replace(current_player_index=index)

    # Reset player state
    game_state = game_state._replace(turn_phase=TurnPhase.ACTION, actions=1, buys=1, total_money=0)

    return game_state

################################################################################
#                                                              Starting a Game #
################################################################################

def initial_player_state(name: str) -> Player:
    player = Player(name=name,
                    hand=dict_to_card_counts({}),
                    top_of_deck=(),
                    played_actions=dict_to_card_counts({}),
                    deck=dict_to_card_counts({"estate": 3, "copper": 7}),
                    discard_pile=dict_to_card_counts({}))

    for _ in range(5):
        player = draw_card(player)

    return player

def initial_supply_base_cards(num_players: int) -> Dict[str, int]:
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
        raise ValueError(f"Invalid number of players: {num_players}")

def initial_supply(num_players: int) -> Dict[str, int]:
    card_dict = initial_supply_base_cards(num_players)
    card_dict["smithy"] = 10
    card_dict["village"] = 10
    card_dict["laboratory"] = 10
    card_dict["festival"] = 10
    card_dict["market"] = 10
    card_dict["chapel"] = 10
    card_dict["workshop"] = 10
    # card_dict["remodel"] = 10
    # card_dict["mine"] = 10
    # card_dict["cellar"] = 10
    # card_dict["harbinger"] = 10
    # card_dict["moneylender"] = 10
    card_dict["council room"] = 10
    card_dict["witch"] = 10
    # card_dict["militia"] = 10
    # card_dict["bandit"] = 10
    return card_dict

def initial_game_state(player_names: List[str]) -> GameState:
    num_players = len(player_names)
    first_player_index = random.randrange(len(player_names))
    return GameState(players=[initial_player_state(name) for name in player_names],
                     first_player_index=first_player_index,
                     current_player_index=first_player_index,
                     max_turns_per_player=0,
                     pending_effects=(),
                     actions=1,
                     buys=1,
                     total_money=0,
                     supply=dict_to_card_counts(initial_supply(num_players)),
                     turn_phase=TurnPhase.ACTION)
