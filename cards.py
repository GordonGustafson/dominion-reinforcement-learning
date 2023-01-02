import random

from multiset import Multiset
from typing import NamedTuple, Tuple, List, Dict


# TODO: make this a proper Python enum
class EFFECT_NAME:
    DRAW_CARDS = "draw_cards"
    PLUS_ACTIONS = "plus_actions"
    PLUS_BUYS = "plus_buys"
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

# TODO: make this a proper Python enum
class TURN_PHASES:
    ACTION = "ACTION"
    TREASURE = "TREASURE"
    BUY = "BUY"
    CLEANUP = "CLEANUP"

_GameStateBase = NamedTuple("GameState", [
    ("players", List[Player]),
    ("current_player_index", int),
    ("max_turns_per_player", int),
    ("supply", CardCounts),
    # TODO: proper type annotation for this
    ("turn_phase", str),
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
        current_player_index=0,
        max_turns_per_player=0,
        supply=Multiset(),
        turn_phase=TURN_PHASES.ACTION,
        actions=1,
        buys=1,
        total_money=0,
        pending_effects=()):
    return GameState(players, current_player_index, max_turns_per_player, supply, turn_phase,
                     actions, buys, total_money, pending_effects)

Choice = NamedTuple("Choice", [
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

    "curse":    make_card(name="curse",    cost=0, vp_effects=(Effect(EFFECT_NAME.VP, -1),)),

    # +cards
    "smithy":    make_card(name="smithy",  cost=4, action_effects=(Effect(EFFECT_NAME.DRAW_CARDS, 3),)),

    # +actions
    "laboratory": make_card(name="laboratory", cost=5, action_effects=(Effect(EFFECT_NAME.DRAW_CARDS, 2), Effect(EFFECT_NAME.PLUS_ACTIONS, 1))),
    "village":    make_card(name="village",    cost=3, action_effects=(Effect(EFFECT_NAME.DRAW_CARDS, 1), Effect(EFFECT_NAME.PLUS_ACTIONS, 2))),

    # trashing cards
    "chapel":     make_card(name="chapel",     cost=2, action_effects=(Effect(EFFECT_NAME.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None),
                                                                       Effect(EFFECT_NAME.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None),
                                                                       Effect(EFFECT_NAME.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None),
                                                                       Effect(EFFECT_NAME.MAY_TRASH_A_CARD_FROM_YOUR_HAND, None))),

    # gaining cards
    "workshop":  make_card(name="workshop", cost=3, action_effects=(Effect(EFFECT_NAME.GAIN_A_CARD_COSTING_UP_TO, 4),)),
    "remodel":   make_card(name="remodel",  cost=4, action_effects=(Effect(EFFECT_NAME.TRASH_GAIN_A_CARD_COSTING_UP_TO_X_MORE, 2),)),
    "mine":      make_card(name="mine",     cost=5, action_effects=(Effect(EFFECT_NAME.MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE, 3),)),

    # +buys
    "festival":  make_card(name="festival", cost=5, action_effects=(Effect(EFFECT_NAME.PLUS_ACTIONS, 2),
                                                                    Effect(EFFECT_NAME.PLUS_BUYS, 1),
                                                                    Effect(EFFECT_NAME.PRODUCE_MONEY, 2))),
    "market":    make_card(name="market", cost=5, action_effects=(Effect(EFFECT_NAME.PLUS_ACTIONS, 1),
                                                                  Effect(EFFECT_NAME.PLUS_BUYS, 1),
                                                                  Effect(EFFECT_NAME.PRODUCE_MONEY, 1, ),
                                                                  Effect(EFFECT_NAME.DRAW_CARDS, 1))),

    # simple draw/discard effects
    "cellar":    make_card(name="cellar", cost=2, action_effects=(Effect(EFFECT_NAME.PLUS_ACTIONS, 1),
                                                                  Effect(EFFECT_NAME.DISCARD_ANY_NUMBER_THEN_DRAW_THAT_MANY, None))),
    "harbinger": make_card(name="harbinger", cost=3, action_effects=(Effect(EFFECT_NAME.PLUS_ACTIONS, 1),
                                                                     Effect(EFFECT_NAME.DRAW_CARDS, 1),
                                                                     Effect(EFFECT_NAME.MAY_PUT_ANY_CARD_FROM_DISCARD_PILE_ONTO_DECK, None))),
    # {"name": "Poacher",      "cost": 4, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, @"+1$, discard a card per empty supply pile"

    # interacting with other cards
    "moneylender": make_card(name="moneylender", cost=4, action_effects=(Effect(EFFECT_NAME.MAY_TRASH_A_COPPER_TO_PRODUCE_MONEY, 3),)),
    # {"name": "Merchant",     "cost": 3, "type": "action", EFFECT_NAME.DRAW_CARDS: 1, "actions": 1, "the_first_time_you_play_a_silver_this_turn_+1_money": 1,

    # VP cards
    # {"name": "Gardens",      "cost": 4, "type": "victory", @"worth 1 vp per 10 cards you have (rounded down)"

    # attacks
    "council room": make_card(name="council room", cost=5, action_effects=(Effect(EFFECT_NAME.DRAW_CARDS, 4),
                                                                           Effect(EFFECT_NAME.PLUS_BUYS, 1),
                                                                           Effect(EFFECT_NAME.EACH_OTHER_PLAYER_DRAWS_A_CARD, None))),
    "witch":        make_card(name="witch", cost=5, action_effects=(Effect(EFFECT_NAME.DRAW_CARDS, 2),
                                                                    Effect(EFFECT_NAME.EACH_OTHER_PLAYER_GAINS_A_CURSE, None))),

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

def add_card_counts(card_counts_lhs: CardCounts, card_counts_rhs: CardCounts) -> CardCounts:
    return card_counts_lhs + card_counts_rhs

def is_treasure(card: Card) -> bool:
    return len(card.treasure_effects) > 0

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

def total_player_vp(player: Player) -> int:
    return vp_total(player.hand) + vp_total(player.deck) + vp_total(player.discard_pile)

def average_treasure_value_per_card(player: Player) -> float:
    player_treasure_total = money_from_treasures(player.hand) + money_from_treasures(player.deck) + money_from_treasures(player.discard_pile)
    player_card_total = num_cards(player.hand) + num_cards(player.deck) + num_cards(player.discard_pile)
    return player_treasure_total / player_card_total

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

def dict_to_card_counts(card_names_dict: Dict[str, int]) -> CardCounts:
    result = empty_card_counts()
    for card_name, card_occurrences in card_names_dict.items():
        result[card_name_to_card(card_name)] = card_occurrences
    return result


################################################################################
#                                                                      Choices #
################################################################################

def action_phase_choices(action_game_state: GameState) -> List[Choice]:
    player = action_game_state.current_player()
    treasure_game_state = action_game_state._replace(turn_phase=TURN_PHASES.TREASURE)

    move_to_treasure_phase = Choice(game_state=treasure_game_state, description="move to treasure phase")
    choices = [move_to_treasure_phase]

    assert action_game_state.actions >= 0
    if action_game_state.actions == 0:
        return choices

    playable_actions = [card for card in player.hand
                        if len(card.action_effects) > 0]
    for action in playable_actions:
        # The action effects from this card happen before any other pending
        # effects, so we put them on the left
        pending_effects = action.action_effects + action_game_state.pending_effects
        game_state = action_game_state._replace(pending_effects=pending_effects,
                                                actions=action_game_state.actions - 1)
        game_state = move_specific_card_to_played_actions(game_state, action)

        choices.append(Choice(game_state, f"play {action.name}"))

    return choices

def treasure_phase_choices(treasure_game_state: GameState) -> List[Choice]:
    # We don't support treasure choices yet, so we always return only a single
    # choice of playing all your treasures.
    buy_game_state = treasure_game_state._replace(
        turn_phase=TURN_PHASES.BUY,
        total_money=(treasure_game_state.total_money
                     + money_from_treasures(treasure_game_state.current_player().hand)))
    return [Choice(buy_game_state, f"play all treasures")]

# TODO: make this return a set? Will need to stop using List in GameState
def buy_phase_choices(buy_game_state: GameState) -> List[Choice]:
    # Move to cleanup state if-and-only-if the player buys nothing OR if they
    # buy something with only 1 buy left.
    turn_phase_after_one_buy = TURN_PHASES.BUY if buy_game_state.buys > 1 else TURN_PHASES.CLEANUP
    game_state_after_one_buy = buy_game_state._replace(turn_phase=turn_phase_after_one_buy,
                                                       buys=buy_game_state.buys-1)

    player = buy_game_state.current_player()
    buyable_cards = cards_in_supply_costing_less_than(buy_game_state, buy_game_state.total_money)

    buy_choices = gainable_cards_to_choices(game_state_after_one_buy,
                                            buyable_cards,
                                            pay_card_cost=True,
                                            description_prefix="buy")

    cleanup_game_state = buy_game_state._replace(turn_phase=TURN_PHASES.CLEANUP)
    buy_nothing = Choice(game_state=cleanup_game_state, description="buy nothing")

    return [buy_nothing] + buy_choices

def add_card_to_top_of_deck(player: Player, card: Card) -> Player:
    return player._replace(top_of_deck=(card,) + player.top_of_deck)

def draw_card(player: Player) -> Player:
    if len(player.top_of_deck) == 0 and num_cards(player.deck) == 0 and num_cards(player.discard_pile) == 0:
        return player

    if len(player.top_of_deck) > 0:
        player = player._replace(hand=add_card(player.hand, player.top_of_deck[0]),
                                 top_of_deck=player.top_of_deck[1:])
        return player

    if len(player.top_of_deck) == 0 and num_cards(player.deck) == 0:
        player = player._replace(deck=player.discard_pile,
                                 discard_pile=empty_card_counts())

    deck = player.deck
    unique_cards, frequencies = zip(*deck.items())

    card_drawn = random.choices(unique_cards, weights=frequencies, k=1)[0]
    player = player._replace(hand=add_card(player.hand, card_drawn),
                             deck=remove_card(deck, card_drawn))

    return player

def discard_specific_card_current_player(game_state: GameState, card_discarded: Card) -> GameState:
    player = game_state.current_player()
    return game_state.replace_current_player_kwargs(
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
        print(f"Player index {index} not gaining {card.name} because its supply pile is empty")
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
        print(f"Current player not gaining {card.name} to hand because its supply pile is empty")
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

def gainable_cards_to_choices(game_state: GameState,
                              gainable_cards: List[Card],
                              pay_card_cost: bool,
                              description_prefix: str) -> List[Choice]:
    """
    Returns empty list if gainable_cards is empty
    """
    return [Choice(gain_card_current_player(game_state, card)
                   ._replace(total_money=game_state.total_money - (card.cost if pay_card_cost else 0)),
                   f"{description_prefix} {card.name}")
            for card in gainable_cards]

def gainable_cards_to_hand_to_choices(game_state: GameState,
                                      gainable_cards: List[Card],
                                      description_prefix: str) -> List[Choice]:
    return [Choice(gain_card_to_hand_current_player(game_state, card),
                   f"{description_prefix} {card.name} to hand")
            for card in gainable_cards]

def resolve_pending_effect(game_state: GameState, choosers: List) -> GameState:
    effect = game_state.pending_effects[0]
    remaining_effects = game_state.pending_effects[1:]
    game_state = game_state._replace(pending_effects=remaining_effects)
    current_player_chooser = choosers[game_state.current_player_index]

    # Always setting all these variables makes the code in the if branches a little more concise.
    current_player = game_state.current_player()
    discard_pile = current_player.discard_pile
    hand = current_player.hand

    if effect.name == EFFECT_NAME.DRAW_CARDS:
        return draw_cards_current_player(game_state, effect.value)
    elif effect.name == EFFECT_NAME.PLUS_ACTIONS:
        return game_state._replace(actions=game_state.actions + effect.value)
    elif effect.name == EFFECT_NAME.PRODUCE_MONEY:
        return game_state._replace(total_money=game_state.total_money + effect.value)
    elif effect.name == EFFECT_NAME.PLUS_BUYS:
        return game_state._replace(buys=game_state.buys + effect.value)
    elif effect.name == EFFECT_NAME.MAY_TRASH_A_CARD_FROM_YOUR_HAND:
        trash_nothing = Choice(game_state=game_state, description="trash nothing")
        choices = [trash_nothing]
        for card, freq in hand.items():
            after_trashing_card = game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            choices.append(Choice(game_state=after_trashing_card,
                                  description=f"trash {card.name}"))
        return offer_choice(game_state, choices, current_player_chooser)
    elif effect.name == EFFECT_NAME.GAIN_A_CARD_COSTING_UP_TO:
        gainable_cards = cards_in_supply_costing_less_than(game_state, effect.value)
        if len(gainable_cards) == 0:
            # Is the single choice here helpful for logging when these edge cases happen?
            # Can rethink this in the future.
            single_choice = [Choice(game_state=game_state,
                                    description=f"gain nothing since no cards in supply cost {effect.value} or less")]
            return offer_choice(game_state, single_choice, current_player_chooser)
        choices = gainable_cards_to_choices(game_state,
                                            gainable_cards,
                                            pay_card_cost=False,
                                            description_prefix="gain")
        return offer_choice(game_state, choices, current_player_chooser)
    elif effect.name == EFFECT_NAME.GAIN_A_TREASURE_TO_HAND_COSTING_UP_TO:
        gainable_cards = [card for card
                          in cards_in_supply_costing_less_than(game_state, effect.value)
                          if is_treasure(card)]
        if len(gainable_cards) == 0:
            single_choice = [Choice(game_state=game_state,
                                    description=f"gain nothing since no treasures in supply cost {effect.value} or less")]
            return offer_choice(game_state, single_choice, current_player_chooser)
        choices = gainable_cards_to_hand_to_choices(game_state, gainable_cards, "gain")
        return offer_choice(game_state, choices, current_player_chooser)
    elif effect.name == EFFECT_NAME.TRASH_GAIN_A_CARD_COSTING_UP_TO_X_MORE:
        if len(hand) == 0:
            single_choice = [Choice(game_state=game_state,
                                    description="trash and gain nothing since there are no cards in your hand")]
            return offer_choice(game_state, single_choice, current_player_chooser)

        choices = []
        for card, freq in hand.items():
            gain_effect = Effect(EFFECT_NAME.GAIN_A_CARD_COSTING_UP_TO, card.cost + effect.value)
            new_game_state = game_state.prepend_effect(gain_effect)
            new_game_state = new_game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            choices.append(Choice(game_state=new_game_state,
                                  description=f"trash {card.name}"))
        return offer_choice(game_state, choices, current_player_chooser)
    elif effect.name == EFFECT_NAME.MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE:
        choices = [Choice(game_state=game_state, description="trash nothing")]

        for card in (c for c, f in hand.items() if is_treasure(c)):
            gain_effect = Effect(EFFECT_NAME.GAIN_A_TREASURE_TO_HAND_COSTING_UP_TO, card.cost + effect.value)
            new_game_state = game_state.prepend_effect(gain_effect)
            new_game_state = new_game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            choices.append(Choice(game_state=new_game_state,
                                  description=f"trash {card.name}"))
        return offer_choice(game_state, choices, current_player_chooser)

    elif effect.name == EFFECT_NAME.DISCARD_ANY_NUMBER_THEN_DRAW_THAT_MANY:
        # Iterate over every card, including duplicatest
        for card in hand:
            discard_to_draw_game_state = (discard_specific_card_current_player(game_state, card)
                                          .prepend_effect(Effect(EFFECT_NAME.DRAW_CARDS, 1)))
            choices = [Choice(game_state=discard_to_draw_game_state, description=f"discard {card.name} to draw a card"),
                       Choice(game_state=game_state, description=f"don't discard {card.name}")]
            game_state = offer_choice(game_state, choices, current_player_chooser)
        return game_state
    elif effect.name == EFFECT_NAME.MAY_PUT_ANY_CARD_FROM_DISCARD_PILE_ONTO_DECK:
        choices = [Choice(game_state=game_state, description="do not put a card from discard pile onto deck")]
        for card, freq in discard_pile.items():
            new_current_player = current_player._replace(discard_pile=remove_card(discard_pile, card))
            new_current_player = add_card_to_top_of_deck(new_current_player, card)
            card_onto_deck_game_state = game_state.replace_current_player(new_current_player)
            choices.append(Choice(game_state=card_onto_deck_game_state,
                                  description=f"put {card.name} from discard pile on top of your deck"))
        return offer_choice(game_state, choices, current_player_chooser)

    elif effect.name == EFFECT_NAME.MAY_TRASH_A_COPPER_TO_PRODUCE_MONEY:
        choices = [Choice(game_state=game_state, description="Trash nothing")]
        copper = card_name_to_card("copper")
        if copper in hand:
            new_game_state = (game_state
                              .replace_current_player_kwargs(hand=remove_card(hand, copper))
                              .prepend_effect(Effect(EFFECT_NAME.PRODUCE_MONEY, effect.value)))
            choices.append(Choice(game_state=new_game_state, description="Trash a copper for 3 money"))
        return offer_choice(game_state, choices, current_player_chooser)

    elif effect.name == EFFECT_NAME.EACH_OTHER_PLAYER_DRAWS_A_CARD:
        new_players = [(p if index == game_state.current_player_index else draw_card(p))
                       for index, p in enumerate(game_state.players)]
        return game_state._replace(players=new_players)
    elif effect.name == EFFECT_NAME.EACH_OTHER_PLAYER_GAINS_A_CURSE:
        non_current_player_indices = list(range(len(game_state.players)))
        non_current_player_indices.remove(game_state.current_player_index)

        curse = card_name_to_card("curse")
        for index in non_current_player_indices:
            game_state = gain_card_by_player_index(game_state, curse, index)

        return game_state
    else:
        raise ValueError("resolve_pending_effect does not support effect named '{effec.name}'")

def do_cleanup_phase(game_state: GameState) -> GameState:
    assert game_state.turn_phase == TURN_PHASES.CLEANUP

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
    game_state = game_state._replace(turn_phase=TURN_PHASES.ACTION, actions=1, buys=1, total_money=0)

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
        assert False, f"Invalid number of players: {num_players}"

def initial_supply(num_players: int) -> Dict[str, int]:
    card_dict = initial_supply_base_cards(num_players)
    card_dict["smithy"] = 10
    card_dict["village"] = 10
    card_dict["laboratory"] = 10
    card_dict["chapel"] = 10
    card_dict["workshop"] = 10
    card_dict["remodel"] = 10
    card_dict["mine"] = 10
    card_dict["festival"] = 10
    card_dict["market"] = 10
    card_dict["cellar"] = 10
    card_dict["harbinger"] = 10
    card_dict["moneylender"] = 10
    card_dict["council room"] = 10
    card_dict["witch"] = 10
    return card_dict

def initial_game_state(player_names: List[str]) -> GameState:
    num_players = len(player_names)
    return GameState(players=[initial_player_state(name) for name in player_names],
                     current_player_index=0,
                     max_turns_per_player=0,
                     pending_effects=(),
                     actions=1,
                     buys=1,
                     total_money=0,
                     supply=dict_to_card_counts(initial_supply(num_players)),
                     turn_phase=TURN_PHASES.ACTION)

################################################################################
#                                                               Playing a Game #
################################################################################

def game_completed(game_state: GameState) -> bool:
    # distinguish between an card that has been fully bought up and a card that wasn't in the game
    # HACK for now assumes all cards defined are in the game
    original_non_empty_piles = len(CARD_DEFS)
    current_non_empty_piles = len([True for card in CARD_DEFS if game_state.supply.get(card, 0) == 0])
    num_empty_piles = original_non_empty_piles - current_non_empty_piles
    return (num_empty_piles >= 3
            or num_copies_of_card(game_state.supply, "province") == 0)

def offer_choice(game_state, choices, chooser) -> GameState:
    # ASSUMPTION: the choice is being made by the current player
    current_player_name = game_state.current_player().name
    if len(choices) == 1:
        print(f"{current_player_name}: {choices[0].description}")
        return choices[0].game_state

    # Keeping game_state as an argument, even though it may not be needed by value function approximation
    selected_choice_index = chooser(game_state, choices)
    selected_choice = choices[selected_choice_index]
    print(f"{current_player_name}: {selected_choice.description}")
    return selected_choice.game_state

def game_step(game_state: GameState, choosers: List) -> GameState:
    current_player_chooser = choosers[game_state.current_player_index]

    if len(game_state.pending_effects) > 0:
        return resolve_pending_effect(game_state, choosers)
    elif game_state.turn_phase == TURN_PHASES.ACTION:
        choices = action_phase_choices(game_state)
        return offer_choice(game_state, choices, current_player_chooser)
    elif game_state.turn_phase == TURN_PHASES.TREASURE:
        choices = treasure_phase_choices(game_state)
        return offer_choice(game_state, choices, current_player_chooser)
    elif game_state.turn_phase == TURN_PHASES.BUY:
        choices = buy_phase_choices(game_state)
        return offer_choice(game_state, choices, current_player_chooser)
    elif game_state.turn_phase == TURN_PHASES.CLEANUP:
        return do_cleanup_phase(game_state)
    else:
        raise ValueError("Unrecognized turn phase '{game_state.turn_phase}'")

def game_flow(player_names: List[str], choosers: List):
    game_state = initial_game_state(player_names)

    while not game_completed(game_state):
        game_state = game_step(game_state, choosers)

    print("----------------------------")
    for i, player in enumerate(game_state.players):
        print(f"{player.name} score: {total_player_vp(player)}")

    print(f"max turns per player: {game_state.max_turns_per_player}")
