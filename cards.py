import random

from multiset import Multiset
from typing import NamedTuple, Tuple, List, Dict


# TODO: make this a proper Python enum
class EFFECT_NAME:
    DRAW_CARDS = "draw_cards"
    PLUS_ACTIONS = "plus_actions"
    TRASH_A_CARD_FROM_YOUR_HAND = "trash_a_card_from_your_hand"
    GAIN_A_CARD_COSTING_UP_TO = "gain_a_card_costing_up_to"
    GAIN_A_TREASURE_TO_HAND_COSTING_UP_TO = "gain_a_treasure_to_hand_costing_up_to"
    TRASH_GAIN_A_CARD_COSTING_UP_TO_X_MORE = "trash_gain_a_card_costing_up_to_x_more"
    MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE = "may_trash_treasure_gain_treasure_to_hand_costing_up_to_x_more"

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
    ("discard_pile", CardCounts),
    ("name", str),
])

def make_player(hand=Multiset(),
                deck=Multiset(),
                discard_pile=Multiset(),
                name="unnamed player"):
    return Player(hand, deck, discard_pile, name)

# TODO: make this a proper Python enum
class TURN_PHASES:
    ACTION = "ACTION"
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
    ("pending_effects", Tuple[Effect])
])

class GameState(_GameStateBase):
    def current_player(self) -> Player:
        return self.players[self.current_player_index]

    # These replace_current_player... methods are syntactic sugar for making other code clearer
    def replace_current_player(self, player):
        players_copy = self.players[:]
        players_copy[self.current_player_index] = player
        return self._replace(players=players_copy)

    def replace_current_player_kwargs(self, **kwargs):
        return self.replace_current_player(self.current_player()._replace(**kwargs))

def make_game_state(
        players,
        current_player_index=0,
        max_turns_per_player=0,
        round=0,
        supply=Multiset(),
        turn_phase=TURN_PHASES.ACTION,
        actions=1,
        pending_effects=()):
    return GameState(players, current_player_index, max_turns_per_player, supply, turn_phase, actions, pending_effects)

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
    "chapel":     make_card(name="chapel",     cost=2, action_effects=(Effect(EFFECT_NAME.TRASH_A_CARD_FROM_YOUR_HAND, 1),
                                                                       Effect(EFFECT_NAME.TRASH_A_CARD_FROM_YOUR_HAND, 1),
                                                                       Effect(EFFECT_NAME.TRASH_A_CARD_FROM_YOUR_HAND, 1),
                                                                       Effect(EFFECT_NAME.TRASH_A_CARD_FROM_YOUR_HAND, 1))),

    # gaining cards
    "workshop":  make_card(name="workshop", cost=3, action_effects=(Effect(EFFECT_NAME.GAIN_A_CARD_COSTING_UP_TO, 4),)),
    "remodel":   make_card(name="remodel",  cost=4, action_effects=(Effect(EFFECT_NAME.TRASH_GAIN_A_CARD_COSTING_UP_TO_X_MORE, 2),)),
    "mine":      make_card(name="mine",     cost=5, action_effects=(Effect(EFFECT_NAME.MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE, 3),)),

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

def add_card_counts(card_counts_lhs: CardCounts, card_counts_rhs: CardCounts) -> CardCounts:
    return card_counts_lhs + card_counts_rhs

def is_treasure(card: Card) -> bool:
    return len(card.treasure_effects) > 0

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

def total_player_vp(player: Player) -> int:
    return vp_total(player.hand) + vp_total(player.deck) + vp_total(player.discard_pile)

def average_treasure_value_per_card(player: Player) -> float:
    player_treasure_total = treasure_total(player.hand) + treasure_total(player.deck) + treasure_total(player.discard_pile)
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
    buy_game_state = action_game_state._replace(turn_phase=TURN_PHASES.BUY)

    move_to_buy_phase = Choice(game_state=buy_game_state, description="move to buy phase")
    choices = [move_to_buy_phase]

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
        game_state = discard_specific_card_current_player(game_state, action)

        choices.append(Choice(game_state, f"play {action.name}"))

    return choices

# TODO: make this return a set? Will need to stop using List in GameState
def buy_phase_choices(buy_game_state: GameState) -> List[Choice]:
    player = buy_game_state.current_player()
    total_money_for_turn = treasure_total(player.hand)

    buyable_cards = cards_in_supply_costing_less_than(buy_game_state, total_money_for_turn)

    cleanup_game_state = buy_game_state._replace(turn_phase=TURN_PHASES.CLEANUP)
    buy_nothing = Choice(game_state=cleanup_game_state, description="buy nothing")
    buy_choices = gainable_cards_to_choices(cleanup_game_state, buyable_cards, "buy")

    return [buy_nothing] + buy_choices

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

def discard_specific_card_current_player(game_state: GameState, card_discarded: Card) -> Player:
    player = game_state.current_player()
    return game_state.replace_current_player_kwargs(
        hand=remove_card(player.hand, card_discarded),
        discard_pile=add_card(player.discard_pile, card_discarded))

def draw_cards_current_player(game_state: GameState, num_cards_to_draw: int) -> GameState:
    player = game_state.current_player()
    for _ in range(num_cards_to_draw):
        player = draw_card(player)
    return game_state.replace_current_player(player)

def gain_card_current_player(game_state: GameState, card: Card) -> GameState:
    return (game_state
            ._replace(supply=remove_card(game_state.supply, card))
            .replace_current_player_kwargs(discard_pile=add_card(game_state.current_player().discard_pile,
                                                                 card)))

def gain_card_to_hand_current_player(game_state: GameState, card: Card) -> GameState:
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
                              description_prefix: str) -> List[Choice]:
    """
    Returns empty list if gainable_cards is empty
    """
    return [Choice(gain_card_current_player(game_state, card),
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

    if effect.name == EFFECT_NAME.DRAW_CARDS:
        return draw_cards_current_player(game_state, effect.value)
    elif effect.name == EFFECT_NAME.PLUS_ACTIONS:
        return game_state._replace(actions=game_state.actions + effect.value)
    elif effect.name == EFFECT_NAME.TRASH_A_CARD_FROM_YOUR_HAND:
        trash_nothing = Choice(game_state=game_state, description="trash nothing")
        choices = [trash_nothing]
        hand = game_state.current_player().hand
        for card in hand:
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
        choices = gainable_cards_to_choices(game_state, gainable_cards, "gain")
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
        hand = game_state.current_player().hand
        if len(hand) == 0:
            single_choice = [Choice(game_state=game_state,
                                    description="trash and gain nothing since there are no cards in your hand")]
            return offer_choice(game_state, single_choice, current_player_chooser)

        choices = []
        for card in hand:
            gain_effect = Effect(EFFECT_NAME.GAIN_A_CARD_COSTING_UP_TO, card.cost + effect.value)
            new_pending_effects = (gain_effect,) + game_state.pending_effects
            new_game_state = game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            new_game_state = new_game_state._replace(pending_effects=new_pending_effects)
            choices.append(Choice(game_state=new_game_state,
                                  description=f"trash {card.name}"))
        return offer_choice(game_state, choices, current_player_chooser)
    elif effect.name == EFFECT_NAME.MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE:
        hand = game_state.current_player().hand
        choices = [Choice(game_state=game_state, description="trash nothing")]

        for card in (c for c in hand if is_treasure(c)):
            gain_effect = Effect(EFFECT_NAME.GAIN_A_TREASURE_TO_HAND_COSTING_UP_TO, card.cost + effect.value)
            new_pending_effects = (gain_effect,) + game_state.pending_effects
            new_game_state = game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            new_game_state = new_game_state._replace(pending_effects=new_pending_effects)
            choices.append(Choice(game_state=new_game_state,
                                  description=f"trash {card.name}"))
        return offer_choice(game_state, choices, current_player_chooser)

    else:
        raise ValueError("resolve_pending_effect does not support effect named '{effec.name}'")

def do_cleanup_phase(game_state: GameState) -> GameState:
    assert game_state.turn_phase == TURN_PHASES.CLEANUP

    # Discard your hand
    game_state = game_state.replace_current_player_kwargs(
        hand=empty_card_counts(),
        discard_pile=add_card_counts(game_state.current_player().hand,
                                     game_state.current_player().discard_pile),
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

    # Next player gets 1 action
    game_state = game_state._replace(turn_phase=TURN_PHASES.ACTION, actions=1)

    return game_state

################################################################################
#                                                              Starting a Game #
################################################################################

def initial_player_state(name: str) -> Player:
    player = Player(name=name,
                    hand=dict_to_card_counts({}),
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
    return card_dict

def initial_game_state(player_names: List[str]) -> GameState:
    num_players = len(player_names)
    return GameState(players=[initial_player_state(name) for name in player_names],
                     current_player_index=0,
                     max_turns_per_player=0,
                     pending_effects=(),
                     actions=1,
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
