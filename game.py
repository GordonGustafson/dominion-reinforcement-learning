from typing import List, NamedTuple

from cards import GameState, TurnPhase, move_specific_card_to_played_actions, money_from_treasures, \
    cards_in_supply_costing_less_than, Card, gain_card_current_player, gain_card_to_hand_current_player, \
    discard_specific_card_player_index, unique_cards, EffectName, draw_cards_current_player, \
    card_name_to_card, remove_card, is_treasure, num_cards, \
    Effect, discard_specific_card_current_player, \
    add_card_to_top_of_deck, \
    draw_card, \
    non_current_player_indices, gain_card_by_player_index, take_top_card_off_of_deck, is_treasure_other_than_a_copper, \
    card_sequence_to_card_counts, add_card_counts, add_card, CARD_DICT, num_copies_of_card, \
    do_cleanup_phase, initial_game_state, get_total_player_vp, GameOutcome
from actions import GainCard, GainNothing, GainCardToHand, PlayActionCard, PlayNoActionCard, DiscardCard, \
    DiscardCardToDrawACard, DontDiscardCardToDrawACard, PutCardFromDiscardPileOntoDeck, \
    PutNoCardFromDiscardPileOntoDeck, TrashCardFromHand, TrashNoCardFromHand, TrashRevealedCard, \
    TrashCardFromHandToGainCardCostingUpTo2More, TrashTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More, \
    TrashNoTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More, TrashACopperFor3Money, \
    DontTrashACopperFor3Money, PlayAllTreasures, Action

################################################################################
#                                                                      Choices #
################################################################################

Choice = NamedTuple("Choice", [
    ("game_state", GameState),
    ("action", Action),
])


def action_phase_choices(action_game_state: GameState) -> List[Choice]:
    player = action_game_state.current_player()
    treasure_game_state = action_game_state._replace(turn_phase=TurnPhase.TREASURE)

    move_to_treasure_phase = Choice(game_state=treasure_game_state, action=PlayNoActionCard())
    choices = [move_to_treasure_phase]

    assert action_game_state.actions >= 0
    if action_game_state.actions == 0:
        return choices

    playable_action_cards = [card for card in player.hand
                        if len(card.action_effects) > 0]
    for action_card in playable_action_cards:
        # The action effects from this card happen before any other pending
        # effects, so we put them on the left
        pending_effects = action_card.action_effects + action_game_state.pending_effects
        game_state = action_game_state._replace(pending_effects=pending_effects,
                                                actions=action_game_state.actions - 1)
        game_state = move_specific_card_to_played_actions(game_state, action_card)

        choices.append(Choice(game_state=game_state, action=PlayActionCard(action_card)))

    return choices


def treasure_phase_choices(treasure_game_state: GameState) -> List[Choice]:
    # We don't support treasure choices yet, so we always return only a single
    # choice of playing all your treasures.
    buy_game_state = treasure_game_state._replace(
        turn_phase=TurnPhase.BUY,
        total_money=(treasure_game_state.total_money
                     + money_from_treasures(treasure_game_state.current_player().hand)))
    return [Choice(game_state=buy_game_state, action=PlayAllTreasures())]


# TODO: make this return a set? Will need to stop using List in GameState
def buy_phase_choices(buy_game_state: GameState) -> List[Choice]:
    # Move to cleanup state if-and-only-if the player buys nothing OR if they
    # buy something with only 1 buy left.
    turn_phase_after_one_buy = TurnPhase.BUY if buy_game_state.buys > 1 else TurnPhase.CLEANUP
    game_state_after_one_buy = buy_game_state._replace(turn_phase=turn_phase_after_one_buy,
                                                       buys=buy_game_state.buys-1)

    buyable_cards = cards_in_supply_costing_less_than(buy_game_state, buy_game_state.total_money)

    buy_choices = gainable_cards_to_choices(game_state_after_one_buy,
                                            buyable_cards,
                                            pay_card_cost=True)

    cleanup_game_state = buy_game_state._replace(turn_phase=TurnPhase.CLEANUP)
    buy_nothing = Choice(game_state=cleanup_game_state, action=GainNothing())

    return [buy_nothing] + buy_choices


StateActionPair = NamedTuple("StateActionPair", [
    ("state", GameState),
    ("possible_actions", List[Choice]),
    ("selected_action", int),
])


def gainable_cards_to_choices(game_state: GameState,
                              gainable_cards: List[Card],
                              pay_card_cost: bool) -> List[Choice]:
    """
    Returns empty list if gainable_cards is empty
    """
    return [Choice(game_state=gain_card_current_player(game_state, card)
                   ._replace(total_money=game_state.total_money - (card.cost if pay_card_cost else 0)),
                   action=GainCard(card))
            for card in gainable_cards]


def gainable_cards_to_hand_to_choices(game_state: GameState,
                                      gainable_cards: List[Card]) -> List[Choice]:
    return [Choice(game_state=gain_card_to_hand_current_player(game_state, card),
                   action=GainCardToHand(card))
            for card in gainable_cards]


def player_index_discards_one_card_choices(game_state: GameState, player_index: int) -> List[Choice]:
    player_hand = game_state.players[player_index].hand
    return [
        Choice(game_state=discard_specific_card_player_index(game_state, card, player_index),
               action=DiscardCard(card))
        for card in unique_cards(player_hand)
    ]


def resolve_pending_effect(game_state: GameState, choosers: List) -> GameState:
    effect = game_state.pending_effects[0]
    remaining_effects = game_state.pending_effects[1:]
    game_state = game_state._replace(pending_effects=remaining_effects)
    current_player_chooser = choosers[game_state.current_player_index]

    # Always setting all these variables makes the code in the if branches a little more concise.
    current_player_index = game_state.current_player_index
    current_player = game_state.current_player()
    discard_pile = current_player.discard_pile
    hand = current_player.hand

    if effect.name == EffectName.DRAW_CARDS:
        return draw_cards_current_player(game_state, effect.value)
    elif effect.name == EffectName.PLUS_ACTIONS:
        return game_state._replace(actions=game_state.actions + effect.value)
    elif effect.name == EffectName.PRODUCE_MONEY:
        return game_state._replace(total_money=game_state.total_money + effect.value)
    elif effect.name == EffectName.PLUS_BUYS:
        return game_state._replace(buys=game_state.buys + effect.value)
    elif effect.name == EffectName.GAIN_A_GOLD:
        return gain_card_current_player(game_state, card_name_to_card("gold"))
    elif effect.name == EffectName.MAY_TRASH_A_CARD_FROM_YOUR_HAND:
        trash_nothing = Choice(game_state=game_state, action=TrashNoCardFromHand())
        choices = [trash_nothing]
        for card, freq in hand.items():
            after_trashing_card = game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            choices.append(Choice(game_state=after_trashing_card, action=TrashCardFromHand(card)))
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)
    elif effect.name == EffectName.GAIN_A_CARD_COSTING_UP_TO:
        gainable_cards = cards_in_supply_costing_less_than(game_state, effect.value)
        if len(gainable_cards) == 0:
            single_choice = [Choice(game_state=game_state, action=GainNothing())]
            return offer_choice(game_state, single_choice, current_player_chooser, current_player_index)
        choices = gainable_cards_to_choices(game_state,
                                            gainable_cards,
                                            pay_card_cost=False)
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)
    elif effect.name == EffectName.GAIN_A_TREASURE_TO_HAND_COSTING_UP_TO:
        gainable_cards = [card for card
                          in cards_in_supply_costing_less_than(game_state, effect.value)
                          if is_treasure(card)]
        if len(gainable_cards) == 0:
            single_choice = [Choice(game_state=game_state, action=GainNothing())]
            return offer_choice(game_state, single_choice, current_player_chooser, current_player_index)
        choices = gainable_cards_to_hand_to_choices(game_state, gainable_cards)
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)
    elif effect.name == EffectName.TRASH_GAIN_A_CARD_COSTING_UP_TO_X_MORE:
        if num_cards(hand) == 0:
            single_choice = [Choice(game_state=game_state, action=GainNothing())]
            return offer_choice(game_state, single_choice, current_player_chooser, current_player_index)

        choices = []
        for card, freq in hand.items():
            gain_effect = Effect(EffectName.GAIN_A_CARD_COSTING_UP_TO, card.cost + effect.value)
            new_game_state = game_state.prepend_effect(gain_effect)
            new_game_state = new_game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            choices.append(Choice(game_state=new_game_state,
                                  action=TrashCardFromHandToGainCardCostingUpTo2More(card)))
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)
    elif effect.name == EffectName.MAY_TRASH_TREASURE_GAIN_TREASURE_TO_HAND_COSTING_UP_TO_X_MORE:
        choices = [
            Choice(game_state=game_state, action=TrashNoTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More())]

        for card in (c for c, f in hand.items() if is_treasure(c)):
            gain_effect = Effect(EffectName.GAIN_A_TREASURE_TO_HAND_COSTING_UP_TO, card.cost + effect.value)
            new_game_state = game_state.prepend_effect(gain_effect)
            new_game_state = new_game_state.replace_current_player_kwargs(hand=remove_card(hand, card))
            choices.append(Choice(game_state=new_game_state,
                                  action=TrashTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More(card)))
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)

    elif effect.name == EffectName.DISCARD_ANY_NUMBER_THEN_DRAW_THAT_MANY:
        # Iterate over every card, including duplicates
        for card in hand:
            discard_to_draw_game_state = (discard_specific_card_current_player(game_state, card)
                                          .prepend_effect(Effect(EffectName.DRAW_CARDS, 1)))
            choices = [Choice(game_state=discard_to_draw_game_state, action=DiscardCardToDrawACard(card)),
                       Choice(game_state=game_state, action=DontDiscardCardToDrawACard(card))]
            game_state = offer_choice(game_state, choices, current_player_chooser, current_player_index)
        return game_state
    elif effect.name == EffectName.MAY_PUT_ANY_CARD_FROM_DISCARD_PILE_ONTO_DECK:
        choices = [Choice(game_state=game_state, action=PutNoCardFromDiscardPileOntoDeck())]
        for card, freq in discard_pile.items():
            new_current_player = current_player._replace(discard_pile=remove_card(discard_pile, card))
            new_current_player = add_card_to_top_of_deck(new_current_player, card)
            card_onto_deck_game_state = game_state.replace_current_player(new_current_player)
            choices.append(Choice(game_state=card_onto_deck_game_state, action=PutCardFromDiscardPileOntoDeck(card)))
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)

    elif effect.name == EffectName.MAY_TRASH_A_COPPER_TO_PRODUCE_MONEY:
        choices = [Choice(game_state=game_state, action=DontTrashACopperFor3Money())]
        copper = card_name_to_card("copper")
        if copper in hand:
            new_game_state = (game_state
                              .replace_current_player_kwargs(hand=remove_card(hand, copper))
                              .prepend_effect(Effect(EffectName.PRODUCE_MONEY, effect.value)))
            choices.append(Choice(game_state=new_game_state, action=TrashACopperFor3Money()))
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)

    elif effect.name == EffectName.EACH_OTHER_PLAYER_DRAWS_A_CARD:
        new_players = [(p if index == game_state.current_player_index else draw_card(p))
                       for index, p in enumerate(game_state.players)]
        return game_state._replace(players=new_players)
    elif effect.name == EffectName.EACH_OTHER_PLAYER_GAINS_A_CURSE:
        curse = card_name_to_card("curse")
        for index in non_current_player_indices(game_state):
            game_state = gain_card_by_player_index(game_state, curse, index)

        return game_state
    elif effect.name == EffectName.EACH_OTHER_PLAYER_DISCARDS_DOWN_TO:
        for other_player_index in non_current_player_indices(game_state):
            while num_cards(game_state.players[other_player_index].hand) > effect.value:
                # TODO: Does game_state need to be rotated to enable the other chooser to see itself as the current player?
                game_state = offer_choice(game_state,
                                          player_index_discards_one_card_choices(game_state, other_player_index),
                                          choosers[other_player_index],
                                          other_player_index)
        return game_state
    elif effect.name == EffectName.EACH_OTHER_PLAYER_BANDIT_EFFECT:
        # each other player reveals the top 2 cards of their deck, trashes a
        # revealed treasure other than copper, and discards the rest
        for other_player_index in non_current_player_indices(game_state):
            other_player = game_state.players[other_player_index]
            other_player, first_card = take_top_card_off_of_deck(other_player)
            other_player, second_card = take_top_card_off_of_deck(other_player)
            revealed_cards_list = [card for card in [first_card, second_card] if card is not None]
            revealed_cards_is_treasure_other_than_a_copper = [is_treasure_other_than_a_copper(c) for c in revealed_cards_list]
            if not any(revealed_cards_is_treasure_other_than_a_copper):
                # Put all cards that were revealed into the discard pile. It's possible that 0, 1, or 2 cards were revealed, but none should be trashed.
                revealed_card_counts = card_sequence_to_card_counts(revealed_cards_list)
                game_state = game_state.replace_player_by_index(other_player_index,
                                                                other_player._replace(discard_pile=add_card_counts(other_player.discard_pile, revealed_card_counts)))
            elif revealed_cards_is_treasure_other_than_a_copper == [True]:
                # We took one treasure other than a copper off the deck and didn't have another card to take.
                # The card was removed from the deck already ("trashed").
                game_state = game_state.replace_player_by_index(other_player_index, other_player)
            elif revealed_cards_is_treasure_other_than_a_copper == [True, False]:
                # Trash the first card, they get the second card back.
                game_state = game_state.replace_player_by_index(other_player_index,
                                                                other_player._replace(discard_pile=add_card(other_player.discard_pile, second_card)))
            elif revealed_cards_is_treasure_other_than_a_copper == [False, True]:
                # Trash the second card, they get the first card back.
                game_state = game_state.replace_player_by_index(other_player_index,
                                                                other_player._replace(discard_pile=add_card(other_player.discard_pile, first_card)))
            elif revealed_cards_is_treasure_other_than_a_copper == [True, True]:
                choices = [Choice(game_state.replace_player_by_index(other_player_index,
                                                                     other_player._replace(discard_pile=add_card(other_player.discard_pile, first_card))),
                                  action=TrashRevealedCard(second_card)),
                           Choice(game_state.replace_player_by_index(other_player_index,
                                                                     other_player._replace(discard_pile=add_card(other_player.discard_pile, second_card))),
                                  action=TrashRevealedCard(first_card))]

                game_state = offer_choice(game_state,
                                          choices,
                                          choosers[other_player_index],
                                          other_player_index)
            else:
                assert False
        return game_state

    else:
        raise ValueError(f"resolve_pending_effect does not support effect named '{effect.name}'")


def offer_choice(game_state, choices, chooser, player_index_making_choice: int) -> GameState:
    player_name = game_state.players[player_index_making_choice].name
    if len(choices) == 1:
        return choices[0].game_state

    # Keeping game_state as an argument, even though it may not be needed by value function approximation
    selected_choice_index = chooser.make_choice(game_state, choices, player_index_making_choice)
    selected_choice = choices[selected_choice_index]
    print(f"{player_name}: {selected_choice.action.get_description()}")
    return selected_choice.game_state


def game_completed(game_state: GameState) -> bool:
    # distinguish between an card that has been fully bought up and a card that wasn't in the game
    # HACK for now assumes all cards defined are in the game
    original_non_empty_piles = len(CARD_DICT)
    current_non_empty_piles = len([True for card in CARD_DICT if game_state.supply.get(card, 0) == 0])
    num_empty_piles = original_non_empty_piles - current_non_empty_piles
    return (num_empty_piles >= 3
            or num_copies_of_card(game_state.supply, "province") == 0)


def game_step(game_state: GameState, choosers: List) -> GameState:
    current_player_index = game_state.current_player_index
    current_player_chooser = choosers[current_player_index]

    if len(game_state.pending_effects) > 0:
        return resolve_pending_effect(game_state, choosers)
    elif game_state.turn_phase == TurnPhase.ACTION:
        choices = action_phase_choices(game_state)
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)
    elif game_state.turn_phase == TurnPhase.TREASURE:
        choices = treasure_phase_choices(game_state)
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)
    elif game_state.turn_phase == TurnPhase.BUY:
        choices = buy_phase_choices(game_state)
        return offer_choice(game_state, choices, current_player_chooser, current_player_index)
    elif game_state.turn_phase == TurnPhase.CLEANUP:
        return do_cleanup_phase(game_state)
    else:
        raise ValueError("Unrecognized turn phase '{game_state.turn_phase}'")


def game_flow(player_names: List[str], choosers: List):
    game_state = initial_game_state(player_names)

    while not game_completed(game_state) or game_state.turn_phase != TurnPhase.CLEANUP:
        game_state = game_step(game_state, choosers)

    print("----------------------------")
    for i, player in enumerate(game_state.players):
        print(f"{player.name} score: {get_total_player_vp(player)}")

    # Hardcoding logic to assume two players for now. With three there can be
    # two players that draw and one that loses.
    player_vps = [get_total_player_vp(player) for player in game_state.players]
    # We end the game before `do_cleanup_phase` rotates the current player, so
    # game_state.current_player_index had the last turn.
    players_had_equal_number_of_turns = game_state.current_player_index != game_state.first_player_index
    if player_vps[0] == player_vps[1] and players_had_equal_number_of_turns:
        print("GAME OUTPUT: DRAW")
        choosers[0]._game_outcome = GameOutcome.DRAW
        choosers[1]._game_outcome = GameOutcome.DRAW
    elif player_vps[0] == player_vps[1] and not players_had_equal_number_of_turns:
        print(f"GAME OUTPUT: {game_state.players[1-game_state.first_player_index].name} WINS BY TIE-BREAKER")
        choosers[game_state.first_player_index]._game_outcome = GameOutcome.LOSS
        choosers[1-game_state.first_player_index]._game_outcome = GameOutcome.WIN
    elif player_vps[1] > player_vps[0]:
        print(f"GAME OUTPUT: {game_state.players[1].name} WINS")
        choosers[0]._game_outcome = GameOutcome.LOSS
        choosers[1]._game_outcome = GameOutcome.WIN
    elif player_vps[0] > player_vps[1]:
        print(f"GAME OUTPUT: {game_state.players[0].name} WINS")
        choosers[0]._game_outcome = GameOutcome.WIN
        choosers[1]._game_outcome = GameOutcome.LOSS
    else:
        assert False

    print(f"max turns per player: {game_state.max_turns_per_player}")
