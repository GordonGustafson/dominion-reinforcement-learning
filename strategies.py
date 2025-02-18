from collections.abc import Callable

from actions import action_to_action_id, GainNothingOnFirstBuy, GainMostExpensiveCardAvailable, PlayActionCard, \
    PlayNoActionCard, \
    GainCardInsteadOfMoreExpensiveCard, GainNothingOnSecondOrLaterBuy
from chooser import Chooser
from game import Choice
from pytorch.dataloader import tensorify_inputs
import torch

import featurizer
from cards import *

ChooserFunction = Callable[[Chooser, GameState, List[Choice], int], int]

def random_strategy(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
    return random.randrange(len(choices))

def read_int_from_stdin() -> int:
    input_string = input()
    while not input_string.isdecimal():
        print("non-numeric input, try again")
        input_string = input()
    return int(input_string)

def user_chooser(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
    player = game_state.current_player()
    # HACK: assumes only one opponent
    opponent = game_state.players[non_current_player_indices(game_state)[0]]
    print(f"{opponent.name} hand: {card_counts_to_dict(opponent.hand)}, deck: {card_counts_to_dict(opponent.deck)}, discard pile: {card_counts_to_dict(opponent.discard_pile)}, top of deck: {opponent.top_of_deck}")
    print(f"{player.name} hand: {card_counts_to_dict(player.hand)}, deck: {card_counts_to_dict(player.deck)}, discard pile: {card_counts_to_dict(player.discard_pile)}, top of deck: {player.top_of_deck}")
    print(f"{player.name} money: {game_state.total_money}, actions: {game_state.actions}, hand: {card_counts_to_dict(player.hand)}")
    for i, choice in enumerate(choices):
        print(f"{i}: {choice.action.get_description()}")

    selected_choice = read_int_from_stdin()
    while selected_choice not in range(0, len(choices)):
        print("index out of bounds, try again")
        selected_choice = read_int_from_stdin()

    return int(selected_choice)

def scikit_learn_max_state_score_strategy(scikit_learn_model) -> ChooserFunction:
    def choose_with_model(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
        state_values = [scikit_learn_model.predict(featurizer.game_state_to_df(c.game_state, player_index)) for c in choices]
        return state_values.index(max(state_values))

    return choose_with_model

def pytorch_max_state_score_strategy(pytorch_model) -> ChooserFunction:
    def choose_with_model(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
        state_values = [pytorch_model.forward(tensorify_inputs(featurizer.game_state_to_df(c.game_state, player_index))) for c in choices]
        return state_values.index(max(state_values))

    return choose_with_model

def pytorch_max_action_score_strategy(pytorch_model) -> ChooserFunction:
    def choose_with_model(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
        action_scores = pytorch_model.forward(
            tensorify_inputs(featurizer.game_state_to_df(game_state, player_index)))
        action_scores = action_scores.squeeze()
        choice_with_max_action_score = max(choices,
                                           key=lambda choice: action_scores[action_to_action_id(choice.action)])
        return choices.index(choice_with_max_action_score)

    return choose_with_model

def pytorch_sampled_action_strategy(pytorch_model) -> ChooserFunction:
    def sample_from_model(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
        action_scores = pytorch_model.forward(
            tensorify_inputs(featurizer.game_state_to_df(game_state, player_index)))
        action_scores = action_scores.squeeze()
        valid_action_ids = [action_to_action_id(choice.action) for choice in choices]
        choice_scores = torch.index_select(input=action_scores, dim=0, index=torch.tensor(valid_action_ids))
        action_probabilities = torch.nn.functional.softmax(choice_scores, dim=0)
        distribution = torch.distributions.Categorical(probs=action_probabilities)
        selected_choice_index = distribution.sample().item()
        chooser.action_probability_tensors.append(action_probabilities[selected_choice_index])
        chooser.valid_action_distribution_entropies.append(distribution.entropy())
        return selected_choice_index

    return sample_from_model

def wrap_with_epsilon_greedy(chooser_function: ChooserFunction, epsilon: float) -> ChooserFunction:
    def choose_with_epsilon_greedy(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
        num_choices = len(choices)
        greedy_choice = chooser_function(chooser, game_state, choices, player_index)
        weights = [epsilon / num_choices] * num_choices
        weights[greedy_choice] += 1 - epsilon

        return random.choices(range(num_choices), weights=weights, k=1)[0]

    return choose_with_epsilon_greedy

################################################################################
# Playing Strategies                                                           #
################################################################################

def combination_of_gaining_strategy_and_playing_strategy(gaining_strategy: ChooserFunction, playing_strategy: ChooserFunction) -> ChooserFunction:
    def choose_with_appropriate_strategy(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
        possible_actions = [choice.action for choice in choices]
        if all(type(action) in {GainMostExpensiveCardAvailable,
                                GainCardInsteadOfMoreExpensiveCard,
                                GainNothingOnFirstBuy,
                                GainNothingOnSecondOrLaterBuy} for action in possible_actions):
            return gaining_strategy(chooser, game_state, choices, player_index)
        return playing_strategy(chooser, game_state, choices, player_index)

    return choose_with_appropriate_strategy

def play_plus_actions_first(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int):
    possible_actions = [choice.action for choice in choices]
    action_to_score = {
        PlayNoActionCard(): -1,
        PlayActionCard(card_name_to_card("smithy")): 0,
        PlayActionCard(card_name_to_card("market")): 1,
        PlayActionCard(card_name_to_card("laboratory")): 1,
        PlayActionCard(card_name_to_card("festival")): 2,
        PlayActionCard(card_name_to_card("village")): 2,
    }
    possible_action_scores = [action_to_score[action] for action in possible_actions]
    return possible_action_scores.index(max(possible_action_scores))


################################################################################
# Baseline Strategies                                                         # 
################################################################################

def big_money_until_province_then_all_victory(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
    current_vp = get_total_player_vp(game_state.current_player())
    choice_delta_vps = [get_total_player_vp(choice.game_state.current_player()) - current_vp
                        for choice in choices]
    max_delta_vp = max(choice_delta_vps)

    # if we can buy a province OR already have one, maximize our VP
    if max_delta_vp >= 6 or current_vp >= 6:
        return choice_delta_vps.index(max_delta_vp)

    # otherwise, maximize our average treasure value per card we have
    choice_with_best_average_treasure_value = (
        max(choices,
            key=lambda c: get_average_treasure_value_per_card(c.game_state.current_player())))
    return choices.index(choice_with_best_average_treasure_value)

def big_money_provinces_only(chooser: Chooser, game_state: GameState, choices: List[Choice], player_index: int) -> int:
    current_vp = get_total_player_vp(game_state.current_player())
    choice_delta_vps = [get_total_player_vp(choice.game_state.current_player()) - current_vp
                        for choice in choices]
    max_delta_vp = max(choice_delta_vps)

    # if we can buy a province, maximize our VP
    if max_delta_vp >= 6:
        return choice_delta_vps.index(max_delta_vp)

    # otherwise, maximize our average treasure value per card we have
    choice_with_best_average_treasure_value = (
        max(choices,
            key=lambda c: get_average_treasure_value_per_card(c.game_state.current_player())))
    return choices.index(choice_with_best_average_treasure_value)
