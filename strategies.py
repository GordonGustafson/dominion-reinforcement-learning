from cards import *

def user_chooser(game_state: GameState, choices: List[Choice]) -> int:
    print(f"actions: {game_state.actions}, hand: {card_counts_to_dict(game_state.current_player().hand)}")
    for i, choice in enumerate(choices):
        print(f"{i}: {choice.description}")

    selected_choice = input()
    while selected_choice == '' or int(selected_choice) >= len(choices) or int(selected_choice) < 0:
        print("index out of bounds, try again")
        selected_choice = input()

    return int(selected_choice)

def big_money_until_province_then_all_victory(game_state: GameState, choices: List[Choice]) -> int:
    current_vp = total_player_vp(game_state.current_player())
    choice_delta_vps = [total_player_vp(choice.game_state.current_player()) - current_vp
                        for choice in choices]
    max_delta_vp = max(choice_delta_vps)

    # if we can buy a province OR already have one, maximize our VP
    if max_delta_vp >= 6 or current_vp >= 6:
        return choice_delta_vps.index(max_delta_vp)

    # otherwise, maximize our average treasure value per card we have
    choice_with_best_average_treasure_value = (
        max(choices,
            key=lambda c: average_treasure_value_per_card(c.game_state.current_player())))
    return choices.index(choice_with_best_average_treasure_value)

def big_money_provinces_only(game_state: GameState, choices: List[Choice]) -> int:
    current_vp = total_player_vp(game_state.current_player())
    choice_delta_vps = [total_player_vp(choice.game_state.current_player()) - current_vp
                        for choice in choices]
    max_delta_vp = max(choice_delta_vps)

    # if we can buy a province OR already have one, maximize our VP
    if max_delta_vp >= 6:
        return choice_delta_vps.index(max_delta_vp)

    # otherwise, maximize our average treasure value per card we have
    choice_with_best_average_treasure_value = (
        max(choices,
            key=lambda c: average_treasure_value_per_card(c.game_state.current_player())))
    return choices.index(choice_with_best_average_treasure_value)
