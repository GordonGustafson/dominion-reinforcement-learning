from actions import GainMostExpensiveCardAvailable, GainCardInsteadOfMoreExpensiveCard
from cards import CARD_LIST
from chooser import Chooser

import strategies
import play

VP_REWARD_MULTIPLIER = 0.01
ACTION_TO_REWARD = {}
for card in CARD_LIST:
    _card_reward = VP_REWARD_MULTIPLIER * (card.vp_effects[0].value if len(card.vp_effects) > 0 else 0)
    ACTION_TO_REWARD[GainMostExpensiveCardAvailable(card)] = _card_reward
    ACTION_TO_REWARD[GainCardInsteadOfMoreExpensiveCard(card)] = _card_reward

print(ACTION_TO_REWARD)

games_df, win_rates = play.play_n_games(
    player_names=["model_chooser", "big_money_provinces_only"],
    choosers=[Chooser(strategies.big_money_until_province_then_all_victory),
              Chooser(strategies.big_money_until_province_then_all_victory)],
    n=1,
    action_to_reward=ACTION_TO_REWARD)

print(games_df.to_string())