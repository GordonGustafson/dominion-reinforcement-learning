from cards import *
from actions import *

import unittest

from game import buy_phase_choices, Choice


class TestCards(unittest.TestCase):
    def test_money_from_treasures(self):
        cases = [
            ({"copper": 1, "silver": 1, "gold": 1}, 6),
            ({"silver": 2, "gold": 3}, 13),
            ({"copper": 8, "estate": 1, "duchy": 2, "province": 3}, 8),
            ({"estate": 1}, 0),
            ({}, 0),
        ]
        for (card_names_dict, expected_treasure_total) in cases:
            card_counts = dict_to_card_counts(card_names_dict)
            self.assertEqual(money_from_treasures(card_counts), expected_treasure_total)

    def test_vp_total(self):
        cases = [
            ({"copper": 1, "silver": 1, "gold": 1}, 0),
            ({"silver": 2, "gold": 3}, 0),
            ({"copper": 8, "estate": 1, "duchy": 2, "province": 3}, 25),
            ({"estate": 1}, 1),
            ({}, 0),
        ]
        for (card_names_dict, expected_vp_total) in cases:
            card_counts = dict_to_card_counts(card_names_dict)
            self.assertEqual(vp_total(card_counts), expected_vp_total)

    def test_game_state_equals(self):
        game_state         = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 3}))])
        different_cleanup  = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 3}))], turn_phase=TurnPhase.CLEANUP)
        different_supply   = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 3}))], supply=dict_to_card_counts({"province": 9}))
        different_index    = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 3}))], current_player_index=1)
        different_effects  = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 3}))], pending_effects=(Effect(EffectName.PRODUCE_MONEY, 1)))
        different_actions  = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 3}))], actions=2)
        different_hand     = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 9}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 3}))],)
        different_deck     = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 9}), discard_pile=dict_to_card_counts({"silver": 3}))],)
        different_discard  = make_game_state(players=[make_player(hand=dict_to_card_counts({"copper": 1}), deck=dict_to_card_counts({"estate": 2}), discard_pile=dict_to_card_counts({"silver": 9}))],)


        self.assertEqual(game_state, game_state)
        self.assertNotEqual(game_state, different_cleanup)
        self.assertNotEqual(game_state, different_supply)
        self.assertNotEqual(game_state, different_index)
        self.assertNotEqual(game_state, different_effects)
        self.assertNotEqual(game_state, different_actions)
        self.assertNotEqual(game_state, different_hand)
        self.assertNotEqual(game_state, different_deck)
        self.assertNotEqual(game_state, different_discard)

    def test_buy_phase_choices(self):
        buy_game_state = make_game_state(turn_phase=TurnPhase.BUY,
                                         supply=dict_to_card_counts({"copper": 1, "silver": 1, "gold": 1, "estate": 0, "duchy": 1, "province": 1}),
                                         total_money=5,
                                         buys=2,
                                         players=[make_player(hand=dict_to_card_counts({"silver": 1, "gold": 1}),
                                                              deck=dict_to_card_counts({"copper": 1}),
                                                              discard_pile=dict_to_card_counts({"estate": 1}))])

        discard_pile = buy_game_state.current_player().discard_pile
        cleanup_game_state = buy_game_state._replace(turn_phase=TurnPhase.CLEANUP)
        expected_choices = [
            Choice(cleanup_game_state, GainNothing()),
            Choice(buy_game_state._replace(supply=remove_card_by_name(cleanup_game_state.supply, "copper"), total_money=5, buys=1).replace_current_player_kwargs(discard_pile=add_card_by_name(discard_pile, "copper")), GainCard(card_name_to_card("copper"))),
            Choice(buy_game_state._replace(supply=remove_card_by_name(cleanup_game_state.supply, "silver"), total_money=2, buys=1).replace_current_player_kwargs(discard_pile=add_card_by_name(discard_pile, "silver")), GainCard(card_name_to_card("silver"))),
            Choice(buy_game_state._replace(supply=remove_card_by_name(cleanup_game_state.supply, "duchy"), total_money=0, buys=1).replace_current_player_kwargs(discard_pile=add_card_by_name(discard_pile, "duchy")), GainCard(card_name_to_card("duchy"))),
            # Can't buy estates because the supply pile is empty
        ]
        self.assertEqual(buy_phase_choices(buy_game_state), expected_choices)

    def test_draw_card(self):
        player = make_player(hand=dict_to_card_counts({"silver": 1}),
                             deck=dict_to_card_counts({"copper": 1, "gold": 1}),
                             top_of_deck=(card_name_to_card("curse"),),
                             discard_pile=dict_to_card_counts({"estate": 1}))

        exp_after_one_draw = make_player(hand=dict_to_card_counts({"silver": 1, "curse": 1}),
                                         deck=dict_to_card_counts({"copper": 1, "gold": 1}),
                                         top_of_deck=(),
                                         discard_pile=dict_to_card_counts({"estate": 1}))

        possibilities_after_two_draws = [make_player(hand=dict_to_card_counts({"copper": 1, "silver": 1, "curse": 1}),
                                                    deck=dict_to_card_counts({"gold": 1}),
                                                    discard_pile=dict_to_card_counts({"estate": 1})),
                                        make_player(hand=dict_to_card_counts({"gold": 1, "silver": 1, "curse": 1}),
                                                    deck=dict_to_card_counts({"copper": 1}),
                                                    discard_pile=dict_to_card_counts({"estate": 1}))]

        exp_after_three_draws = make_player(hand=dict_to_card_counts({"silver": 1, "gold": 1, "copper": 1, "curse": 1}),
                                          deck=dict_to_card_counts({}),
                                          discard_pile=dict_to_card_counts({"estate": 1}))

        exp_after_four_draws = make_player(hand=dict_to_card_counts({"silver": 1, "gold": 1, "copper": 1, "estate": 1, "curse": 1}),
                                            deck=dict_to_card_counts({}),
                                            discard_pile=dict_to_card_counts({}))

        self.assertEqual(draw_card(player), exp_after_one_draw)
        self.assertTrue(draw_card(draw_card(player)) in possibilities_after_two_draws)
        self.assertEqual(draw_card(draw_card(draw_card(player))), exp_after_three_draws)
        self.assertEqual(draw_card(draw_card(draw_card(draw_card(player)))), exp_after_four_draws)
        # No change when top_of_deck, deck, and discard pile are all empty
        self.assertEqual(draw_card(draw_card(draw_card(draw_card(draw_card(player))))), exp_after_four_draws)

    def test_do_cleanup_phase(self):
        game_state = make_game_state(turn_phase=TurnPhase.CLEANUP,
                                     current_player_index=1,
                                     max_turns_per_player=0,
                                     players=[make_player(hand=dict_to_card_counts({"copper": 5}),
                                                          deck=dict_to_card_counts({"estate": 2}),
                                                          discard_pile=dict_to_card_counts({"copper": 2})),
                                              make_player(hand=dict_to_card_counts({"copper": 1}),
                                                          deck=dict_to_card_counts({"estate": 1}),
                                                          discard_pile=dict_to_card_counts({"copper": 1})),
                                              ])

        game_state_after_cleanup = make_game_state(turn_phase=TurnPhase.ACTION,
                                                   pending_effects=(),
                                                   current_player_index=0,
                                                   max_turns_per_player=1,
                                                   players=[make_player(hand=dict_to_card_counts({"copper": 5}),
                                                                        deck=dict_to_card_counts({"estate": 2}),
                                                                        discard_pile=dict_to_card_counts({"copper": 2})),
                                                            make_player(hand=dict_to_card_counts({"copper": 2, "estate": 1}),
                                                                        deck=dict_to_card_counts({}),
                                                                        discard_pile=dict_to_card_counts({})),
                                                            ])

        self.assertEqual(do_cleanup_phase(game_state), game_state_after_cleanup)


if __name__ == '__main__':
    unittest.main()
