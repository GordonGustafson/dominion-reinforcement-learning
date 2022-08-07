from cards import *

import unittest


class TestCards(unittest.TestCase):
    def test_treasure_total(self):
        cases = [
            ({"copper": 1, "silver": 1, "gold": 1}, 6),
            ({"silver": 2, "gold": 3}, 13),
            ({"copper": 8, "estate": 1, "duchy": 2, "province": 3}, 8),
            ({"estate": 1}, 0),
            ({}, 0),
        ]
        for (card_names_dict, expected_treasure_total) in cases:
            card_counts = dict_to_card_counts(card_names_dict)
            self.assertTrue(treasure_total(card_counts) == expected_treasure_total)

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
            self.assertTrue(vp_total(card_counts) == expected_vp_total)

    def test_buy_phase_options(self):
        buy_game_state = GameState(cleanup_phase=False,
                                   hand=dict_to_card_counts({"silver": 1, "gold": 1}),
                                   deck=dict_to_card_counts({"copper": 1}),
                                   discard_pile=dict_to_card_counts({"estate": 1}))

        discard_pile = buy_game_state.discard_pile
        cleanup_game_state = buy_game_state._replace(cleanup_phase=True)
        expected_options = [
            cleanup_game_state,
            cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "copper")),
            cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "silver")),
            cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "estate")),
            cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "duchy")),
        ]
        self.assertTrue(buy_phase_options(buy_game_state) == expected_options)



if __name__ == '__main__':
    unittest.main()
