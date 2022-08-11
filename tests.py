from cards import *

import unittest


def dict_to_card_counts(card_names_dict):
    return np.array([card_names_dict.get(card_name, 0) for card_name in CARD_DEFS['name'].to_list()])

def add_card_by_name(card_counts: CardCounts, card_name: str) -> CardCounts:
    card_index = CARD_DEFS.index[CARD_DEFS['name'] == card_name].item()
    return add_card(card_counts, card_index)

def card_counts_equal(lhs: CardCounts, rhs: CardCounts) -> bool:
    return np.array_equal(lhs, rhs)

def game_states_equal(lhs: GameState, rhs: GameState) -> bool:
    return (lhs.cleanup_phase == rhs.cleanup_phase
            and card_counts_equal(lhs.hand, rhs.hand)
            and card_counts_equal(lhs.deck, rhs.deck)
            and card_counts_equal(lhs.discard_pile, rhs.discard_pile))

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
            Action(cleanup_game_state, "buy nothing"),
            Action(cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "copper")), "buy copper"),
            Action(cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "silver")), "buy silver"),
            Action(cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "estate")), "buy estate"),
            Action(cleanup_game_state._replace(discard_pile=add_card_by_name(discard_pile, "duchy")), "buy duchy"),
        ]
        self.assertTrue(buy_phase_options(buy_game_state) == expected_options)

    def test_draw_card(self):
        game_state = GameState(cleanup_phase=False,
                               hand=dict_to_card_counts({"silver": 1}),
                               deck=dict_to_card_counts({"copper": 1, "gold": 1}),
                               discard_pile=dict_to_card_counts({"estate": 1}))

        possibilities_after_one_draw = [GameState(cleanup_phase=False,
                                                  hand=dict_to_card_counts({"copper": 1, "silver": 1}),
                                                  deck=dict_to_card_counts({"gold": 1}),
                                                  discard_pile=dict_to_card_counts({"estate": 1})),
                                        GameState(cleanup_phase=False,
                                                  hand=dict_to_card_counts({"gold": 1, "silver": 1}),
                                                  deck=dict_to_card_counts({"copper": 1}),
                                                  discard_pile=dict_to_card_counts({"estate": 1}))]

        exp_after_two_draws = GameState(cleanup_phase=False,
                                              hand=dict_to_card_counts({"silver": 1, "gold": 1, "copper": 1}),
                                              deck=dict_to_card_counts({}),
                                              discard_pile=dict_to_card_counts({"estate": 1}))

        exp_after_three_draws = GameState(cleanup_phase=False,
                                               hand=dict_to_card_counts({"silver": 1, "gold": 1, "copper": 1, "estate": 1}),
                                               deck=dict_to_card_counts({}),
                                               discard_pile=dict_to_card_counts({}))


        self.assertTrue(draw_card(game_state) in possibilities_after_one_draw)
        self.assertTrue(draw_card(draw_card(game_state)) == exp_after_two_draws)
        # No change when deck and discard pile are both empty
        self.assertTrue(draw_card(draw_card(draw_card(game_state))) == exp_after_three_draws)
        self.assertTrue(draw_card(draw_card(draw_card(draw_card(game_state)))) == exp_after_three_draws)


if __name__ == '__main__':
    unittest.main()
