from actions import _ACTIONS_LIST, action_to_action_id, action_id_to_action

import unittest


class TestCards(unittest.TestCase):
    def test_action_to_action_id(self):
        for action in _ACTIONS_LIST:
            self.assertEqual(action, action_id_to_action(action_to_action_id(action)))
