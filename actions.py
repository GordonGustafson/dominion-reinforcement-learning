from dataclasses import dataclass

import typing
from cards import Card, CARD_LIST


@dataclass(frozen=True)
class GainMostExpensiveCardAvailable:
    card: Card

    def get_description(self) -> str:
        return f"gain {self.card.name}"

@dataclass(frozen=True)
class GainCardInsteadOfMoreExpensiveCard:
    card: Card

    def get_description(self) -> str:
        return f"gain {self.card.name} instead of more expensive card"


@dataclass(frozen=True)
class GainNothing:

    def get_description(self) -> str:
        return f"gain nothing"


@dataclass(frozen=True)
class GainCardToHand:
    card: Card

    def get_description(self) -> str:
        return f"gain {self.card.name} to hand"


@dataclass(frozen=True)
class PlayActionCard:
    card: Card

    def get_description(self) -> str:
        return f"play {self.card.name}"


@dataclass(frozen=True)
class PlayNoActionCard:

    def get_description(self) -> str:
        return f"play no action card"


@dataclass(frozen=True)
class DiscardCard:
    card: Card

    def get_description(self) -> str:
        return f"discard {self.card.name}"


@dataclass(frozen=True)
class DiscardCardToDrawACard:
    card: Card

    def get_description(self) -> str:
        return f"discard {self.card.name} to draw a card"


@dataclass(frozen=True)
class DontDiscardCardToDrawACard:
    card: Card

    def get_description(self) -> str:
        return f"don't discard {self.card.name} to draw a card"


@dataclass(frozen=True)
class PutCardFromDiscardPileOntoDeck:
    card: Card

    def get_description(self) -> str:
        return f"Put {self.card.name} from discard pile onto deck"


@dataclass(frozen=True)
class PutNoCardFromDiscardPileOntoDeck:

    def get_description(self) -> str:
        return f"Put no card from discard pile onto deck"


@dataclass(frozen=True)
class TrashCardFromHand:
    card: Card

    def get_description(self) -> str:
        return f"trash {self.card.name} from hand"


@dataclass(frozen=True)
class TrashNoCardFromHand:

    def get_description(self) -> str:
        return f"trash no card from hand"


@dataclass(frozen=True)
class TrashRevealedCard:
    card: Card

    def get_description(self) -> str:
        return f"trash revealed {self.card.name}"


@dataclass(frozen=True)
class TrashCardFromHandToGainCardCostingUpTo2More:
    card: Card

    def get_description(self) -> str:
        return f"trash {self.card.name} from hand to gain a card costing up to 2 more."


@dataclass(frozen=True)
class TrashTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More:
    card: Card

    def get_description(self) -> str:
        return f"trash {self.card.name} from hand to gain a treasure card to hand costing up to 3 more."


@dataclass(frozen=True)
class TrashNoTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More:

    def get_description(self) -> str:
        return f"trash no card from hand to gain a treasure card to hand costing up to 3 more."


@dataclass(frozen=True)
class TrashACopperFor3Money:

    def get_description(self) -> str:
        return f"trash a copper for 3 money"


@dataclass(frozen=True)
class DontTrashACopperFor3Money:

    def get_description(self) -> str:
        return f"do not trash a copper for 3 money"


# Modeling the treasure phase is out of scope right now, so we use a single dummy action
# for the treasure phase.
@dataclass(frozen=True)
class PlayAllTreasures:

    def get_description(self) -> str:
        return f"play all treasures"

# We're modeling gaining and buying cards as the same action for now for better generalization.
_ACTION_TYPES_WITH_ANY_CARD_PARAMETER = [
    GainMostExpensiveCardAvailable,
    GainCardInsteadOfMoreExpensiveCard,
#    GainCardToHand,
#    DiscardCard,
#    DiscardCardToDrawACard,
#    DontDiscardCardToDrawACard,
#    PutCardFromDiscardPileOntoDeck,
#    TrashCardFromHand,
#    TrashRevealedCard,
#    TrashCardFromHandToGainCardCostingUpTo2More,
]

_ACTION_TYPES_WITH_ACTION_CARD_PARAMETER = [
    PlayActionCard,
]

_ACTION_TYPES_WITH_TREASURE_CARD_PARAMETER = [
#    TrashTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More,
]

_ACTION_TYPES_WITHOUT_CARD_PARAMETER = [
    GainNothing,
    PlayNoActionCard,
#    PutNoCardFromDiscardPileOntoDeck,
#    TrashNoCardFromHand,
#    TrashNoTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More,
#    TrashACopperFor3Money,
#    DontTrashACopperFor3Money,
#    PlayAllTreasures,
]

Action = typing.Union[*(_ACTION_TYPES_WITH_ANY_CARD_PARAMETER
                        + _ACTION_TYPES_WITH_ACTION_CARD_PARAMETER
                        + _ACTION_TYPES_WITH_TREASURE_CARD_PARAMETER
                        + _ACTION_TYPES_WITHOUT_CARD_PARAMETER)]

_ACTIONS_LIST = ([action_type(card)
                  for action_type in _ACTION_TYPES_WITH_ANY_CARD_PARAMETER
                  for card in CARD_LIST] +
                 [action_type(card)
                  for action_type in _ACTION_TYPES_WITH_ACTION_CARD_PARAMETER
                  for card in CARD_LIST if len(card.action_effects) > 0] +
                 [action_type(card)
                  for action_type in _ACTION_TYPES_WITH_TREASURE_CARD_PARAMETER
                  for card in CARD_LIST if len(card.treasure_effects) > 0] +
                 [action_type()
                  for action_type in _ACTION_TYPES_WITHOUT_CARD_PARAMETER])

print(_ACTIONS_LIST)

_ACTION_TO_ACTION_ID = {action: action_id for action_id, action in enumerate(_ACTIONS_LIST)}

NUM_ACTIONS = len(_ACTIONS_LIST)

def action_to_action_id(action: Action):
    return _ACTION_TO_ACTION_ID[action]

def action_id_to_action(action_id: int):
    if action_id < 0:
        raise ValueError(f"Invalid action_id: {action_id}")
    return _ACTIONS_LIST[action_id]
