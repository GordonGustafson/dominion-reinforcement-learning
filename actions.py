from dataclasses import dataclass

from cards import Card


@dataclass(frozen=True)
class GainCard:
    card: Card

    def get_description(self) -> str:
        return f"gain {self.card.name}"


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
class DoNotTrashACopperFor3Money:

    def get_description(self) -> str:
        return f"do not trash a copper for 3 money"


# Modeling the treasure phase is out of scope right now, so we use a single dummy action
# for the treasure phase.
@dataclass(frozen=True)
class PlayAllTreasures:

    def get_description(self) -> str:
        return f"play all treasures"


# We're modeling gaining and buying cards as the same action for now for better generalization.
Action = (GainCard | GainNothing
          | GainCardToHand
          | PlayActionCard | PlayNoActionCard
          | DiscardCard
          | DiscardCardToDrawACard | DontDiscardCardToDrawACard
          | PutCardFromDiscardPileOntoDeck | PutNoCardFromDiscardPileOntoDeck
          | TrashCardFromHand | TrashNoCardFromHand
          | TrashRevealedCard
          | TrashCardFromHandToGainCardCostingUpTo2More
          | TrashTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More | TrashNoTreasureCardFromHandToGainTreasureCardToHandCostingUpTo3More
          | TrashACopperFor3Money | DoNotTrashACopperFor3Money
          | PlayAllTreasures)
