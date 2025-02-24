import pandas as pd
from torch.utils.data import Dataset

from cards import card_name_to_card, CARD_LIST

import torch

NUM_INPUT_FEATURES = 22

class DominionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.feature_dataframe = dataframe.drop(columns=["reward"])
        self.reward_dataframe = dataframe[["reward"]]

    def __len__(self) -> int:
        return len(self.feature_dataframe)

    def __getitem__(self, i: int):
        features = self.feature_dataframe.iloc[i]
        rewards = self.reward_dataframe.iloc[i]

        return features, rewards

def tensorify_inputs(df: pd.DataFrame) -> torch.tensor:
    # Index the DF to make sure the columns show up in the right order? I'm not sure how this is typically done.
    # TODO: Why is the shape 1-dimensional here until I manually reshape it?

    exclude_num_card_owned = {card_name_to_card(card_name) for card_name in ["estate", "duchy", "province", "curse"]}
    num_card_owned_feature_names = [f"num_{card.name}_owned_self" for card in sorted(set(CARD_LIST) - exclude_num_card_owned)]

    feature_names = [
        "max_turns_per_player",
        "num_provinces_remaining",
        "two_provinces_remaining",
        "one_province_remaining",
        "player_vp_lead",
        "num_vp_self",
        "average_treasure_value_self",
        "num_victory_cards_owned_self",
        "num_actions_owned_with_plus_zero_actions_self",
        "num_actions_owned_with_plus_one_action_self",
        "num_actions_owned_with_plus_two_actions_self",
    ] + num_card_owned_feature_names

    return torch.tensor(df[feature_names].to_numpy().reshape((-1, NUM_INPUT_FEATURES)))

def tensorify_reward(df: pd.DataFrame) -> torch.tensor:
    return torch.tensor(df[["reward"]].to_numpy().squeeze())

def collate_fn(list_of_pairs_of_dataframes):
    pair_of_list_data_frames = list(zip(*list_of_pairs_of_dataframes))
    merged_features_df = pd.concat(pair_of_list_data_frames[0], axis="index")
    merged_reward_df = pd.concat(pair_of_list_data_frames[1], axis="index")

    return tensorify_inputs(merged_features_df), tensorify_reward(merged_reward_df)
