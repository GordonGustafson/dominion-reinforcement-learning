import pandas as pd
from torch.utils.data import Dataset

import torch

NUM_INPUT_FEATURES = 16

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
    return torch.tensor(df[[
        "player_vp_lead",
        "num_provinces_remaining",
        "average_treasure_value_self",
        "num_vp_self",
        "num_victory_cards_owned_self",
        "num_copper_owned_self",
        "num_silver_owned_self",
        "num_gold_owned_self",
        "num_smithy_owned_self",
        "num_laboratory_owned_self",
        "num_village_owned_self",
        "num_festival_owned_self",
        "num_market_owned_self",
        #"average_treasure_value_opponent",
        "max_turns_per_player",
        "two_provinces_remaining",
        "one_province_remaining",

        # "zero_copper_owned_self",
        # "zero_silver_owned_self",
        # "zero_gold_owned_self",
        # "zero_smithy_owned_self",
        # "zero_laboratory_owned_self",
        # "zero_village_owned_self",
        # "zero_festival_owned_self",
        # "zero_market_owned_self",

        # "one_copper_owned_self",
        # "one_silver_owned_self",
        # "one_gold_owned_self",
        # "one_smithy_owned_self",
        # "one_laboratory_owned_self",
        # "one_village_owned_self",
        # "one_festival_owned_self",
        # "one_market_owned_self",

        # "two_copper_owned_self",
        # "two_silver_owned_self",
        # "two_gold_owned_self",
        # "two_smithy_owned_self",
        # "two_laboratory_owned_self",
        # "two_village_owned_self",
        # "two_festival_owned_self",
        # "two_market_owned_self",

    ]].to_numpy().reshape((-1, NUM_INPUT_FEATURES)))

def tensorify_reward(df: pd.DataFrame) -> torch.tensor:
    return torch.tensor(df[["reward"]].to_numpy().squeeze())

def collate_fn(list_of_pairs_of_dataframes):
    pair_of_list_data_frames = list(zip(*list_of_pairs_of_dataframes))
    merged_features_df = pd.concat(pair_of_list_data_frames[0], axis="index")
    merged_reward_df = pd.concat(pair_of_list_data_frames[1], axis="index")

    return tensorify_inputs(merged_features_df), tensorify_reward(merged_reward_df)
