import pandas as pd
from torch.utils.data import Dataset

import torch

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
    return torch.tensor(df[["player_vp_lead",
                            "num_provinces_remaining",
                            "average_treasure_value_self",
                            "average_treasure_value_opponent",
                            "max_turns_per_player"
                            ]].to_numpy().reshape((-1, 5)))

def tensorify_reward(df: pd.DataFrame) -> torch.tensor:
    return torch.tensor(df[["reward"]].to_numpy().squeeze())

def collate_fn(list_of_pairs_of_dataframes):
    pair_of_list_data_frames = list(zip(*list_of_pairs_of_dataframes))
    merged_features_df = pd.concat(pair_of_list_data_frames[0], axis="index")
    merged_reward_df = pd.concat(pair_of_list_data_frames[1], axis="index")

    return tensorify_inputs(merged_features_df), tensorify_reward(merged_reward_df)
