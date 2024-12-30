import pandas as pd
from torch.utils.data import Dataset

import torch

class DominionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.feature_dataframe = dataframe.drop(columns=["reward"])
        # TODO: Google right way to do this
        self.reward_dataframe = dataframe.drop(columns=["player_vp_lead", "num_provinces_remaining", "average_treasure_value_self", "average_treasure_value_opponent"])

    def __len__(self) -> int:
        return len(self.feature_dataframe)

    def __getitem__(self, i: int):
        features = self.feature_dataframe.iloc[i]
        reward = self.reward_dataframe.iloc[i]

        return features, reward

def tensorify_dataframe(df: pd.DataFrame) -> torch.tensor:
    return torch.tensor(df.to_numpy().reshape((-1, 4)))

def collate_fn(list_of_pairs_of_dataframes):
    pair_of_list_data_frames = list(zip(*list_of_pairs_of_dataframes))
    merged_features_df = pd.concat(pair_of_list_data_frames[0], axis="index", ignore_index=True)
    merged_reward_df = pd.concat(pair_of_list_data_frames[1], axis="index", ignore_index=True)

    # TODO: Why is the shape 1-dimensional here until I manually reshape it?
    return tensorify_dataframe(merged_features_df), torch.tensor(merged_reward_df)
