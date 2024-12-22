import lightning as L
import torch


class DominionModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, batch_idx):
        features, reward = batch
        raw_scores = self.model.forward(features)
        return torch.nn.functional.sigmoid(raw_scores)

    def training_step(self, batch, batch_idx):
        predicted_rewards = self.forward(batch, batch_idx).squeeze(1)
        features, rewards = batch
        # print(f"predicted_rewards: {predicted_rewards}, rewards: {rewards}")
        train_loss = torch.nn.functional.binary_cross_entropy(predicted_rewards, target=rewards)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999))
