
from model_arcs.geo_features import *
from model_arcs.unet_base import UNetWithGeoWrapper as UNet
from utils.metrics import *
import lightning as L
import matplotlib.pyplot as plt
# class CU_NetTrainer(L.LightningModule)

class LitUNet(L.LightningModule):
    """
    Trainer for testing the UNet architecture
    """
    def __init__(self, image_size, learning_rate=0.001):
        super().__init__()
        
        self.save_hyperparameters("learning_rate")
        self._val_preview = None

        self.model = UNet(
            in_channels=1, 
            unet_channels=64,
            out_channels=1, 
            geo_encoding_type=GeoEncodingType.STD,
            img_size=image_size
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch, batch_idx):
        sparse_depth = batch['sparse_depth']
        positions_map = batch['position']
        gt_depth = batch['gt_depth']
        pred_depth = self.model(sparse_depth, positions_map)
        loss = L2_depth_loss(pred_depth, gt_depth)
        rmse = RMSE_depth_metric(pred_depth, gt_depth)
        self.log('train_loss', loss)
        self.log('train_rmse', rmse)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sparse_depth = batch['sparse_depth']
        positions_map = batch['position']
        gt_depth = batch['gt_depth']
        pred_depth = self.model(sparse_depth, positions_map)
        loss = L2_depth_loss(pred_depth, gt_depth)
        rmse = RMSE_depth_metric(pred_depth, gt_depth)
        self.log('val_loss', loss)
        self.log('val_rmse', rmse)

        if batch_idx == 0:
            self._val_preview = {
                "sparse": sparse_depth[0].detach().cpu(),
                "pred": pred_depth[0].detach().cpu(),
                "gt": gt_depth[0].detach().cpu(),
            }

        return loss
    
    def on_validation_epoch_end(self):
        if self._val_preview is None:
            return

        if not hasattr(self.logger, "experiment"):
            return

        sparse = self._val_preview["sparse"].squeeze().numpy()
        pred = self._val_preview["pred"].squeeze().numpy()
        gt = self._val_preview["gt"].squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(sparse, cmap="viridis")
        axes[0].set_title("Sparse depth")
        axes[1].imshow(pred, cmap="viridis")
        axes[1].set_title("Prediction")
        axes[2].imshow(gt, cmap="viridis")
        axes[2].set_title("Ground truth")

        for ax in axes:
            ax.axis("off")

        self.logger.experiment.add_figure(
            "val/examples",
            fig,
            global_step=self.current_epoch,
        )
        plt.close(fig)

        self._val_preview = None