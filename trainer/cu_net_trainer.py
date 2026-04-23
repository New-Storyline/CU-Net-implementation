
from model_arcs.geo_features import *
from model_arcs.unet_base import UNetWithGeoWrapper as UNet
from utils.metrics import *
import lightning as L

# class CU_NetTrainer(L.LightningModule)

class LitUNet(L.LightningModule):
    """
    Trainer for testing the UNet architecture
    """
    def __init__(self, image_size, learning_rate=0.001):
        super().__init__()
        
        self.save_hyperparameters("learning_rate")

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
        return loss
    