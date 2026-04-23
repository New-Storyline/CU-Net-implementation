
from ..model_arcs.geo_features import *
from ..model_arcs.unet_base import UNetWithGeoWrapper as UNet
import lightning as L

# class CU_NetTrainer(L.LightningModule)

class UNetTrainer(L.LightningModule):
    """
    Trainer for testing the UNet architecture
    """
    def __init__(self):

        self.model = UNet(
            in_channels=1, 
            out_channels=1, 
            geo_encoding_type=GeoEncodingType.Z,
            img_size=(256, 256)
        )
    
    def training_step(self, batch, batch_idx):
        sparse_depth, positions_map, gt_depth = batch
        pred_depth = self.model(sparse_depth, positions_map)
        loss = self.model.loss(pred_depth, gt_depth)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sparse_depth, positions_map, gt_depth = batch
        pred_depth = self.model(sparse_depth, positions_map)
        loss = self.model.loss(pred_depth, gt_depth)
        self.log('val_loss', loss)
        return loss
    