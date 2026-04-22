
from ..model_arcs.unet_base import UNetBase as UNet
import lightning as L

# class CU_NetTrainer(L.LightningModule)

class UNetTrainer(L.LightningModule):
    """
    Trainer for testing the UNet architecture
    """
    def __init__(self):

        self.model = UNet(in_channels=16, out_channels=1, geoplanes=3)