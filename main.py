import os
import torch
from functools import partial

import lightning as L
from torch.utils.data import DataLoader
from dataloaders.depth_from_files_dataset import DepthFromFilesDataset
from dataloaders.depth_from_mat_dataset import DepthFromMatDataset
from utils.nyu_dataset_utils import get_train_pathes, get_val_pathes
from utils.data_utils import create_sparse_depth
from trainer.cu_net_trainer import LitUNet

IMAGE_SIZE = (480, 640)
DATASET_ROOT = "datasets"
SPARSE_DEPTH_DROP_RATE = 0.9
DATASET_MAT_FILE = "datasets/nyu_labled (depthes in float16)/nyu_depth_v2_labeled.mat"

def identity_transform(sparse_depth, gt_depth, rgb, position):
    return sparse_depth, gt_depth, rgb, position

def unet_training():
    use_image = False
    
    dataset_train, dataset_val = DepthFromMatDataset.create_train_val_datasets_from_mat(
        mat_path=DATASET_MAT_FILE,
        gt_depth_key="depths",
        rgb_key="images" if use_image else None,
        sparse_depth_key=None,
        create_sparse_depth_fn=partial(create_sparse_depth, drop_rate=SPARSE_DEPTH_DROP_RATE),
        use_image=use_image,
        image_size=(480, 640),
    )

    checkpoint_path = "lightning_logs/version_1/checkpoints/epoch=20-step=3045.ckpt"
    batch_size = 8
    # On Windows, DataLoader workers use spawn and duplicate this in-memory dataset.
    num_workers = 2 #0 if os.name == "nt" else 4
    
    """
    w = 6 -> 2.7 iter/s
    w = 4 -> 2.8 iter/s (optimal)
    w = 2 -> 1.9 iter/s
    """

    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )

    #dataset_train.visualize_first_sample()
    #dataset_val.visualize_first_sample()

    model = LitUNet(image_size=IMAGE_SIZE, learning_rate=0.001)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        #limit_train_batches=0.01,
        precision='16-mixed',  # Use mixed precision for faster training and reduced memory usage
    )
    trainer.fit(
        model=model, 
        train_dataloaders=dataloader_train, 
        val_dataloaders=dataloader_val,
        ckpt_path=checkpoint_path
    )

if __name__ == "__main__":
    unet_training()
    #DepthFromMatDataset.print_mat_file_keys("datasets/nyu_labled (depthes in float16)/nyu_depth_v2_labeled.mat")
    #DepthFromMatDataset.test()
