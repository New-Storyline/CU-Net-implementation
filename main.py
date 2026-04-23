import torch
from functools import partial

import lightning as L
from torch.utils.data import DataLoader
from dataloaders.custom_dataset import CustomDepthDataset
from utils.nyu_dataset_utils import get_train_pathes, get_val_pathes
from utils.data_utils import create_sparse_depth
from trainer.cu_net_trainer import LitUNet

IMAGE_SIZE = (480, 640)
DATASET_ROOT = "datasets"
SPARSE_DEPTH_NUM_POINTS = 100 * 1000


def identity_transform(sparse_depth, gt_depth, rgb, position):
    return sparse_depth, gt_depth, rgb, position

def unet_training():
    dataset_train = CustomDepthDataset(
        image_size=IMAGE_SIZE,
        get_image_pathes_fn=partial(get_train_pathes, DATASET_ROOT),
        transform_fn=identity_transform,
        create_sparse_depth_fn=partial(create_sparse_depth, num_points=SPARSE_DEPTH_NUM_POINTS),
        load_calib_fn=None,
        use_image=True,
    )

    dataset_val = CustomDepthDataset(
        image_size=IMAGE_SIZE,
        get_image_pathes_fn=partial(get_val_pathes, DATASET_ROOT),
        transform_fn=identity_transform,
        create_sparse_depth_fn=partial(create_sparse_depth, num_points=SPARSE_DEPTH_NUM_POINTS),
        load_calib_fn=None,
        use_image=True,
    )

    batch_size = 8
    num_workers = 4

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
        max_epochs=20,
        #limit_train_batches=0.01,
        precision=16,  # Use mixed precision for faster training and reduced memory usage
    )
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

if __name__ == "__main__":
    unet_training()
