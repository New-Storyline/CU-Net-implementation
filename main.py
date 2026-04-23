from dataloaders.custom_dataset import CustomDepthDataset, DataElement
from dataloaders.nyu_dataset_utils import get_train_pathes,get_val_pathes

IMAGE_SIZE = (480, 640)

def unet_training():
    dataset_train = CustomDepthDataset(
        image_size=IMAGE_SIZE,
        get_image_pathes_fn=lambda: get_train_pathes('datasets'),
        transform_fn=lambda sparse, gt, rgb, pos: (sparse, gt, rgb, pos),
        create_sparse_depth_fn=lambda depth: depth,  # Placeholder for actual sparse depth creation
        load_calib_fn=None,
        use_image=False,
    )

    dataset_val = CustomDepthDataset(
        image_size=IMAGE_SIZE,
        get_image_pathes_fn=lambda: get_val_pathes('datasets'),
        transform_fn=lambda sparse, gt, rgb, pos: (sparse, gt, rgb, pos),
        create_sparse_depth_fn=lambda depth: depth,  # Placeholder for actual sparse depth creation
        load_calib_fn=None,
        use_image=False,
    )

    trainer = UNetTrainer()
    # trainer.fit(dataset)