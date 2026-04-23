from dataloaders.custom_dataset import CustomDepthDataset, DataElement, test_dataset
from utils.nyu_dataset_utils import get_train_pathes,get_val_pathes
from utils.data_utils import create_sparse_depth

IMAGE_SIZE = (480, 640)

def unet_training():
    dataset_train = CustomDepthDataset(
        image_size=IMAGE_SIZE,
        get_image_pathes_fn=lambda: get_train_pathes('datasets'),
        transform_fn=lambda sparse, gt, rgb, pos: (sparse, gt, rgb, pos),
        create_sparse_depth_fn=lambda gt_depth: create_sparse_depth(gt_depth, num_points=100 * 1000),  # Placeholder for actual sparse depth creation
        load_calib_fn=None,
        use_image=True,
    )

    dataset_val = CustomDepthDataset(
        image_size=IMAGE_SIZE,
        get_image_pathes_fn=lambda: get_val_pathes('datasets'),
        transform_fn=lambda sparse, gt, rgb, pos: (sparse, gt, rgb, pos),
        create_sparse_depth_fn=lambda gt_depth: create_sparse_depth(gt_depth, num_points=100 * 1000),  # Placeholder for actual sparse depth creation
        load_calib_fn=None,
        use_image=True,
    )

    dataset_train.visualize_first_sample()
    dataset_val.visualize_first_sample()

    #trainer = UNetTrainer()
    # trainer.fit(dataset)

if __name__ == "__main__":
    unet_training()