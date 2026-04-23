from dataloaders.custom_dataset import CustomDepthDataset, DataElement

IMAGE_SIZE = (480, 640)

def unet_training():
    dataset_train = CustomDepthDataset(
        paths=,
        image_size=IMAGE_SIZE,
        use_image=False
    )

    trainer = UNetTrainer()
    # trainer.fit(dataset)