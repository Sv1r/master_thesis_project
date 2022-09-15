import torch

import utils
import settings
import transforms

# Get train/valid DataFrames
train, valid = utils.data_preparation(settings.DATA_FOLDER)
# Datasets
train_dataset = utils.StressDataset(train, transform=transforms.train_aug)
valid_dataset = utils.StressDataset(train, transform=transforms.valid_aug)
# Dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

if __name__ == '__main__':
    # Check data shape for image-mask
    print(f'Image shape:\n{list(train_dataset[0][0].shape)}')
    print(f'Mask shape:\n{list(train_dataset[0][1].shape)}')
    # Check train-valid size
    print(f'Train dataset length: {train_dataset.__len__()}')
    print(f'Valid dataset length: {valid_dataset.__len__()}\n')
    # Plot data
    utils.image_show_tensor(train_dataloader)


