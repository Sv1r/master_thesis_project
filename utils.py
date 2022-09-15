import cv2
import tqdm
import time
import glob
import torch
import typing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import segmentation_models_pytorch
from sklearn.model_selection import train_test_split

import settings

sns.set_style('darkgrid')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_preparation(data_folder: str) -> typing.Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    """
    Read data from folder and create Pandas DataFrames for train/valid subsets.
    """
    # Read specific folders
    geometry_list = sorted(glob.glob(rf'{data_folder}\geom\*'))
    stress_list = sorted(glob.glob(rf'{data_folder}\stress\*'))
    results_list = sorted(glob.glob(rf'{data_folder}\*\*.txt'))
    # Geometry
    df_geometry = pd.DataFrame()
    df_geometry['geometry'] = geometry_list
    df_geometry['name'] = df_geometry['geometry'].apply(
        lambda x: x.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
    )
    # Stress
    df_stress = pd.DataFrame()
    df_stress['stress'] = stress_list
    df_stress['name'] = df_stress['stress'].apply(
        lambda x: x.split(sep='\\')[-1].split(sep='.')[0].split(sep='_')[0]
    )
    # Results
    df_results = pd.DataFrame()
    df_results['results'] = results_list
    df_results['name'] = df_results['results'].apply(
        lambda x: x.split(sep='\\')[-1].split(sep='.')[0].split(sep='-')[-1]
    )
    # Merge dataframes on inner type by name column
    df = df_geometry.merge(df_stress, on='name', how='inner').merge(df_results, on='name', how='inner')
    # Split data on train/valid
    train, valid = train_test_split(
        df,
        train_size=.9,
        random_state=42,
        shuffle=True
    )
    return train, valid


class StressDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):
        self.geometry_files = data['geometry'].tolist()
        self.stress_files = data['stress'].tolist()
        self.results_files = data['results'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.geometry_files)

    def __getitem__(self, index):
        geometry_path = self.geometry_files[index]
        stress_path = self.stress_files[index]
        result_path = self.results_files[index]

        # min_stress, max_stress = np.loadtxt(result_path)

        geometry = cv2.imread(geometry_path, cv2.IMREAD_GRAYSCALE)
        geometry = geometry[17:678, 331:994]
        geometry[geometry == 0] = np.unique(geometry)[-1]
        geometry = cv2.resize(geometry, (settings.IMAGE_SIZE, settings.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        geometry = cv2.threshold(geometry, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        geometry = geometry.astype(np.uint8)
        # geometry = np.expand_dims(geometry, axis=-1)
        geometry = np.stack([geometry, geometry, geometry], axis=-1)

        stress = cv2.imread(stress_path, cv2.IMREAD_GRAYSCALE)
        stress = stress[103:553, 499:951]
        stress = cv2.resize(stress, (settings.IMAGE_SIZE, settings.IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        stress = stress / 255
        # stress = cv2.normalize(stress, None, alpha=min_stress, beta=max_stress, norm_type=cv2.NORM_MINMAX)
        stress = np.expand_dims(stress, axis=-1)
        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=geometry, mask=stress)
            geometry = aug['image']
            stress = aug['mask']

        return geometry, stress


def image_show_tensor(dataloader, number_of_images=5, initial_index=0, values_name=None):
    geometry, stress = next(iter(dataloader))
    geometry = geometry.numpy().transpose(0, 2, 3, 1)
    geometry = np.array(settings.STD) * geometry + np.array(settings.MEAN)
    stress = stress.numpy().transpose(0, 2, 3, 1)

    fig, ax = plt.subplots(2, number_of_images, figsize=(10, 10))
    for i in range(number_of_images):
        one_geom = geometry[i]
        ax[0, i].imshow(one_geom)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[0, i].set_title(f'Geometry Inclusion={int(one_geom.min()):.0f}; Matrix={int(one_geom.max()):.0f}')

        one_stress = stress[i]
        ax[1, i].imshow(one_stress, cmap='jet')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[1, i].set_title(f'von Mises Stress Min={one_stress.min():.2f}; Max={one_stress.max():.2f}')

    fig.tight_layout()
    plt.show()


def train_model(
        discriminator,
        generator,
        train_dataloader,
        valid_dataloader,
        loss_classification,
        loss_pixel_wise,
        l1_lambda,
        optimizer_discriminator,
        optimizer_generator,
        num_epochs
):
    """Train and Validate Model"""
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                dataloader = train_dataloader
                discriminator.train()  # Set model to training mode
                generator.train()
            else:
                dataloader = valid_dataloader
                discriminator.eval()  # Set model to evaluate mode
                generator.eval()
            # Iterate over data.
            with tqdm.tqdm(dataloader, unit='batch') as tqdm_loader:
                for geometry, stress in tqdm_loader:
                    tqdm_loader.set_description(f'Epoch {epoch} - {phase}')
                    # Image/Mask to device
                    geometry = geometry.to(device, dtype=torch.float32)
                    stress = stress.to(device, dtype=torch.float32)
                    optimizer_discriminator.zero_grad()
                    optimizer_generator.zero_grad()
                    # forward and backward
                    with torch.set_grad_enabled(phase == 'Train'):
                        # GENERATOR PART
                        fake_stress = generator(geometry)
                        prob_of_fake = discriminator(fake_stress, geometry)
                        real_label = torch.ones(prob_of_fake.size()).to(device)
                        loss_value_generator_fake = loss_classification(prob_of_fake, real_label)
                        loss_value_generator_pixel_wise = loss_pixel_wise(fake_stress, stress)
                        # Final Generator loss
                        loss_value_generator = loss_value_generator_fake + l1_lambda * loss_value_generator_pixel_wise
                        # backward + optimize generator only if in training phase
                        if phase == 'Train':
                            loss_value_generator.backward()
                            optimizer_generator.step()
                        # DISCRIMINATOR PART
                        prob_of_real = discriminator(stress, geometry)
                        real_labels = torch.ones(prob_of_real.size()).to(device)
                        loss_value_discriminator_real = loss_classification(prob_of_real, real_labels)

                        fake_stress = generator(geometry)
                        prob_of_fake = discriminator(fake_stress.detach(), geometry)
                        fake_labels = torch.zeros(prob_of_fake.size()).to(device)
                        loss_value_discriminator_fake = loss_classification(prob_of_fake, fake_labels)

                        # Final Discriminator loss
                        loss_value_discriminator = torch.mean(
                            loss_value_discriminator_real + loss_value_discriminator_fake
                        )
                        # Backward + optimize discriminator only if in training phase
                        if phase == 'Train':
                            loss_value_discriminator.backward()
                            optimizer_discriminator.step()
                        # Current statistics
                        tqdm_loader.set_postfix(
                            Discriminator_Loss=loss_value_discriminator.item(),
                            Generator_Loss=loss_value_generator.item()
                        )
                        time.sleep(.1)

    # Save model on last epoch
    torch.save(generator, rf'checkpoint\last.pth')
    return generator
