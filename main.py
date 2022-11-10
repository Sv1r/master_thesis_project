import torch
import random
import numpy as np

import utils
import models
import dataset
import settings


def main():
    # Fix random
    random.seed(settings.RANDOM_STATE)
    np.random.seed(settings.RANDOM_STATE)
    torch.manual_seed(settings.RANDOM_STATE)
    torch.cuda.manual_seed(settings.RANDOM_STATE)
    # Models
    discriminator = models.Discriminator()
    discriminator = discriminator.apply(utils.weights_init)
    generator = models.generator
    # Freeze Encoder
    for param in generator.encoder.parameters():
        param.requires_grad = False
    # losses
    loss_classification = torch.nn.BCELoss()
    loss_pixel_wise = torch.nn.L1Loss()
    # optimizers
    optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=settings.LEARNING_RATE)
    optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=settings.LEARNING_RATE)
    # Train Model
    model, loss_and_metrics = utils.train_model(
        discriminator=discriminator,
        generator=generator,
        train_dataloader=dataset.train_dataloader,
        valid_dataloader=dataset.valid_dataloader,
        loss_classification=loss_classification,
        loss_pixel_wise=loss_pixel_wise,
        l1_lambda=settings.LAMBDA_L1,
        optimizer_discriminator=optimizer_discriminator,
        optimizer_generator=optimizer_generator,
        num_epochs=settings.EPOCHS,
        avg_results=True
    )
    # Result Plot of Loss and Metrics
    utils.result_plot(loss_and_metrics)


if __name__ == '__main__':
    main()
