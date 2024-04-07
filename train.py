import pandas as pd
import torch.utils.data
import torchvision.utils as utils
import metrics.ssim as ssim

from math import log10
from tqdm import tqdm
from data_loader import SrganDataLoader, display_transform

from model import SrganNetwork


def train(dataset_dir, batch_size, num_epochs, training_results_dir, epochs_dir, statistics_dir):

    # Create data loader for SRGAN training data
    srgan_data_loader = SrganDataLoader(dataset_dir, batch_size=batch_size)

    # Initialize the SRGAN model
    srgan = SrganNetwork()

    # Initialize dictionary to store training results
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    # Iterate over each epoch
    for epoch in range(1, num_epochs + 1):
        # Initialize dictionary to store running results for current epoch
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        # Train the SRGAN model for the current epoch
        train_networks(srgan, srgan_data_loader, epoch, num_epochs, running_results)

        # Evaluate the SRGAN model for the current epoch
        validating_results, result_images = evaluate(srgan, srgan_data_loader)

        # Save epoch snapshots
        save_model_parameters(srgan, epoch, epochs_dir)

        # Calculate and save evaluation results to csv
        save_evaluation_results(results, running_results, validating_results, epoch, statistics_dir)

        # Save result images
        save_result_images(result_images, epoch, training_results_dir)


def train_networks(srgan, srgan_data_loader, epoch, num_epochs, running_results):
    """
    Function to train the SRGAN (Super-Resolution Generative Adversarial Network).

    Args:
    - srgan: SRGAN model instance containing generator, discriminator, optimizers, and loss functions.
    - srgan_data_loader: Data loader for SRGAN training data.
    - epoch: Current epoch number.
    - num_epochs: Total number of epochs for training.
    - running_results: Dictionary to store running results such as batch sizes.
    """

    # Initialize progress bar for training data loader
    train_bar = tqdm(srgan_data_loader.train_loader)

    # Set the generator and discriminator networks to train mode
    srgan.net_g.train()
    srgan.net_d.train()

    # Iterate through each batch in the training data loader
    for lr_image, hr_image in train_bar:
        batch_size = lr_image.size(0)
        running_results['batch_sizes'] += batch_size

        # Move images to GPU if available
        if torch.cuda.is_available():
            hr_image = hr_image.cuda()
            lr_image = lr_image.cuda()

        # Generate super-resolved (SR) image from low-resolution (LR) input
        sr_image = srgan.net_g(lr_image)

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################

        # Zero the gradients of the discriminator network
        srgan.net_d.zero_grad()

        # Calculate discriminator loss
        real_out = srgan.net_d(hr_image).mean()
        fake_out = srgan.net_d(sr_image).mean()
        d_loss = 1 - real_out + fake_out

        # Backpropagate and update discriminator parameters
        d_loss.backward(retain_graph=True)
        srgan.optimizer_d.step()

        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss

        # Zero the gradients of the generator network
        srgan.net_g.zero_grad()

        # Generate super-resolved image again (due to potential modification during discriminator update)
        sr_image = srgan.net_g(lr_image)

        # Calculate fake output again (due to potential modification during discriminator update)
        fake_out = srgan.net_d(sr_image).mean()

        # Calculate generator loss
        g_loss = srgan.loss_function(fake_out, sr_image, hr_image)

        # Backpropagate and update generator parameters
        g_loss.backward()
        srgan.optimizer_g.step()

        calculate_training_metrics(running_results, g_loss, d_loss, real_out, fake_out, batch_size, train_bar, epoch,
                                   num_epochs)


def evaluate(srgan, srgan_data_loader):
    """
    Evaluates the SRGAN model's performance on the validation dataset.

    Args:
    - srgan: SRGAN model instance containing generator network.
    - srgan_data_loader: Data loader for validation dataset.

    Returns:
    - validating_results: Dictionary containing validation results.
    - val_images: List of validation images for visualization.
    """
    # Set generator network to evaluation mode
    srgan.net_g.eval()

    # Disable gradient computation for validation
    with torch.no_grad():
        # Initialize progress bar for validation data loader
        val_bar = tqdm(srgan_data_loader.valid_loader)

        # Initialize dictionary to store validation results
        validating_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

        # List to store validation images for visualization
        val_images = []

        # Counter to limit the number of saved result images
        result_images_counter = 0

        # Iterate through each batch in the validation data loader
        for val_lr, val_hr in val_bar:
            batch_size = val_lr.size(0)
            validating_results['batch_sizes'] += batch_size

            # Move images to GPU if available
            if torch.cuda.is_available():
                val_lr = val_lr.cuda()
                val_hr = val_hr.cuda()

            # Generate super-resolved (SR) image from low-resolution (LR) input
            sr = srgan.net_g(val_lr)

            # Calculate Mean Squared Error (MSE) for the batch
            batch_mse = ((sr - val_hr) ** 2).data.mean()
            validating_results['mse'] += batch_mse * batch_size

            # Calculate Structural Similarity Index (SSIM) for the batch
            batch_ssim = ssim.ssim(sr.data.cpu(), val_hr.data.cpu()).item()
            validating_results['ssims'] += batch_ssim * batch_size

            # Calculate Peak Signal-to-Noise Ratio (PSNR) for the batch
            validating_results['psnr'] = 10 * log10((val_hr.max() ** 2) / (validating_results['mse'] / validating_results['batch_sizes']))

            # Calculate overall SSIM for the batch
            validating_results['ssim'] = validating_results['ssims'] / validating_results['batch_sizes']

            # Update progress bar with current PSNR and SSIM
            val_bar.set_description(desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (validating_results['psnr'], validating_results['ssim']))

            # Append LR, SR, and HR images to the list for visualization
            if result_images_counter < 15:
                val_images.extend(
                    [display_transform()(val_lr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0)),
                     display_transform()(val_hr.data.cpu().squeeze(0))])
                result_images_counter += 1

        # Return validation results and images
        return validating_results, val_images


def calculate_training_metrics(running_results, g_loss, d_loss, real_out, fake_out, batch_size, train_bar, epoch,
                               num_epochs):
    running_results['g_loss'] += g_loss.item() * batch_size
    running_results['d_loss'] += d_loss.item() * batch_size
    running_results['d_score'] += real_out.item() * batch_size
    running_results['g_score'] += fake_out.item() * batch_size

    train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
        epoch, num_epochs, running_results['d_loss'] / running_results['batch_sizes'],
        running_results['g_loss'] / running_results['batch_sizes'],
        running_results['d_score'] / running_results['batch_sizes'],
        running_results['g_score'] / running_results['batch_sizes']))


def save_model_parameters(srgan, epoch, epochs_dir):
    torch.save(srgan.net_g.state_dict(), f'{epochs_dir}/netG_epoch_%d.pth' % epoch)
    torch.save(srgan.net_d.state_dict(), f'{epochs_dir}/netD_epoch_%d.pth' % epoch)


def save_evaluation_results(results, running_results, validating_results, epoch, statistics_dir):
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(validating_results['psnr'])
    results['ssim'].append(validating_results['ssim'])

    out_path = statistics_dir + "/"
    data_frame = pd.DataFrame(data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                              'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']}, index=range(1, epoch + 1))
    data_frame.to_csv(out_path + 'srf_' + 'train_results.csv', index_label='Epoch')


def save_result_images(images, epoch, out_path):
    """
    Save processed images to file.

    Args:
    - images: List of images to save.
    - epoch: Current epoch number.
    - out_path: Path to save images.

    """
    images = torch.stack(images)
    images = torch.chunk(images, images.size(0) // 15)

    for image in images:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + '/epoch_%d.png' % epoch, padding=5)
        break  # Remove this line if you want to save all images


if __name__ == '__main__':
    train(dataset_dir='data/debug',
          batch_size=16,
          num_epochs=2,
          training_results_dir='training_results',
          epochs_dir='epochs',
          statistics_dir='statistics')
