import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models.vgg import VGG19_Weights


class GeneratorLoss(nn.Module):
    """
    Generator Loss Module.

    This class defines the loss function used to train the generator in a
    generative adversarial network (GAN). The loss function consists of
    multiple components including adversarial loss, perception loss, image
    loss, and total variation (TV) loss. These components are combined
    using weighted sums to form the final loss.

    Attributes:
        loss_network (nn.Sequential): Pre-trained VGG-19 network used for
            calculating perception loss.
        mse_loss (nn.MSELoss): Mean squared error loss function.
        tv_loss (TVLoss): Total variation loss function.
    """

    def __init__(self):
        """
        Initializes the GeneratorLoss module.

        This method initializes the GeneratorLoss module by setting up the
        pre-trained VGG-19 network for perception loss calculation, and
        defining the mean squared error loss function and the total variation
        loss function. It freezes the parameters of the VGG-19 network.
        """
        super(GeneratorLoss, self).__init__()

        # Initialize pre-trained VGG-19 network
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        loss_network = nn.Sequential(*list(vgg.features)).eval()

        # Freeze parameters of the VGG-19 network
        for param in loss_network.parameters():
            param.requires_grad = False

        # Set attributes
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        """
        Computes the generator loss.

        This method computes the generator loss, which is a combination of
        adversarial loss, perception loss, image loss, and total variation
        loss. It returns the weighted sum of these loss components.

        Args:
            out_labels: Output labels from the discriminator.
            out_images: Generated images produced by the
                generator.
            target_images: Target images that the generator
                aims to produce.
        """
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        # Perception Loss
        perception_loss = self.mse_loss(
            self.loss_network(out_images),
            self.loss_network(target_images)
        )

        # Content Loss (pixel-wise)
        image_loss = self.mse_loss(out_images, target_images)

        # Total variation Loss
        tv_loss = self.tv_loss(out_images)

        # Combine loss components with weights and return
        return (
                image_loss
                + 0.001 * adversarial_loss
                + 0.006 * perception_loss
                + 2e-8 * tv_loss
        )


class TVLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        """
        Computes the Total Variation (TV) loss.

        This method calculates the TV loss, which measures the total
        variation or smoothness of the input images. It penalizes abrupt
        changes in pixel intensities along both horizontal and vertical
        directions.
        """
        # Get batch size
        batch_size = x.size()[0]

        # Get height and width of the input images
        h_x = x.size()[2]
        w_x = x.size()[3]

        # Compute the number of elements along the height and width dimensions
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        # Compute the TV loss along the height and width dimensions
        # TV loss measures the squared difference between adjacent pixel values
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        # Scale the TV loss by the TV loss weight, and normalize by the number of elements
        # in the input batch, and return
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        """
        Computes the size of a tensor.

        This static method computes the size of a tensor in terms of the
        number of elements.

        Args:
            t: Input tensor.

        Returns:
            int: Size of the tensor in terms of the number of elements.
        """
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
