import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        moudle = []
        dims = [64, 128, 256, 512]
        moudle += [nn.Conv2d(in_channels=self.in_size[0], out_channels=dims[0], kernel_size=4, padding=1, stride=2)]
        for in_channels, out_channels in zip(dims[:-1], dims[1:]):
            moudle += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, padding=1, stride=2),
                        nn.BatchNorm2d(out_channels), nn.ReLU()]
        moudle += [nn.Conv2d(in_channels=dims[-1], out_channels=1, kernel_size=4, padding=0, stride=2)]
        self.cnn = nn.Sequential(*moudle)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.cnn(x).reshape(shape=(x.size(0), -1))
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        module = []
        dims = [512, 256, 128, 64]
        module += [
            nn.ConvTranspose2d(in_channels=self.z_dim, out_channels=dims[0], kernel_size=featuremap_size,
                               padding=0, stride=2)]
        for i, (cin, cout) in enumerate(zip(dims[:-1], dims[1:])):
            module += [nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=featuremap_size, padding=1, stride=2),
                        nn.BatchNorm2d(cout), nn.ReLU()]
        module += [
            nn.ConvTranspose2d(in_channels=dims[-1], out_channels=out_channels, kernel_size=featuremap_size,
                               padding=1, stride=2), nn.Tanh()]
        self.cnn = nn.Sequential(*module)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        if with_grad:
            samples = self.cnn(torch.randn((n, self.z_dim, 1, 1)).to(device))
        else:
            with torch.no_grad():
                samples = self.cnn(torch.randn((n, self.z_dim, 1, 1)).to(device))
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        return self.cnn(z.reshape((z.size(0), self.z_dim, 1, 1)))


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    rand_label = label_noise * torch.rand(size=y_data.size()).to(y_data.device)
    rand_label2 = label_noise * torch.rand(size=y_generated.size()).to(y_data.device)
    pred_data = rand_label + data_label - label_noise / 2
    gdn =  rand_label2 + (1 - data_label) - label_noise / 2
    loss_data = nn.BCEWithLogitsLoss()(y_data, pred_data).to(y_data.device)
    loss_generated = nn.BCEWithLogitsLoss()(y_generated, gdn).to(y_data.device)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    if not data_label:
        y_labels = torch.zeros_like(y_generated).to(y_generated.device)
    else:
        y_labels = torch.ones_like(y_generated).to(y_generated.device)
    loss = nn.BCEWithLogitsLoss()(y_generated, y_labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    gen = dsc_model.forward(gen_model.sample(x_data.size(0), False).to(x_data.device)).to(x_data.device)
    y = dsc_model.forward(x_data).to(x_data.device)
    #loss
    dsc_loss = dsc_loss_fn(y, gen)
    #step
    dsc_optimizer.zero_grad()
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_loss = gen_loss_fn(dsc_model.forward(gen_model.sample(x_data.size(0), True).to(x_data.device)).to(x_data.device)).to(x_data.device)
    #step
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======

    if len(dsc_losses) < 2:
        return False

    if gen_losses[-1] < gen_losses[-2] and checkpoint_file and dsc_losses[-1] < dsc_losses[-2] and checkpoint_file:
        torch.save(gen_model, checkpoint_file)
        saved = True
    return saved
