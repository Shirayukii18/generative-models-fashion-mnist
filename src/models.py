"""Model architectures untuk benchmark Fashion-MNIST."""
import math
import torch
from torch import nn
import torch.nn.functional as F


# - Encoder dan decoder MLP
def encoder():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())


def decoder():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, images):
        latent = self.encoder(images)
        return self.decoder(latent), latent


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mean = nn.Linear(256, 64)
        self.logvar = nn.Linear(256, 64)
        self.decoder = decoder()

    def encode(self, images):
        hidden = self.features(images)
        return self.mean(hidden), self.logvar(hidden).clamp(-20, 20)

    def forward(self, images):
        mean, logvar = self.encode(images)
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.decoder(latent), mean, logvar


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.head = nn.Linear(64, 10)

    def forward(self, images):
        return self.head(self.encoder(images))


class Generator(nn.Module):
    def __init__(self, convolutional=False):
        super().__init__()
        self.latent_dim = 100 if convolutional else 64
        if convolutional:
            self.net = nn.Sequential(
                nn.Unflatten(1, (100, 1, 1)),
                nn.ConvTranspose2d(100, 128, 7, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(64, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 784), nn.Tanh(), nn.Unflatten(1, (1, 28, 28)),
            )

    def forward(self, latent):
        return self.net(latent)


class Discriminator(nn.Module):
    def __init__(self, convolutional=False):
        super().__init__()
        if convolutional:
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1),
            )

    def forward(self, images):
        return self.net(images).flatten()


# - Time-conditioned U-Net
class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.time = nn.Linear(64, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, images, time_embedding):
        hidden = F.silu(self.norm1(self.conv1(images)))
        hidden = hidden + self.time(time_embedding)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(hidden)) + self.skip(images))


class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.down1 = TimeBlock(1, 16)
        self.down2 = TimeBlock(16, 32)
        self.middle = TimeBlock(32, 64)
        self.up2 = TimeBlock(96, 32)
        self.up1 = TimeBlock(48, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, images, timesteps):
        frequency = torch.exp(-math.log(10000) * torch.arange(32, device=images.device) / 31)
        angles = timesteps.float()[:, None] * frequency[None]
        time_embedding = self.time(torch.cat([angles.sin(), angles.cos()], dim=1))
        first = self.down1(images, time_embedding)
        second = self.down2(F.avg_pool2d(first, 2), time_embedding)
        middle = self.middle(F.avg_pool2d(second, 2), time_embedding)
        hidden = self.up2(torch.cat([F.interpolate(middle, size=14, mode='nearest'), second], 1), time_embedding)
        hidden = self.up1(torch.cat([F.interpolate(hidden, size=28, mode='nearest'), first], 1), time_embedding)
        return self.output(hidden)


class Diffusion:
    def __init__(self, device, steps=200):
        self.steps = steps
        time = torch.linspace(0, steps, steps + 1, device=device) / steps
        cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        self.beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(0.0001, 0.999)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(0)
        previous = F.pad(self.alpha_bar[:-1], (1, 0), value=1)
        self.variance = self.beta * (1 - previous) / (1 - self.alpha_bar)
        self.coef1 = self.beta * previous.sqrt() / (1 - self.alpha_bar)
        self.coef2 = (1 - previous) * self.alpha.sqrt() / (1 - self.alpha_bar)

    def add_noise(self, images, timesteps, noise):
        weight = self.alpha_bar[timesteps, None, None, None]
        return weight.sqrt() * images + (1 - weight).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, count, trajectory=False):
        images = torch.randn(count, 1, 28, 28, device=self.beta.device)
        frames = [images[:1].cpu()]
        for step in reversed(range(self.steps)):
            timesteps = torch.full((count,), step, device=images.device, dtype=torch.long)
            noise = model(images, timesteps)
            original = ((images - (1 - self.alpha_bar[step]).sqrt() * noise) / self.alpha_bar[step].sqrt()).clamp(-1, 1)
            images = self.coef1[step] * original + self.coef2[step] * images
            if step > 0:
                images += self.variance[step].sqrt() * torch.randn_like(images)
            if trajectory and step in [159, 119, 79, 39, 0]:
                frames.append(images[:1].cpu())
        result = (images.clamp(-1, 1) + 1) / 2
        return (result, frames) if trajectory else result
