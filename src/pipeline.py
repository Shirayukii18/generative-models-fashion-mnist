"""Training pipeline dengan split, checkpoint, dan log yang dapat diaudit."""
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
from src.models import Autoencoder, VAE, Classifier, Generator, Discriminator, NoisePredictor, Diffusion, decoder


# - Experiment configuration
CONFIG = {
    'seed': 42,
    'batch_size': 256,  # - Jumlah gambar dalam satu batch
    'latent_dim': 64,
    'sparsity_weight': 0.001,  # - Penalti rata-rata aktivasi laten
    'diffusion_steps': 200,
    'epochs': {'AE': 25, 'SparseAE': 25, 'VAE': 25, 'GAN': 40, 'DCGAN': 40, 'DDPM': 30, 'Classifier': 35, 'FrozenDecoder': 25},
    'fid_samples': 3000,  # - Ukuran sama untuk real dan generated samples
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path.cwd()
OUTPUT = ROOT / 'outputs'
OUTPUT.mkdir(exist_ok=True)
torch.set_num_threads(4)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def prepare_data():
    config_path = OUTPUT / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG:
        raise ValueError('Configuration berubah. Arsipkan outputs sebelum memulai eksperimen baru.')
    # - Gabungkan official partitions sebelum split internal 80/10/10
    original_train = FashionMNIST(ROOT / 'data', train=True, download=True)
    original_test = FashionMNIST(ROOT / 'data', train=False, download=True)
    images = torch.cat([original_train.data, original_test.data]).unsqueeze(1).float() / 255
    labels = torch.cat([original_train.targets, original_test.targets])
    indices = np.arange(len(labels))
    train_index, remaining = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels.numpy())
    val_index, test_index = train_test_split(remaining, test_size=0.5, random_state=42, stratify=labels[remaining].numpy())
    np.savez(OUTPUT / 'split_indices.npz', train=train_index, validation=val_index, test=test_index)
    data = {}
    for split, index in [('train', train_index), ('validation', val_index), ('test', test_index)]:
        # - Cache tensor pada GPU agar transfer tidak mendominasi training
        data[split] = TensorDataset(images[index].to(DEVICE), labels[index].to(DEVICE))
    (OUTPUT / 'config.json').write_text(json.dumps(CONFIG, indent=2))
    (OUTPUT / 'hardware.json').write_text(json.dumps({'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None, 'torch': torch.__version__, 'precision': 'FP32', 'split': {key: len(value) for key, value in data.items()}}, indent=2))
    return data


def loader(dataset, training=False):
    return DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=training, num_workers=0)


def objective(name, model, images, labels, classifier=None, diffusion=None):
    if name == 'Classifier':
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        return loss, {'accuracy': (logits.argmax(1) == labels).float().mean()}
    if name == 'FrozenDecoder':
        with torch.no_grad():
            latent = classifier.encoder(images)
        loss = F.mse_loss(model(latent), images)
        return loss, {'reconstruction': loss}
    if name == 'DDPM':
        images = images * 2 - 1
        timestep = torch.randint(diffusion.steps, (len(images),), device=DEVICE)
        noise = torch.randn_like(images)
        prediction = model(diffusion.add_noise(images, timestep, noise), timestep)
        loss = F.mse_loss(prediction, noise)
        return loss, {'noise_mse': loss}
    if name == 'VAE':
        reconstructed, mean, logvar = model(images)
        reconstruction = F.binary_cross_entropy(reconstructed, images, reduction='none').flatten(1).sum(1).mean()
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(1).mean()
        return reconstruction + kl, {'reconstruction_bce': reconstruction, 'kl': kl}
    reconstructed, latent = model(images)
    reconstruction = F.mse_loss(reconstructed, images)
    penalty = latent.abs().mean()
    loss = reconstruction + (CONFIG['sparsity_weight'] * penalty if name == 'SparseAE' else 0)
    return loss, {'reconstruction': reconstruction, 'latent_l1': penalty}


def train_model(name, data, classifier=None):
    constructors = {'AE': Autoencoder, 'SparseAE': Autoencoder, 'VAE': VAE, 'Classifier': Classifier, 'FrozenDecoder': decoder, 'DDPM': NoisePredictor}
    seed_everything()
    model = constructors[name]().to(DEVICE)
    diffusion = Diffusion(DEVICE, CONFIG['diffusion_steps'])
    checkpoint = OUTPUT / f'{name}.pt'
    if checkpoint.exists():
        if not (OUTPUT / f'{name}_selection.json').exists():
            raise RuntimeError(f'{name}: training sebelumnya belum lengkap. Arsipkan checkpoint parsial sebelum mengulang.')
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        print(f'{name}: memakai checkpoint tersimpan', flush=True)
        return model.eval()
    learning_rate = 2e-4 if name == 'DDPM' else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_loss, best_epoch = [], float('inf'), 0
    for epoch in range(1, CONFIG['epochs'][name] + 1):
        synchronize()
        started = time.perf_counter()
        model.train()
        sums = {}
        for images, labels in loader(data['train'], training=True):
            optimizer.zero_grad(set_to_none=True)
            loss, components = objective(name, model, images, labels, classifier, diffusion)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {name}, epoch {epoch}')
            loss.backward()
            optimizer.step()
            for key, value in {'loss': loss, **components}.items():
                sums[key] = sums.get(key, 0) + value.detach() * len(images)
        synchronize()
        seconds = time.perf_counter() - started  # - Training saja, tanpa validation
        model.eval()
        validation = {}
        # - Validation memakai noise tetap dan tidak mengubah RNG training
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE.type == 'cuda' else []):
            torch.manual_seed(31415)
            with torch.no_grad():
                for images, labels in loader(data['validation']):
                    loss, components = objective(name, model, images, labels, classifier, diffusion)
                    for key, value in {'loss': loss, **components}.items():
                        validation[key] = validation.get(key, 0) + value.detach() * len(images)
        record = {'epoch': epoch, 'seconds': seconds, **{key: float(value / len(data['train'])) for key, value in sums.items()}, **{'val_' + key: float(value / len(data['validation'])) for key, value in validation.items()}}
        history.append(record)
        if record['val_loss'] < best_loss:
            best_loss, best_epoch = record['val_loss'], epoch
            torch.save(model.state_dict(), checkpoint)
        pd.DataFrame(history).to_csv(OUTPUT / f'{name}_history.csv', index=False)
        print(f"{name} {epoch}/{CONFIG['epochs'][name]} loss={record['loss']:.5f} val={record['val_loss']:.5f} sec={seconds:.1f}", flush=True)
    (OUTPUT / f'{name}_selection.json').write_text(json.dumps({'rule': 'minimum validation objective', 'best_epoch': best_epoch, 'val_loss': best_loss}))
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    return model.eval()


def train_gan(name, data):
    seed_everything()
    convolutional = name == 'DCGAN'
    generator = Generator(convolutional).to(DEVICE)
    discriminator = Discriminator(convolutional).to(DEVICE)
    checkpoint = OUTPUT / f'{name}.pt'
    if checkpoint.exists():
        if not (OUTPUT / f'{name}_selection.json').exists():
            raise RuntimeError(f'{name}: training sebelumnya belum lengkap. Arsipkan checkpoint parsial sebelum mengulang.')
        generator.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        return generator.eval()
    if convolutional:
        for layer in list(generator.modules()) + list(discriminator.modules()):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight, 0, 0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight, 1, 0.02)
                nn.init.zeros_(layer.bias)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    history = []
    for epoch in range(1, CONFIG['epochs'][name] + 1):
        synchronize()
        started = time.perf_counter()
        generator.train()
        discriminator.train()
        sums = torch.zeros(2, device=DEVICE)
        for images, _ in loader(data['train'], training=True):
            real = images * 2 - 1
            count = len(real)
            discriminator_optimizer.zero_grad(set_to_none=True)
            fake = generator(torch.randn(count, generator.latent_dim, device=DEVICE))
            real_logits, fake_logits = discriminator(real), discriminator(fake.detach())
            discriminator_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            discriminator_loss.backward()
            discriminator_optimizer.step()
            generator_optimizer.zero_grad(set_to_none=True)
            discriminator.requires_grad_(False)
            fake_logits = discriminator(fake)
            generator_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            if not torch.isfinite(generator_loss + discriminator_loss):
                raise RuntimeError(f'Non-finite GAN loss: {name}')
            generator_loss.backward()
            generator_optimizer.step()
            discriminator.requires_grad_(True)
            discriminator.train()
            sums += torch.stack([generator_loss.detach(), discriminator_loss.detach()]) * count
        synchronize()
        record = {'epoch': epoch, 'seconds': time.perf_counter() - started, 'generator_loss': float(sums[0] / len(data['train'])), 'discriminator_loss': float(sums[1] / len(data['train']))}
        history.append(record)
        pd.DataFrame(history).to_csv(OUTPUT / f'{name}_history.csv', index=False)
        print(f"{name} {epoch}/{CONFIG['epochs'][name]} G={record['generator_loss']:.4f} D={record['discriminator_loss']:.4f} sec={record['seconds']:.1f}", flush=True)
    torch.save(generator.state_dict(), checkpoint)
    torch.save(discriminator.state_dict(), OUTPUT / f'{name}_discriminator.pt')
    (OUTPUT / f'{name}_selection.json').write_text(json.dumps({'rule': 'fixed final epoch; no test-based selection', 'best_epoch': CONFIG['epochs'][name]}))
    return generator.eval()


def train_all(data):
    models = {}
    for name in ['AE', 'SparseAE', 'VAE', 'Classifier']:
        models[name] = train_model(name, data)
    models['Classifier'].requires_grad_(False)
    models['FrozenDecoder'] = train_model('FrozenDecoder', data, models['Classifier'])
    for name in ['GAN', 'DCGAN']:
        models[name] = train_gan(name, data)
    models['DDPM'] = train_model('DDPM', data)
    return models
