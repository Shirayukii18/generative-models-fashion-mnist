"""Test evaluation dan lima figur komparatif."""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
from src.pipeline import CONFIG, DEVICE, ROOT, OUTPUT, seed_everything, loader, synchronize
from src.models import Diffusion


@torch.no_grad()
def reconstruct(name, model, images, classifier):
    if name in ['AE', 'SparseAE']:
        return model(images)[0]
    if name == 'VAE':
        return model.decoder(model.encode(images)[0])  # - Posterior mean untuk evaluation deterministik
    return model(classifier.encoder(images))


@torch.no_grad()
def generate(name, model, count, diffusion):
    if name == 'DDPM':
        return diffusion.sample(model, count)
    if name == 'VAE':
        return model.decoder(torch.randn(count, 64, device=DEVICE))
    return (model(torch.randn(count, model.latent_dim, device=DEVICE)) + 1) / 2


@torch.no_grad()
def reconstruction_metrics(name, model, dataset, classifier):
    values = []
    for images, _ in loader(dataset):
        prediction = reconstruct(name, model, images, classifier)
        mse = (prediction - images).square().flatten(1).mean(1)
        psnr = -10 * torch.log10(mse.clamp_min(1e-12))
        ssim = structural_similarity_index_measure(prediction, images, data_range=1.0, reduction='none')
        values.append(torch.stack([mse, psnr, ssim], 1).cpu())
    values = torch.cat(values).numpy()
    pd.DataFrame(values, columns=['mse', 'psnr', 'ssim']).to_csv(OUTPUT / f'{name}_test_per_image.csv', index=False)
    return dict(zip(['test_mse', 'test_psnr', 'test_ssim'], values.mean(0).tolist()))


@torch.no_grad()
def measure_latency(function, repeats=10):
    for _ in range(3):
        function()
    times = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        function()
        synchronize()
        times.append((time.perf_counter() - started) * 1000)
    return {'latency_ms': float(np.median(times)), 'latency_p95_ms': float(np.percentile(times, 95)), 'latency_repeats': repeats}


@torch.no_grad()
def evaluate(models, data):
    seed_everything(2026)
    diffusion = Diffusion(DEVICE, CONFIG['diffusion_steps'])
    classifier = models['Classifier']
    test_images, test_labels = data['test'].tensors
    accuracy = sum(int((classifier(images).argmax(1) == labels).sum()) for images, labels in loader(data['test'])) / len(test_images)
    summary = {'classifier_test_accuracy': accuracy, 'classifier_meets_88_percent': accuracy >= 0.88}
    rows = []
    for name in ['AE', 'SparseAE', 'VAE', 'GAN', 'DCGAN', 'DDPM', 'FrozenDecoder']:
        model = models[name].eval()
        history = pd.read_csv(OUTPUT / f'{name}_history.csv')
        row = {'model': name, 'latent_dim': 784 if name == 'DDPM' else (100 if name == 'DCGAN' else 64), 'parameters': sum(parameter.numel() for parameter in model.parameters()), 'training_seconds_per_epoch': history.seconds.mean(), 'epochs': len(history)}
        if name in ['AE', 'SparseAE', 'VAE', 'FrozenDecoder']:
            row.update(reconstruction_metrics(name, model, data['test'], classifier))
        if name in ['AE', 'SparseAE', 'FrozenDecoder']:
            row.update(measure_latency(lambda: reconstruct(name, model, test_images[:64], classifier)))
        else:
            row.update(measure_latency(lambda: generate(name, model, 64, diffusion), repeats=5 if name == 'DDPM' else 20))
        rows.append(row)
        print('Test metrics:', row, flush=True)
    # - FID memakai Inception-v3 2048-dim dari torch-fidelity
    torch.hub.set_dir(str(ROOT / 'data' / 'torch_cache'))
    # - Features FP32; akumulasi covariance TorchMetrics tetap FP64
    fid = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False, antialias=False).to(DEVICE)
    indices = np.random.default_rng(2026).choice(len(test_images), CONFIG['fid_samples'], replace=False)
    np.save(OUTPUT / 'fid_test_indices.npy', indices)
    def to_rgb(images):
        return (images.clamp(0, 1) * 255).round().to(torch.uint8).repeat(1, 3, 1, 1)
    for start in range(0, len(indices), 64):
        fid.update(to_rgb(test_images[indices[start:start + 64]]), real=True)
    print('Real FID features ready.', flush=True)
    for name in ['VAE', 'GAN', 'DCGAN', 'DDPM']:
        seed_everything(2026)
        fid.reset()
        generated = []
        counts = torch.zeros(10, device=DEVICE)
        confidences = []
        for start in range(0, CONFIG['fid_samples'], 64):
            images = generate(name, models[name], min(64, CONFIG['fid_samples'] - start), diffusion)
            generated.append(images.cpu())
            fid.update(to_rgb(images), real=False)
            probabilities = classifier(images).softmax(1)
            counts += torch.bincount(probabilities.argmax(1), minlength=10)
            confidences.append(probabilities.max(1).values)
        torch.save(torch.cat(generated), OUTPUT / f'{name}_samples.pt')
        score = float(fid.compute())
        distribution = counts / counts.sum()
        entropy = float(-(distribution * distribution.clamp_min(1e-12).log()).sum())
        next(row for row in rows if row['model'] == name).update({'fid': score, 'predicted_class_entropy': entropy, 'mean_classifier_confidence': float(torch.cat(confidences).mean())})
        summary[name + '_class_distribution'] = distribution.cpu().tolist()
        print(f'{name} FID ({CONFIG["fid_samples"]} samples) = {score:.4f}', flush=True)
        pd.DataFrame(rows).to_csv(OUTPUT / 'metrics.csv', index=False)
    # - Eksperimen proyeksi transpos dan struktur matriks
    first, second, head = classifier.encoder[1], classifier.encoder[3], classifier.head
    latent = classifier.encoder(test_images)
    transpose = torch.sigmoid(F.relu(latent @ second.weight) @ first.weight).reshape(-1, 1, 28, 28)
    summary['latent_transpose_test_mse'] = float((transpose - test_images).square().mean())
    summary['first_weight_rank'] = int(torch.linalg.matrix_rank(first.weight))
    summary['first_weight_nullity'] = 784 - summary['first_weight_rank']
    gram = first.weight.T @ first.weight
    summary['gram_identity_relative_error'] = float(torch.linalg.norm(gram - torch.eye(784, device=DEVICE)) / (784 ** 0.5))
    for name in ['AE', 'SparseAE']:
        hidden = models[name].encoder(test_images)
        summary[name + '_latent_mean_abs'] = float(hidden.abs().mean())
        summary[name + '_latent_zero_fraction'] = float((hidden < 1e-3).float().mean())
    summary['fid_protocol'] = {'feature': 2048, 'real_count': CONFIG['fid_samples'], 'fake_count': CONFIG['fid_samples'], 'real_source': 'fixed random subset of internal test', 'input': 'rounded uint8 RGB replication; torch-fidelity TensorFlow-compatible resize; antialias=False', 'feature_precision': 'FP32', 'covariance_precision': 'FP64', 'seed': 2026}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(OUTPUT / 'metrics.csv', index=False)
    return pd.DataFrame(rows), summary


def plot_rows(rows, labels, filename, column_titles=None):
    count = rows[0].shape[0]
    figure, axes = plt.subplots(len(rows), count, figsize=(count * 1.35 + 1, len(rows) * 1.45), squeeze=False)
    for row_index, (images, label) in enumerate(zip(rows, labels)):
        for column, image in enumerate(images):
            axes[row_index, column].imshow(image.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            if column == 0:
                axes[row_index, column].set_ylabel(label, fontsize=9)
            if row_index == 0 and column_titles:
                axes[row_index, column].set_title(column_titles[column], fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=220, bbox_inches='tight')
    plt.close(figure)


@torch.no_grad()
def create_figures(models, data):
    seed_everything(777)
    images, labels = data['test'].tensors
    classifier = models['Classifier']
    selected = torch.stack([torch.where(labels == category)[0][0] for category in range(8)])
    originals = images[selected]
    plot_rows([originals] + [reconstruct(name, models[name], originals, classifier) for name in ['AE', 'SparseAE', 'VAE']], ['Original', 'Vanilla AE', 'Sparse AE', 'VAE'], 'figure1_reconstruction.png')
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for axis, name in zip(axes, ['VAE', 'GAN', 'DCGAN', 'DDPM']):
        samples = torch.load(OUTPUT / f'{name}_samples.pt', weights_only=True)
        axis.imshow(make_grid(samples[:16], nrow=4, padding=2, pad_value=1).permute(1, 2, 0))
        axis.set_title(name)
        axis.axis('off')
    figure.tight_layout()
    figure.savefig(OUTPUT / 'figure2_generation.png', dpi=220, bbox_inches='tight')
    plt.close(figure)
    first, second, head = classifier.encoder[1], classifier.encoder[3], classifier.head
    latent = classifier.encoder(originals)
    one_hot = F.one_hot(labels[selected], 10).float()
    class_projection = torch.sigmoid(F.relu(F.relu(one_hot @ head.weight) @ second.weight) @ first.weight).reshape(-1, 1, 28, 28)
    latent_projection = torch.sigmoid(F.relu(latent @ second.weight) @ first.weight).reshape(-1, 1, 28, 28)
    test_latent = classifier.encoder(images)
    train_latent = classifier.encoder(data['train'].tensors[0])
    noise = torch.randn(8, 64, device=DEVICE)
    diagnostic_samples = models['FrozenDecoder'](test_latent.mean(0) + noise * test_latent.std(0))
    train_samples = models['FrozenDecoder'](train_latent.mean(0) + noise * train_latent.std(0))
    plot_rows([originals, class_projection, latent_projection, models['FrozenDecoder'](latent), models['AE'](originals)[0], diagnostic_samples], ['Original', 'Class transpose', 'Latent transpose', 'Frozen decoder', 'Vanilla AE', 'MLP sampling*'], 'figure3_classifier.png')
    plot_rows([diagnostic_samples, train_samples], ['Test stats*', 'Train stats'], 'mlp_sampling_control.png')
    # - Endpoint AE/VAE berasal dari kelas 0 dan 9 pada validation set
    val_images, val_labels = data['validation'].tensors
    endpoint_images = torch.stack([val_images[torch.where(val_labels == category)[0][0]] for category in [0, 9]])
    alpha = torch.linspace(0, 1, 6, device=DEVICE)[:, None]
    rows, endpoint_info = [], {}
    for name in ['AE', 'SparseAE', 'VAE', 'GAN', 'DCGAN']:
        model = models[name]
        if name in ['AE', 'SparseAE']:
            endpoints = model.encoder(endpoint_images)
        elif name == 'VAE':
            endpoints = model.encode(endpoint_images)[0]
        else:
            # - GAN tanpa encoder: pilih prior endpoint dengan prediction kelas 0 dan 9
            candidates = torch.randn(4096, model.latent_dim, device=DEVICE)
            generated = torch.cat([(model(batch) + 1) / 2 for batch in candidates.split(256)])
            probabilities = classifier(generated).softmax(1)
            chosen = []
            for category in [0, 9]:
                matches = torch.where(probabilities.argmax(1) == category)[0]
                if len(matches) == 0:
                    raise RuntimeError(f'{name}: tidak ada endpoint predicted class {category}')
                chosen.append(matches[probabilities[matches, category].argmax()])
            endpoints = candidates[torch.stack(chosen)]
            endpoint_info[name] = {'predicted_classes': [0, 9], 'selection': 'highest classifier confidence among 4096 prior candidates; not ground truth', 'confidence': [float(probabilities[index, category]) for index, category in zip(chosen, [0, 9])]}
        interpolated = (1 - alpha) * endpoints[0] + alpha * endpoints[1]
        rows.append((model(interpolated) + 1) / 2 if name in ['GAN', 'DCGAN'] else model.decoder(interpolated))
    (OUTPUT / 'interpolation_protocol.json').write_text(json.dumps(endpoint_info, indent=2))
    plot_rows(rows, ['Vanilla AE', 'Sparse AE', 'VAE', 'Vanilla GAN', 'DCGAN'], 'figure4_interpolation.png', [f'alpha={value:.1f}' for value in np.linspace(0, 1, 6)])
    figure, axes = plt.subplots(2, 3, figsize=(12, 6))
    for axis, name in zip(axes.flatten(), ['AE', 'SparseAE', 'VAE', 'GAN', 'DCGAN', 'DDPM']):
        history = pd.read_csv(OUTPUT / f'{name}_history.csv')
        keys = ['generator_loss', 'discriminator_loss'] if name in ['GAN', 'DCGAN'] else ['loss', 'val_loss']
        for key in keys:
            axis.plot(history.epoch, history[key], label=key)
        axis.set(title=name, xlabel='Epoch', ylabel='Objective')
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / 'figure5_loss.png', dpi=220, bbox_inches='tight')
    plt.close(figure)
    diffusion = Diffusion(DEVICE, CONFIG['diffusion_steps'])
    _, frames = diffusion.sample(models['DDPM'], 1, trajectory=True)
    plot_rows([torch.cat([(frame.clamp(-1, 1) + 1) / 2 for frame in frames])], ['DDPM'], 'figure5_trajectory.png', ['t=200', 't=159', 't=119', 't=79', 't=39', 't=0'])
    print('Five comparative figures saved.', flush=True)
