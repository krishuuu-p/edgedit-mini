import torch
import torch.nn as nn
import numpy as np
from scipy import linalg


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, feat_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_features=False):
        f = self.features(x).flatten(1)
        logits = self.classifier(f)
        if return_features:
            return logits, f
        return logits


def train_classifier(train_loader, device, epochs=3, lr=1e-3):
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"[classifier] epoch {ep+1}/{epochs} loss={loss_sum/total:.4f} acc={correct/total:.4f}")
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, images, device, batch_size=128):
    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i+batch_size].to(device)
        _, f = model(batch, return_features=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(feat_real, feat_fake):
    mu1, sigma1 = feat_real.mean(0), np.cov(feat_real, rowvar=False)
    mu2, sigma2 = feat_fake.mean(0), np.cov(feat_fake, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def proxy_fid(classifier, real_images, fake_images, device):
    f_real = extract_features(classifier, real_images, device)
    f_fake = extract_features(classifier, fake_images, device)
    return frechet_distance(f_real, f_fake)
