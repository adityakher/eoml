"""Training and evaluation loops for the EuroSAT classifier."""

import torch
from torch import nn, optim


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, preprocess=None, augment=None) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if preprocess is not None:
            x = preprocess(x)
        if augment is not None:
            x = augment(x)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss


@torch.no_grad()
def evaluate(model, dataloader, device, preprocess=None) -> float:
    """Return accuracy over a dataloader."""
    model.eval()
    correct = 0.0
    for batch in dataloader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if preprocess is not None:
            x = preprocess(x)
        y_hat = model(x)
        correct += (y_hat.argmax(1) == y).float().sum().item()
    return correct / len(dataloader.dataset)


@torch.no_grad()
def predict(model, dataloader, device, preprocess=None):
    """Return (labels, predictions) as numpy arrays over a dataloader."""
    model.eval()
    all_labels, all_preds = [], []
    for batch in dataloader:
        x = batch["image"].to(device)
        if preprocess is not None:
            x = preprocess(x)
        all_preds.append(model(x).argmax(1).cpu())
        all_labels.append(batch["label"])
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def fit(model, dataloaders, device, epochs: int = 3, lr: float = 1e-2, preprocess=None, augment=None):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss = train_one_epoch(
            model, dataloaders["train"], loss_fn, optimizer, device, preprocess, augment
        )
        acc = evaluate(model, dataloaders["val"], device, preprocess)
        print(f"Epoch {epoch}: loss={loss:.2f}, val accuracy={acc:.0%}")
    return model
