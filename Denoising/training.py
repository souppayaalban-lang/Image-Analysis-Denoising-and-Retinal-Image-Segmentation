import models
import prepare_data as prep

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
import torch_directml

device = torch_directml.device()

# Paramètres d'entraînement propres à chaque modèle
CONFIG_MODELES = {
    "FCNN":        {"epochs": 10, "batch_size": 64, "lr": 1e-3},
    "AutoEncoder": {"epochs": 10, "batch_size": 64, "lr": 1e-3},
    "UNet":        {"epochs": 10, "batch_size": 32, "lr": 1e-3},
}


# =============================================================================
# Fonctions de perte et optimiseurs
# =============================================================================
def loss_function(name):
    if name == "MSE":
        return nn.MSELoss()
    elif name == "SSIM":
        # Les images sont dans [0, 1] -> data_range=1.0
        return lambda preds, targets: 1.0 - structural_similarity_index_measure(
            preds, targets, data_range=1.0
        )
    else:
        raise ValueError(f"Fonction de perte inconnue : {name}")


def optim_method(name, model_parameters, lr):
    if name == "Adam":
        return torch.optim.Adam(model_parameters, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(model_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimiseur inconnu : {name}")


# =============================================================================
# Bruit, loaders
# =============================================================================
def bruiter(images, noise_type):
    if noise_type == "gaussien":
        return prep.ajouter_bruit_gaussien(images)
    elif noise_type == "poivre_et_sel":
        return prep.ajouter_bruit_poivre_et_sel(images)
    raise ValueError(f"Type de bruit inconnu : {noise_type}")


def make_loaders(batch_size):
    """Crée les loaders MNIST avec la taille de batch voulue pour le modèle."""
    train_loader = DataLoader(prep.donnees_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(prep.donnees_validation, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(prep.donnees_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# =============================================================================
# Évaluation (MSE, PSNR, SSIM) sur un loader
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, noise_type, criterion=None):
    """Renvoie les métriques moyennes par image : loss (si criterion), mse, psnr, ssim."""
    model.eval()
    total = {"loss": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0

    for images, _ in loader:
        images = images.to(device)
        outputs = model(bruiter(images, noise_type))
        bs = images.size(0)

        mse_img = ((outputs - images) ** 2).mean(dim=(1, 2, 3))
        total["mse"] += mse_img.sum().item()
        total["psnr"] += (10 * torch.log10(1.0 / (mse_img + 1e-10))).sum().item()
        total["ssim"] += structural_similarity_index_measure(
            outputs, images, data_range=1.0, reduction="none"
        ).sum().item()
        if criterion is not None:
            total["loss"] += criterion(outputs, images).item() * bs
        n += bs

    return {k: v / n for k, v in total.items()}


# =============================================================================
# Entraînement
# =============================================================================
def train_model(model_name, epochs=None, batch_size=None, lr=None,
                loss_name="MSE", optim_name="Adam", noise_type="gaussien"):
    # Valeurs par défaut prises dans CONFIG_MODELES
    cfg = CONFIG_MODELES[model_name]
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]
    lr = lr or cfg["lr"]

    train_loader, val_loader, test_loader = make_loaders(batch_size)

    # Initialisation du modèle selon son nom (classe de models.py)
    model = getattr(models, model_name)().to(device)
    criterion = loss_function(loss_name)
    optimizer = optim_method(optim_name, model.parameters(), lr)

    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_psnr": [], "val_ssim": []}

    print(f"\n==================================================")
    print(f"🚀 Début de l'entraînement : {model_name}")
    print(f"   {epochs} epochs | batch {batch_size} | lr {lr} | Loss: {loss_name} | Optim: {optim_name} | Bruit: {noise_type}")
    print(f"==================================================")

    for epoch in range(epochs):
        # --- ENTRAÎNEMENT ---
        model.train()
        running_train_loss = 0.0

        for original_images, _ in train_loader:
            original_images = original_images.to(device)
            images_bruitees = bruiter(original_images, noise_type)

            optimizer.zero_grad()
            outputs = model(images_bruitees)
            loss = criterion(outputs, original_images)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * original_images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- VALIDATION (perte + métriques à chaque epoch) ---
        val = evaluate(model, val_loader, noise_type, criterion)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mse"].append(val["mse"])
        history["val_psnr"].append(val["psnr"])
        history["val_ssim"].append(val["ssim"])

        print(f"Epoch [{epoch+1}/{epochs}] --> Train Loss: {epoch_train_loss:.4f} | Val Loss: {val['loss']:.4f} "
              f"| MSE: {val['mse']:.4f} | PSNR: {val['psnr']:.2f} dB | SSIM: {val['ssim']:.4f}")

    return model, history, test_loader