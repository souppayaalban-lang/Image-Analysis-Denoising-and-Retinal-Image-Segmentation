import os
import torch
import matplotlib.pyplot as plt

import training as tr

# =============================================================================
# CONFIGURATION
# =============================================================================
NOMS_DES_MODELES = ["FCNN", "AutoEncoder", "UNet"]  # Liste des modèles à entraîner et comparer
TYPES_DE_BRUIT = ["gaussien", "poivre_et_sel"]      # ajouter "poivre_et_sel" pour comparer les deux bruits
LOSS_NAME = "SSIM"                  # ou "SSIM"
OPTIM_NAME = "Adam"                # ou "SGD"
DOSSIER_SORTIE = "./resultats"
NB_IMAGES_A_AFFICHER = 8

os.makedirs(DOSSIER_SORTIE, exist_ok=True)


# =============================================================================
# Courbes : pertes et métriques au fil des epochs
# =============================================================================
def tracer_courbes(histories, noise_type):
    """histories : {nom_modele: history}. Une figure pour les pertes, une pour les métriques."""
    # --- Pertes train / validation, un graphique par modèle ---
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4))
    axes = [axes] if len(histories) == 1 else axes
    for ax, (nom, h) in zip(axes, histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="Train")
        ax.plot(epochs, h["val_loss"], label="Validation")
        ax.set_title(nom)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Perte ({LOSS_NAME})")
        ax.legend()
    plt.suptitle(f"Courbes de perte - bruit {noise_type}")
    plt.tight_layout()
    plt.savefig(f"{DOSSIER_SORTIE}/pertes_{noise_type}.png", dpi=150)
    plt.show()

    # --- Métriques de validation, modèles comparés sur le même graphique ---
    metriques = [("val_mse", "MSE"), ("val_psnr", "PSNR (dB)"), ("val_ssim", "SSIM")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (cle, titre) in zip(axes, metriques):
        for nom, h in histories.items():
            ax.plot(range(1, len(h[cle]) + 1), h[cle], label=nom)
        ax.set_title(titre)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.suptitle(f"Métriques sur la validation - bruit {noise_type}")
    plt.tight_layout()
    plt.savefig(f"{DOSSIER_SORTIE}/metriques_{noise_type}.png", dpi=150)
    plt.show()


# =============================================================================
# Visualisation : propre / bruitée / débruitée par chaque modèle
# =============================================================================
@torch.no_grad()
def visualiser_debruitage(modeles, test_loader, noise_type):
    images, _ = next(iter(test_loader))
    images = images[:NB_IMAGES_A_AFFICHER].to(tr.device)

    # Même bruit pour tous les modèles pour que la comparaison soit juste
    torch.manual_seed(0)
    bruitees = tr.bruiter(images, noise_type)

    lignes = [("Propre", images), ("Bruitée", bruitees)]
    for nom, modele in modeles.items():
        modele.eval()
        lignes.append((nom, modele(bruitees)))

    fig, axes = plt.subplots(len(lignes), NB_IMAGES_A_AFFICHER,
                             figsize=(1.6 * NB_IMAGES_A_AFFICHER, 1.7 * len(lignes)))
    for i, (titre, groupe) in enumerate(lignes):
        for j in range(NB_IMAGES_A_AFFICHER):
            axes[i, j].imshow(groupe[j].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if j == 0:
                axes[i, j].set_ylabel(titre)

    plt.suptitle(f"Débruitage - bruit {noise_type}")
    plt.tight_layout()
    plt.savefig(f"{DOSSIER_SORTIE}/debruitage_{noise_type}.png", dpi=150)
    plt.show()


# =============================================================================
# Comparaison des modèles sur le jeu de test
# =============================================================================
def comparer_modeles(resultats_test, noise_type):
    print(f"\n=== Résultats sur le jeu de test (bruit {noise_type}) ===")
    print(f"{'Modèle':<14}{'MSE':>10}{'PSNR (dB)':>12}{'SSIM':>10}")
    for nom, m in resultats_test.items():
        print(f"{nom:<14}{m['mse']:>10.5f}{m['psnr']:>12.2f}{m['ssim']:>10.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (cle, titre) in zip(axes, [("mse", "MSE (plus bas = mieux)"),
                                       ("psnr", "PSNR en dB (plus haut = mieux)"),
                                       ("ssim", "SSIM (plus haut = mieux)")]):
        valeurs = [resultats_test[nom][cle] for nom in resultats_test]
        ax.bar(list(resultats_test.keys()), valeurs)
        ax.set_title(titre)
    plt.suptitle(f"Comparaison sur le test - bruit {noise_type}")
    plt.tight_layout()
    plt.savefig(f"{DOSSIER_SORTIE}/comparaison_{noise_type}.png", dpi=150)
    plt.show()


# =============================================================================
# Programme principal : entraînement + évaluation de chaque modèle
# =============================================================================
if __name__ == "__main__":
    for noise_type in TYPES_DE_BRUIT:
        modeles = {}
        histories = {}
        resultats_test = {}

        for nom in NOMS_DES_MODELES:
            modele, history, test_loader = tr.train_model(
                model_name=nom,
                loss_name=LOSS_NAME,
                optim_name=OPTIM_NAME,
                noise_type=noise_type,
            )
            modeles[nom] = modele
            histories[nom] = history

            # Évaluation sur le test, avec le même tirage de bruit pour tous les modèles
            torch.manual_seed(0)
            resultats_test[nom] = tr.evaluate(modele, test_loader, noise_type)

            torch.save(modele.state_dict(), f"{DOSSIER_SORTIE}/{nom}_{noise_type}.pth")

        tracer_courbes(histories, noise_type)
        visualiser_debruitage(modeles, test_loader, noise_type)
        comparer_modeles(resultats_test, noise_type)

    print("\n✨ Entraînement et évaluation terminés. Figures dans :", DOSSIER_SORTIE)