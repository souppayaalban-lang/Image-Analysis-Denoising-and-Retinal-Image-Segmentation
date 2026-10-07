import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# =============================================================================
# ÉTAPE 1 : Charger et préparer les images
# =============================================================================

transformations = transforms.Compose([
    # Redimentionnement des images en 32x32 pixels
    transforms.Resize((32, 32)),
    # Conversion en Tenseur et Normalisation
    transforms.ToTensor(),
])

""" Téléchargement du jeu de Test MNIST : """

# Données d'apprentissage (60 000 images -> train=True)
donnees_apprentissage_complet = datasets.MNIST(
    root="./data", train=True, download=True, transform=transformations
)
# Données de test (10 000 images -> train=False)
donnees_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transformations
)


# =============================================================================
# ÉTAPE 2 : Séparer les données en données d'entrainement, de validation et de test
# =============================================================================

# On prend un échantillon assez grand (50 000) pour les images d'entrainement parmis les 60 000
taille_train = 50_000
# On garde un échantillon pour les données de validation (10 000)
taille_validation = len(donnees_apprentissage_complet) - taille_train  # 10 000

# On découpe les données d'apprentissage en données d'entrainement et données de validation
donnees_train, donnees_validation = random_split(
    donnees_apprentissage_complet,
    [taille_train, taille_validation],
    generator=torch.Generator().manual_seed(42) # Seed RNG
)

# Affichage pour vérifier les données
print("Nombre d'images :")
print(f"  train      : {len(donnees_train)}")
print(f"  validation : {len(donnees_validation)}")
print(f"  test       : {len(donnees_test)}")


# =============================================================================
# ÉTAPE 3 : Dataloaders
# =============================================================================
# Un Loader est l'outil qui "livre" les images à l'IA

# On ne donne pas les 50 000 images d'un coup à l'IA : on les donne par "Batch" pour éviter que ça soit trop lourd pour la mémoire
# La taille du Batch définis le nombre d'images envoyés à chaque tour
taille_batch = 32

# Pour l'entrainement c'est utile de mélanger les images pour le modèle,
# par contre pour la validation et le test, l'ordre des images importe peu vu que le modèle est "finis"
train_loader = DataLoader(donnees_train, batch_size=taille_batch, shuffle=True)
validation_loader = DataLoader(donnees_validation, batch_size=taille_batch, shuffle=False)
test_loader = DataLoader(donnees_test, batch_size=taille_batch, shuffle=False)


# =============================================================================
# ÉTAPE 4 : Fonctions de bruit
# =============================================================================

def ajouter_bruit_gaussien(image, moyenne=0.0, ecart_type=0.1):
    bruit = torch.randn_like(image) * ecart_type + moyenne
    image_bruitee = image + bruit
    # On force les pixels à rester entre 0 et 1 (Normalisation des pixels)
    return torch.clamp(image_bruitee, 0.0, 1.0)


def ajouter_bruit_poivre_et_sel(image, probabilite=0.05):
    # On tire un nombre au hasard entre 0 et 1
    tirages = torch.rand_like(image)
    # On copie l'image pour ne pas corrompre l'image d'origine
    image_bruitee = image.clone()
    image_bruitee[tirages < probabilite / 2] = 0.0
    image_bruitee[tirages > 1.0 - probabilite / 2] = 1.0
    return image_bruitee


# =============================================================================
# ÉTAPE 5 : Fonctions d'applatissement
# =============================================================================

# Une image est tableaux 2D, on a alors besoin de le convertir en tableau 1D (1*32*32 = 1024 entrées)
def flatten(images):
    """(nb_images, canaux, hauteur, largeur) -> (nb_images, canaux*hauteur*largeur)"""
    return images.flatten(start_dim=1)  # start_dim=1 : on garde la dimension "nb_images"


def deflatten(images, canaux, hauteur, largeur):
    """(nb_images, canaux*hauteur*largeur) -> (nb_images, canaux, hauteur, largeur)"""
    return images.reshape(-1, canaux, hauteur, largeur)


# =============================================================================
# ÉTAPE 6 : VÉRIFIER QUE TOUT FONCTIONNE
# =============================================================================

# On récupère un Batch avec le tenseur (image) et la valeur (le chiffre à l'intérieur)
images, etiquettes = next(iter(train_loader))

# On vérifie que la forme du Batch [Taille, Canaux, Hauteur, Larguer], ici ça devrait être [32, 1, 32, 32]
print("\nForme d'un batch :", tuple(images.shape), "-> (images, canaux, hauteur, largeur)")
# On vérifie la normalisation des pixels
print(f"Normalisation des pixels : {images.min():.2f} | max : {images.max():.2f}")
# On vérifie le format après aplatissement, ici on devrait avoir 1024
images_aplaties = flatten(images)
print("Après aplatissement :", tuple(images_aplaties.shape), "-> 32 images de 1024 nombres")
# On vérifie que le format est correct lorsqu'on les remets au format initial
images_restaurees = deflatten(images_aplaties, 1, 32, 32)
print("Après restauration  :", tuple(images_restaurees.shape))

# On vérifie que les images restaurées sont strictement identiques aux originales
print("Restauration correcte ?", torch.equal(images, images_restaurees))

""""
# =============================================================================
# ÉTAPE 7 : VISUALISER LES DONNÉES
# =============================================================================
# On regarde enfin les images : original, bruit gaussien, bruit poivre et sel.

# Nombre d'images à afficher lors de la visualisation
nombre_a_afficher = 10
originales = images[:nombre_a_afficher]                 # Images Originales
gaussiennes = ajouter_bruit_gaussien(originales)        # Images avec bruit Gaussien
poivre_sel = ajouter_bruit_poivre_et_sel(originales)    # Images avec bruit Poivre et Sel

# Structure d'affichage
lignes = [
    ("Originale", originales),
    ("Bruit gaussien", gaussiennes),
    ("Poivre et sel", poivre_sel),
]

# On crée des plots sur 3 lignes et 10 colonnes
figure, axes = plt.subplots(3, nombre_a_afficher, figsize=(12, 5))

# On affiche chaque images
for i, (titre, groupe_d_images) in enumerate(lignes):
    for j in range(nombre_a_afficher):
        axes[i, j].imshow(groupe_d_images[j].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        if j == 0:
            axes[i, j].set_ylabel(titre)

# Au-dessus de la 1re ligne : le vrai chiffre de chaque image
for j in range(nombre_a_afficher):
    axes[0, j].set_title(f"Chiffre {etiquettes[j].item()}")

plt.suptitle("Images MNIST : originales et bruitées")
plt.tight_layout()
plt.show()
"""