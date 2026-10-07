"""
IMPLÉMENTATION DES MODÈLES - FCNN, Autoencoder, U-Net (version pour débutants)

Les trois modèles font exactement la même chose vu de l'extérieur :
    ENTRÉE : un batch d'images bruitées  de forme (N, 1, 32, 32)
    SORTIE : un batch d'images débruitées de forme (N, 1, 32, 32)
        (N = nombre d'images dans le batch, 1 = canal noir et blanc, 32x32 pixels)

Seul l'intérieur change. Dans chaque modèle on trouve toujours deux méthodes :
    __init__ : on CONSTRUIT les briques (les couches) du réseau, une seule fois ;
    forward  : on dit COMMENT l'image traverse ces briques, dans quel ordre.
"""

import torch
import torch.nn as nn
import prepare_data as prep


# =============================================================================
# MODÈLE 1 : FCNN (réseau entièrement connecté)
# =============================================================================
class FCNN(nn.Module):
    def __init__(self, format_image=(1, 32, 32)):
        super().__init__()
        self.forme_image = format_image
        # Puisque qu'on va aplatir l'image il nous faut la taille du tableau 1D
        taille = format_image[0] * format_image[1] * format_image[2]

        # nn.Sequential = une chaîne de couches exécutées dans l'ordre
        # A chaque couche, on compresse l'image puis on la décompresse
        # La fonction d'activation est ReLu
        # A la fin on reshape l'image avec une fonction Sigmoid
        self.reseau = nn.Sequential(
            nn.Linear(taille, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, taille),
            nn.Sigmoid(),
        )

    def forward(self, images):
        # Input : Images aplaties (N, 1, 32, 32) -> (N, 1024)
        x = x = prep.flatten(images)
        # Couches cachées
        x = self.reseau(x)
        # Output : On reshape l'image sous sa forme d'origine (N, 1024) -> (N, 1, 32, 32)
        return x.reshape(-1, *self.forme_image)


# =============================================================================
# MODÈLE 2 : AUTOENCODER (convolutif)
# =============================================================================
class AutoEncoder(nn.Module):
    def __init__(self, format_image=(1, 32, 32)):
        super().__init__()
        self.forme_image = format_image
        taille = format_image[0] * format_image[1] * format_image[2]

        self.encoder = nn.Sequential(
            nn.Linear(taille, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.latent_space = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, taille),
            nn.Sigmoid()
        )

    def forward(self, images):
        x = prep.flatten(images)

        x = self.encoder(x)
        x = self.latent_space(x)
        x = self.decoder(x)

        return x.reshape(-1, *self.forme_image)


# =============================================================================
# MODÈLE 3 : U-NET
# =============================================================================

"""
    - input_channel = canaux d'entrée (1 pour une image de gris, 3 pour les images couleurs)
    - output_channel = canaux de sortie
    - kernel_size = dimension de la convolution 1x1, 2x2, 3x3
    
    - nn.Con2d(n_input, n_output, padding) pour effectuer une convolution de n_input canaux qui va produire n_output canaux
    - nn.ConvTranspose2d
"""

def crop(x, target):
    ecart_h = x.shape[2] - target.shape[2]
    ecart_l = x.shape[3] - target.shape[3]
    top = ecart_h // 2
    left = ecart_l // 2
    return x[:, :, top: top + target.shape[2], left: left + target.shape[3]]

# Classe pour faire une double convolution 
class DoubleConvolution(nn.Module):
    def __init__(self, input_channel, output_channel, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=1, padding=1):
        super().__init__()
        # Correpond aux flèches rouges pour le pooling Max pool 2x2
        self.pool = nn.MaxPool2d(2)

        self.encoder1 = DoubleConvolution(input_channel, 64, padding)
        self.encoder2 = DoubleConvolution(64, 128, padding)
        self.encoder3 = DoubleConvolution(128, 256, padding)
        self.encoder4 = DoubleConvolution(256, 512, padding)

        self.last_layer = DoubleConvolution(512, 1024, padding)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvolution(1024, 512, padding)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvolution(512, 256, padding)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvolution(256, 128, padding)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvolution(128, 64, padding)

        self.output = nn.Conv2d(64, output_channel, kernel_size=1)

    def forward(self, x):
            e1 = self.encoder1(x)
            e2 = self.encoder2(self.pool(e1))
            e3 = self.encoder3(self.pool(e2))
            e4 = self.encoder4(self.pool(e3))
    
            # Fond
            f = self.last_layer(self.pool(e4))
    
            # Copy then Crop and decode
            d4 = self.up4(f)
            d4 = self.decoder4(torch.cat([crop(e4, d4), d4], dim=1))
            d3 = self.up3(d4)
            d3 = self.decoder3(torch.cat([crop(e3, d3), d3], dim=1))
            d2 = self.up2(d3)
            d2 = self.decoder2(torch.cat([crop(e2, d2), d2], dim=1))
            d1 = self.up1(d2)
            d1 = self.decoder1(torch.cat([crop(e1, d1), d1], dim=1))
    
            return torch.sigmoid(self.output(d1)) 


