# Rpicamera
Interface camera Raspberry Pi optimisée IMX585

# RPiCamera - Interface de contrôle avancée pour caméras Raspberry Pi

**Auteur original** : [Gordon999](https://github.com/Gordon999)

**Version spéciale IMX585** : Cette version est spécialement optimisée pour le capteur IMX585 et fonctionne avec le libcamera custom développé par [will12753](https://github.com/will12753/libcamera-imx585).

## Description

RPiCamera est une application complète de contrôle et d'enregistrement pour caméras Raspberry Pi, développée avec une interface graphique Pygame. Le programme offre un contrôle avancé des paramètres de la caméra, particulièrement optimisé pour l'astrophotographie avec le capteur IMX585.

### Capteur IMX585

Le capteur IMX585 utilisé dans ce projet provient de [SOHO Enterprise](https://soho-enterprise.com/). Ce capteur haute sensibilité est particulièrement adapté pour l'astronomie et les applications en faible luminosité.

### Fonctionnalités principales

- **Interface graphique interactive** : Contrôle complet via une interface Pygame
- **Modes d'enregistrement multiples** :
  - Enregistrement vidéo (H.264, MJPEG, YUV420 et SER pour l'imagerie planétaire)
  - Capture d'images (RAW, JPG, PNG, RGB, YUV)
  - Time-lapse automatisé
- **Streaming vidéo** : Support TCP, UDP et RTSP
- **Analyse en temps réel** :
  - Histogrammes (RGB et luminance)
  - Calcul du HFR (Half-Flux Radius) pour l'aide à la mise au point astrophotographie
  - Détection de centroïdes
- **Contrôles avancés** :
  - Focus manuel/automatique (pour caméras compatibles)
  - Balance des blancs personnalisable
  - Contrôle de l'exposition et du gain
  - Modes de mesure de lumière
  - Correction gamma et débruitage
- **Support GPIO** : Boutons externes pour focus et déclenchement
- **Post-traitement** : Correction automatique des timestamps vidéo via ffmpeg



Le live stacking peut être utilisé en combinant ce programme avec **ALS (Astro Live Stacker)** :

- **ALS (Astro Live Stacker)** : [https://github.com/gehelem/als](https://github.com/gehelem/als)

ALS permet d'empiler les images en temps réel pour révéler des objets célestes faibles, en combinaison avec les flux vidéo générés par RPiCamera.


## Dépendances

### Bibliothèques Python

Installez les dépendances Python avec pip :

```bash
pip3 install pygame opencv-python numpy matplotlib picamera2 gpiozero
```

Détail des bibliothèques :

- **pygame** : Interface graphique et gestion des événements
- **opencv-python (cv2)** : Traitement d'image et analyse
- **numpy** : Calculs numériques et manipulation de tableaux
- **matplotlib** : Génération de graphiques (histogrammes)
- **picamera2** : Interface moderne pour les caméras Raspberry Pi
- **libcamera** : Bibliothèque de contrôle caméra (voir section libcamera custom ci-dessous)
- **gpiozero** : Contrôle des GPIO pour boutons externes


### Outils système

Installez les outils système nécessaires :


sudo apt-get install -y ffmpeg
```

- **ffmpeg** : Conversion et post-traitement vidéo

### Bibliothèques système supplémentaires


sudo apt-get install -y libcamera-dev libcamera-apps
```

### Structure des fichiers

- **Photos** : Enregistrées dans `~/Pictures/`-> je vous conseille de stocker les RAW directement sur clés USB: -  pic_dir     = "/media/admin/..."
- **Vidéos** : Enregistrées dans `~/Videos/` -> gardez les videos sur la mémoire interne
- **Configuration** : `~/PiLCConfig104.txt`


### Fichiers de tuning caméra

Pour les caméras spécialisées, vous pouvez utiliser des fichiers de tuning personnalisés :

- `~/imx290a.json` : Configuration pour IMX290
- `~/imx585_lowlight.json` : Configuration pour IMX585 en mode low-light



**Toutes les contributions sont les bienvenues !**

Le développement de ce programme est ouvert à la communauté. N'hésitez pas à :

- Proposer de nouvelles fonctionnalités
- Corriger des bugs
- Ajouter le support de nouveaux capteurs
- Développer l'intégration d'un live stacking natif


## Licence

Copyright (c) 2025 Gordon999

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Ressources et liens utiles

- **Programme original** : [Gordon999](https://github.com/Gordon999)
- **Libcamera custom IMX585** : [will12753/libcamera-imx585](https://github.com/will12753/libcamera-imx585)
- **Capteur IMX585** : [SOHO Enterprise](https://soho-enterprise.com/)
- **Live Stacking (ALS)** : [Astro Live Stacker](https://github.com/gehelem/als)

