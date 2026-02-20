#!/usr/bin/env python3
"""
JSK LIVE - Module de traitement HDR et Denoise pour RPi5

Pipeline: RAW12 -> Stack (1-4) -> HDR -> Debayer -> Denoise -> Stretch -> 8bits

Adapté du code HDR original (cupy) pour fonctionner sur RPi5 (numpy CPU)
"""

import numpy as np
import cv2


# ============================================================================
# HDR Processing - Adapté pour RAW 12 bits et RPi5 (numpy au lieu de cupy)
# ============================================================================

def HDR_compute_12bit(image_12b, method="Median", bits_to_clip=2, type_bayer=cv2.COLOR_BayerRG2RGB, weights=None):
    """
    Crée une image HDR à partir d'une seule frame RAW 12 bits.
    Génère (bits_to_clip + 1) frames virtuelles par retrait successif
    des bits de poids fort, puis les fusionne.

    Avec bits_to_clip=2, on obtient 3 images :
        12-bit : seuil = 4095 (image originale)
        11-bit : seuil = 2047 (1 bit retiré)
        10-bit : seuil = 1023 (2 bits retirés)

    Args:
        image_12b: Image RAW 12 bits (numpy array, 1 canal Bayer)
        method: Méthode de fusion - "Median", "Mean", ou "Mertens"
        bits_to_clip: Nombre de bits de poids fort à retirer (1-3)
        type_bayer: Pattern Bayer pour debayering (cv2.COLOR_BayerXX2RGB)
        weights: Liste de poids (0-100) pour chaque image, longueur = bits_to_clip+1.
                 Utilisé pour la méthode Mean (moyenne pondérée).
                 None = poids égaux.

    Returns:
        Image HDR 8 bits RGB (numpy array)
    """
    image_float = image_12b.astype(np.float32)
    n_images = bits_to_clip + 1

    # Générer les seuils par retrait réel de bits de poids fort
    thresholds = []
    for i in range(n_images):
        bit_depth = 12 - i
        thresholds.append(2 ** bit_depth - 1)

    # Générer les images clippées (du seuil le plus haut au plus bas)
    img_list = []
    for thres in thresholds:
        img_clipped = image_float.copy()
        img_clipped[img_clipped > thres] = thres
        img_8b = (img_clipped / thres * 255.0).astype(np.uint8)
        img_list.append(img_8b)

    # Préparer les poids normalisés pour la fusion
    if weights is not None:
        w = [max(0, w) for w in weights[:n_images]]
    else:
        w = [100] * n_images
    w_sum = sum(w)
    if w_sum == 0:
        w = [1.0] * n_images
        w_sum = float(n_images)
    w_norm = [float(x) / w_sum for x in w]

    # Fusionner selon la méthode choisie
    if method == "Mertens":
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        HDR_image_gray = np.clip(res_mertens * 255, 0, 255).astype(np.uint8)
    elif method == "Median":
        img_stack = np.stack(img_list, axis=0)
        HDR_image_gray = np.median(img_stack, axis=0).astype(np.uint8)
    elif method == "Mean":
        # Moyenne pondérée par les poids utilisateur
        img_stack = np.stack(img_list, axis=0).astype(np.float32)
        HDR_image_gray = np.average(img_stack, axis=0, weights=w_norm).astype(np.uint8)
    else:
        # Fallback: pas de HDR, juste normalisation
        HDR_image_gray = (image_float / 4095.0 * 255.0).astype(np.uint8)

    # Debayering vers RGB
    HDR_image_rgb = cv2.cvtColor(HDR_image_gray, type_bayer)

    return HDR_image_rgb


def HDR_bypass_12bit(image_12b, type_bayer=cv2.COLOR_BayerRG2RGB):
    """
    Bypass HDR: simple conversion 12bits -> 8bits avec debayering.
    Utilisé quand HDR method = OFF.

    Args:
        image_12b: Image RAW 12 bits
        type_bayer: Pattern Bayer

    Returns:
        Image 8 bits RGB
    """
    # Simple scaling 12 bits -> 8 bits
    image_8b = (image_12b.astype(np.float32) / 4095.0 * 255.0).astype(np.uint8)
    # Debayering
    return cv2.cvtColor(image_8b, type_bayer)


# ============================================================================
# Stacking - Moyenne simple de N images
# ============================================================================

def stack_images(images):
    """
    Empile plusieurs images par moyenne.

    Args:
        images: Liste d'images numpy (même dimensions)

    Returns:
        Image moyennée (même type que l'entrée)
    """
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]

    # Stack et moyenne
    stack = np.stack(images, axis=0).astype(np.float32)
    mean_img = np.mean(stack, axis=0)

    # Retourner au type original
    return mean_img.astype(images[0].dtype)


# ============================================================================
# Denoise Filters
# ============================================================================

def denoise_bilateral(image, strength=5):
    """
    Filtre bilatéral - préserve les bords.

    Args:
        image: Image RGB 8 bits
        strength: Intensité (1-10)

    Returns:
        Image filtrée
    """
    # Paramètres adaptés à l'intensité
    d = 5 + strength  # Diamètre du voisinage
    sigma_color = 10 + strength * 8  # Sigma couleur
    sigma_space = 10 + strength * 8  # Sigma spatial

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_nlm(image, strength=5):
    """
    Non-Local Means - très efficace mais plus lent.

    Args:
        image: Image RGB 8 bits
        strength: Intensité (1-10)

    Returns:
        Image filtrée
    """
    # h = force du filtrage (3-15 typique)
    h = 3 + strength * 1.2

    # FastNlMeansDenoisingColored pour images couleur
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)


def denoise_gaussian(image, strength=5):
    """
    Flou gaussien - très rapide mais floute les détails.

    Args:
        image: Image RGB 8 bits
        strength: Intensité (1-10)

    Returns:
        Image filtrée
    """
    # Taille du kernel (doit être impair)
    ksize = 3 + (strength // 2) * 2
    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def denoise_median(image, strength=5):
    """
    Filtre médian - bon contre le bruit sel/poivre.

    Args:
        image: Image RGB 8 bits
        strength: Intensité (1-10)

    Returns:
        Image filtrée
    """
    # Taille du kernel (doit être impair)
    ksize = 3 + (strength // 3) * 2
    if ksize % 2 == 0:
        ksize += 1

    return cv2.medianBlur(image, ksize)


def apply_denoise(image, denoise_type, strength=5):
    """
    Applique le filtre de denoise sélectionné.

    Args:
        image: Image RGB 8 bits
        denoise_type: 0=OFF, 1=Bilateral, 2=FastNLM, 3=Gaussian, 4=Median
        strength: Intensité (1-10)

    Returns:
        Image filtrée (ou originale si OFF)
    """
    if denoise_type == 0:
        return image
    elif denoise_type == 1:
        return denoise_bilateral(image, strength)
    elif denoise_type == 2:
        return denoise_nlm(image, strength)
    elif denoise_type == 3:
        return denoise_gaussian(image, strength)
    elif denoise_type == 4:
        return denoise_median(image, strength)
    else:
        return image


# ============================================================================
# Pipeline complet JSK LIVE
# ============================================================================

class JSKLiveProcessor:
    """
    Processeur JSK LIVE avec buffer de stacking.
    """

    def __init__(self):
        # Paramètres par défaut
        self.stack_count = 1       # 1-6 images à empiler
        self.hdr_bits_clip = 2     # 1-3 bits à clipper
        self.hdr_method = 1        # 0=OFF, 1=Median, 2=Mean, 3=Mertens
        self.denoise_type = 0      # 0=OFF, 1=Bilateral, 2=NLM, 3=Gaussian, 4=Median
        self.denoise_strength = 5  # 1-10
        self.hdr_weights = [100, 100, 100, 100, 100, 100]  # Poids HDR par niveau de bit (0-100)
        self.bayer_pattern = cv2.COLOR_BayerRG2RGB
        self.crop_square = False   # True = crop carré 1080×1080 centré avant traitement

        # Buffer pour le stacking
        self.frame_buffer = []

        # Méthodes HDR (pour affichage)
        self.hdr_methods = ["OFF", "Median", "Mean", "Mertens"]
        self.denoise_types = ["OFF", "Bilateral", "FastNLM", "Gaussian", "Median"]

    def configure(self, **kwargs):
        """Configure les paramètres du processeur."""
        if 'stack_count' in kwargs:
            self.stack_count = max(1, min(6, kwargs['stack_count']))
        if 'hdr_bits_clip' in kwargs:
            self.hdr_bits_clip = max(0, min(3, kwargs['hdr_bits_clip']))
        if 'hdr_method' in kwargs:
            self.hdr_method = max(0, min(3, kwargs['hdr_method']))
        if 'denoise_type' in kwargs:
            self.denoise_type = max(0, min(4, kwargs['denoise_type']))
        if 'denoise_strength' in kwargs:
            self.denoise_strength = max(1, min(10, kwargs['denoise_strength']))
        if 'hdr_weights' in kwargs:
            self.hdr_weights = list(kwargs['hdr_weights'][:4])
        if 'bayer_pattern' in kwargs:
            self.bayer_pattern = kwargs['bayer_pattern']
        if 'crop_square' in kwargs:
            self.crop_square = bool(kwargs['crop_square'])

    def clear_buffer(self):
        """Vide le buffer de frames."""
        self.frame_buffer = []

    def add_frame(self, raw_frame):
        """
        Ajoute une frame au buffer.
        Retourne True si le buffer est plein.
        """
        self.frame_buffer.append(raw_frame.copy())

        # Limiter la taille du buffer
        while len(self.frame_buffer) > self.stack_count:
            self.frame_buffer.pop(0)

        return len(self.frame_buffer) >= self.stack_count

    def process(self, raw_frame=None):
        """
        Traite une frame RAW 12 bits avec le pipeline complet.
        Pipeline: RAW12 → HDR (clip) → Debayer → Stack → Denoise

        Args:
            raw_frame: Frame RAW 12 bits (optionnel, utilise le buffer sinon)

        Returns:
            Image RGB 8 bits traitée, ou None si buffer incomplet
        """
        if raw_frame is None:
            return None

        # 0. Crop carré centré sur le RAW (avant tout traitement pour gain de performance)
        if self.crop_square:
            h, w = raw_frame.shape[:2]
            crop_size = min(h, w)          # = 1080 sur capteur 1920×1080
            x_start = (w - crop_size) // 2
            # S'assurer que x_start est pair pour préserver l'alignement Bayer
            if x_start % 2 != 0:
                x_start += 1
            raw_frame = raw_frame[:crop_size, x_start:x_start + crop_size]

        # 1. HDR Processing sur la frame RAW brute
        if self.hdr_method == 0 or self.hdr_bits_clip == 0:
            # HDR OFF ou clip=0 - simple conversion 12bit->8bit sans clipping
            rgb_image = HDR_bypass_12bit(raw_frame, self.bayer_pattern)
        else:
            method_name = self.hdr_methods[self.hdr_method]
            rgb_image = HDR_compute_12bit(
                raw_frame,
                method=method_name,
                bits_to_clip=self.hdr_bits_clip,
                type_bayer=self.bayer_pattern,
                weights=self.hdr_weights
            )

        # 2. Ajouter l'image RGB traitée au buffer pour stacking
        self.frame_buffer.append(rgb_image.copy())
        while len(self.frame_buffer) > self.stack_count:
            self.frame_buffer.pop(0)

        # 3. Stacking (si stack_count > 1 et buffer suffisant)
        if len(self.frame_buffer) >= self.stack_count:
            if self.stack_count > 1:
                stacked = stack_images(self.frame_buffer[-self.stack_count:])
            else:
                stacked = self.frame_buffer[-1]
        else:
            # Buffer pas encore plein, utiliser l'image courante
            stacked = rgb_image

        # 4. Denoise
        result = apply_denoise(stacked, self.denoise_type, self.denoise_strength)

        return result

    def process_single(self, raw_frame):
        """
        Traite une seule frame avec le pipeline complet (incluant buffer de stacking).
        Pipeline: RAW12 → HDR (clip) → Debayer → Stack → Denoise
        """
        return self.process(raw_frame)


# ============================================================================
# Video Recording Helper
# ============================================================================

class JSKVideoRecorder:
    """
    Enregistreur vidéo MP4 pour JSK LIVE.
    """

    def __init__(self):
        self.writer = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        self.output_path = None
        self.fps = 25  # FPS par défaut

    def start(self, output_path, width, height, fps=25):
        """
        Démarre l'enregistrement.

        Args:
            output_path: Chemin du fichier MP4
            width, height: Dimensions de la vidéo
            fps: Images par seconde
        """
        import time

        self.fps = fps
        self.output_path = output_path
        self.rec_width = width
        self.rec_height = height

        # Codec H264
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if self.writer.isOpened():
            self.is_recording = True
            self.frame_count = 0
            self.start_time = time.time()
            return True
        else:
            self.writer = None
            return False

    def write_frame(self, frame_rgb):
        """
        Écrit une frame (RGB) dans la vidéo.
        Redimensionne automatiquement si les dimensions ne correspondent pas.
        """
        if not self.is_recording or self.writer is None:
            return False

        # Convertir RGB -> BGR pour OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Redimensionner si les dimensions ne correspondent pas au VideoWriter
        h, w = frame_bgr.shape[:2]
        if w != self.rec_width or h != self.rec_height:
            frame_bgr = cv2.resize(frame_bgr, (self.rec_width, self.rec_height))

        self.writer.write(frame_bgr)
        self.frame_count += 1
        return True

    def stop(self):
        """
        Arrête l'enregistrement et finalise le fichier.
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.is_recording = False
        return self.output_path

    def get_elapsed_time(self):
        """Retourne le temps écoulé en secondes."""
        import time
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def get_elapsed_str(self):
        """Retourne le temps écoulé formaté MM:SS."""
        elapsed = int(self.get_elapsed_time())
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02d}:{seconds:02d}"
