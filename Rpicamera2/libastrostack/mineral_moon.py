#!/usr/bin/env python3
"""
MINERAL MOON - Amplification des couleurs minéralogiques lunaires

Pipeline : RGB8 ISP (Picamera2 main stream)
  → BGR float32
  → Masque protection hautes terres
  → Boost saturation itératif (espace HSV)
  → [Réduction bruit bilatéral]
  → [Composite LRGB]
  → BGR uint8

Note WB : balance des blancs gérée en amont via Picamera2 ColourGains
          (AwbEnable=False + ColourGains=(r, b) dans RPiCamera2.py)
"""

import cv2
import numpy as np


# Presets prédéfinis
MOON_PRESETS = [
    # 0 - Subtil
    {
        'name': 'Subtil',
        'sat_factor': 1.8, 'sat_iter': 6, 'lum_protect': 230,
        'lrgb': False, 'lrgb_w': 0.6,
        'noise': True, 'noise_s': 7,
    },
    # 1 - Standard (défaut)
    {
        'name': 'Standard',
        'sat_factor': 2.5, 'sat_iter': 4, 'lum_protect': 220,
        'lrgb': False, 'lrgb_w': 0.6,
        'noise': True, 'noise_s': 7,
    },
    # 2 - Intense
    {
        'name': 'Intense',
        'sat_factor': 4.0, 'sat_iter': 3, 'lum_protect': 200,
        'lrgb': True, 'lrgb_w': 0.8,
        'noise': True, 'noise_s': 5,
    },
]


class MineralMoonProcessor:
    """
    Processeur Mineral Moon : amplifie les couleurs minéralogiques lunaires.

    Entrée  : frame BGR uint8 (depuis Picamera2 main stream, converti RGB→BGR)
    Sortie  : frame BGR uint8 traitée
    """

    def __init__(self):
        # Paramètres de traitement
        self.saturation_factor     = 2.5   # Multiplicateur total (1.0 – 6.0)
        self.saturation_iterations = 4     # Nombre de passes (1 – 8)
        self.luminance_protect     = 220   # Seuil V au-delà duquel pas de boost (0-255)
        self.noise_reduction       = True  # Filtre bilatéral post-boost
        self.noise_strength        = 7     # Force bilatéral (impair : 3, 5, 7, 9, 11)
        self.lrgb_composite        = False # Composite L (original) + RGB (colorisé)
        self.lrgb_color_weight     = 0.6  # Poids chrominance (0.1 – 1.0)

    def configure(self, **kwargs):
        """Met à jour les paramètres du processeur."""
        if 'saturation_factor' in kwargs:
            self.saturation_factor = max(1.0, min(6.0, float(kwargs['saturation_factor'])))
        if 'saturation_iterations' in kwargs:
            self.saturation_iterations = max(1, min(8, int(kwargs['saturation_iterations'])))
        if 'luminance_protect' in kwargs:
            self.luminance_protect = max(100, min(255, int(kwargs['luminance_protect'])))
        if 'noise_reduction' in kwargs:
            self.noise_reduction = bool(kwargs['noise_reduction'])
        if 'noise_strength' in kwargs:
            v = int(kwargs['noise_strength'])
            if v % 2 == 0:
                v += 1                     # Forcer valeur impaire
            self.noise_strength = max(3, min(11, v))
        if 'lrgb_composite' in kwargs:
            self.lrgb_composite = bool(kwargs['lrgb_composite'])
        if 'lrgb_color_weight' in kwargs:
            self.lrgb_color_weight = max(0.1, min(1.0, float(kwargs['lrgb_color_weight'])))

    def apply_preset(self, preset_idx):
        """Applique un preset (0=Subtil, 1=Standard, 2=Intense)."""
        if 0 <= preset_idx < len(MOON_PRESETS):
            p = MOON_PRESETS[preset_idx]
            self.saturation_factor     = p['sat_factor']
            self.saturation_iterations = p['sat_iter']
            self.luminance_protect     = p['lum_protect']
            self.lrgb_composite        = p['lrgb']
            self.lrgb_color_weight     = p['lrgb_w']
            self.noise_reduction       = p['noise']
            self.noise_strength        = p['noise_s']

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Traitement Mineral Moon sur une frame BGR uint8.
        Retourne une frame BGR uint8.
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return frame_bgr

        # 1. BGR → HSV (float32 pour éviter les débordements)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 2. Masque de protection des hautes terres (zones très lumineuses)
        #    Adouci par un flou gaussien pour éviter les artefacts aux transitions
        luminance    = hsv[:, :, 2]
        protect_mask = (luminance < self.luminance_protect).astype(np.float32)
        protect_mask = cv2.GaussianBlur(protect_mask, (21, 21), 0)

        # 3. Boost de saturation itératif
        #    Distribuer le boost total en N passes = résultat plus doux
        boost_per_iter = self.saturation_factor ** (1.0 / self.saturation_iterations)
        for _ in range(self.saturation_iterations):
            s_boosted        = np.clip(hsv[:, :, 1] * boost_per_iter, 0, 255)
            # Appliquer uniquement là où le masque l'autorise
            hsv[:, :, 1]    = hsv[:, :, 1] * (1 - protect_mask) + s_boosted * protect_mask

        # 4. HSV → BGR
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 5. Réduction bruit couleur (bilatéral — préserve les contours)
        if self.noise_reduction:
            strength = max(3, self.noise_strength | 1)   # Garantir impair
            result   = cv2.bilateralFilter(result, d=strength,
                                           sigmaColor=40, sigmaSpace=40)

        # 6. Composite LRGB (luminance originale + chrominance colorisée)
        if self.lrgb_composite:
            result = self._apply_lrgb(frame_bgr, result)

        return result

    def _apply_lrgb(self, original: np.ndarray, colorized: np.ndarray) -> np.ndarray:
        """
        Mélange la luminance de l'original (netteté maximale)
        avec la chrominance du résultat colorisé.
        """
        orig_ycc  = cv2.cvtColor(original,  cv2.COLOR_BGR2YCrCb).astype(np.float32)
        color_ycc = cv2.cvtColor(colorized, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        w         = self.lrgb_color_weight
        comp      = orig_ycc.copy()
        comp[:, :, 1] = orig_ycc[:, :, 1] * (1 - w) + color_ycc[:, :, 1] * w
        comp[:, :, 2] = orig_ycc[:, :, 2] * (1 - w) + color_ycc[:, :, 2] * w

        return cv2.cvtColor(np.clip(comp, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    def boost_per_pass_str(self) -> str:
        """Retourne le boost par passe sous forme lisible. Ex: '×1.26/passe'"""
        if self.saturation_iterations > 0:
            bpp = self.saturation_factor ** (1.0 / self.saturation_iterations)
            return f"x{bpp:.2f}/passe"
        return ""
