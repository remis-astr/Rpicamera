#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthodes d'étirement d'histogramme pour PNG
"""

import numpy as np
import cv2
from .config import StretchMethod


def stretch_linear(data, clip_low=1.0, clip_high=99.5):
    """
    Étirement linéaire simple
    
    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    stretched = (data - vmin) / (vmax - vmin)
    return np.clip(stretched, 0, 1)


def stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5):
    """
    Étirement arc-sinus hyperbolique (recommandé pour astro)
    
    Args:
        data: Image (float array)
        factor: Facteur d'étirement (5-50)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    # Normaliser d'abord
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Appliquer asinh
    stretched = np.arcsinh(normalized * factor) / np.arcsinh(factor)
    return stretched


def stretch_log(data, factor=100.0, clip_low=1.0, clip_high=99.5):
    """
    Étirement logarithmique
    
    Args:
        data: Image (float array)
        factor: Facteur d'étirement (10-200)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Log stretch
    stretched = np.log1p(normalized * factor) / np.log1p(factor)
    return stretched


def stretch_sqrt(data, clip_low=1.0, clip_high=99.5):
    """
    Étirement racine carrée (bon pour objets brillants)
    
    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    return np.sqrt(normalized)


def stretch_histogram(data, clip_low=1.0, clip_high=99.5):
    """
    Égalisation d'histogramme
    
    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)
    
    Returns:
        Image étirée (0-1)
    """
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    
    if vmax == vmin:
        return np.zeros_like(data)
    
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    # Convertir en uint8 pour cv2.equalizeHist
    img_8 = (normalized * 255).astype(np.uint8)
    equalized = cv2.equalizeHist(img_8)
    
    return equalized / 255.0


def stretch_auto(data, clip_low=0.1, clip_high=99.9):
    """
    Auto-stretch adaptatif (type SIRIL)

    Args:
        data: Image (float array)
        clip_low: Percentile bas (%)
        clip_high: Percentile haut (%)

    Returns:
        Image étirée (0-1)
    """
    # Estimer fond du ciel
    background = np.percentile(data, 5)
    data_clean = np.maximum(data - background, 0)

    # Percentiles adaptatifs
    vmin = np.percentile(data_clean, clip_low)
    vmax = np.percentile(data_clean, clip_high)

    if vmax == vmin:
        return np.zeros_like(data)

    normalized = np.clip((data_clean - vmin) / (vmax - vmin), 0, 1)

    # Appliquer asinh doux
    factor = 5.0
    stretched = np.arcsinh(normalized * factor) / np.arcsinh(factor)

    return stretched


def stretch_ghs(data, D=3.0, b=0.13, SP=0.2, LP=0.0, HP=0.0, clip_low=0.0, clip_high=99.97, B=None):
    """
    Generalized Hyperbolic Stretch (GHS) - Implémentation complète
    Conforme à RPiCamera2 et ghsastro.co.uk

    Args:
        data: Image (float array, 0-1)
        D: Stretch factor (0.0 à 10.0) - force de l'étirement (défaut: 3.0)
        b: Local intensity (-5.0 à 20.0) - concentration du contraste (défaut: 0.13)
        SP: Symmetry point (0.0 à 1.0) - point focal du contraste (défaut: 0.2)
        LP: Protect shadows (0.0 à SP) - protection basses lumières (défaut: 0.0)
        HP: Protect highlights (SP à 1.0) - protection hautes lumières (défaut: 0.0)
        clip_low: Percentile bas (%) - normalisation pré-traitement (défaut: 0.0)
        clip_high: Percentile haut (%) - normalisation pré-traitement (défaut: 99.97)
        B: [DEPRECATED] Ancien paramètre, utilisez 'b' à la place

    Returns:
        Image étirée (0-1 float)

    Notes:
        - Normalisation par percentiles appliquée AVANT la transformation GHS
        - Valeurs par défaut optimisées pour préserver la dynamique complète
        - Pour désactiver normalisation: clip_low=0, clip_high=100
    """
    # Rétro-compatibilité: si B est fourni et b n'est pas modifié
    if B is not None and b == 0.13:
        b = B

    # Normalisation par percentiles AVANT GHS (crucial pour images brutes)
    if clip_low > 0 or clip_high < 100:
        vmin = np.percentile(data, clip_low)
        vmax = np.percentile(data, clip_high)

        if vmax > vmin:
            data = np.clip((data - vmin) / (vmax - vmin), 0, 1)

    # Transformation GHS complète (version RPiCamera2)
    epsilon = 1e-10
    img_float = np.clip(data.astype(np.float64), epsilon, 1.0 - epsilon)

    if abs(D) < epsilon:
        return img_float.astype(np.float32)

    # Contraintes
    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))

    def T_base(x, D, b):
        """Transformation de base selon b"""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            result = np.log1p(D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            exponent = (b + 1.0) / b
            result = (1.0 - np.power(base, exponent)) / (D * (b + 1.0))
        elif abs(b) < epsilon:
            result = 1.0 - np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = 1.0 - 1.0 / (1.0 + D * x)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = 1.0 - np.power(base, -1.0 / b)

        return result

    def T_prime(x, D, b):
        """Dérivée première"""
        x = np.asarray(x, dtype=np.float64)

        if abs(b - (-1.0)) < epsilon:
            return D / (1.0 + D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            return np.power(base, 1.0 / b)
        elif abs(b) < epsilon:
            return D * np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            return D / np.power(1.0 + D * x, 2)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            return D * np.power(base, -(1.0 / b + 1.0))

    T_1 = T_base(1.0, D, b)
    img_stretched = np.zeros_like(img_float)

    # Zone 1: x < LP (linéaire - shadows)
    if LP > epsilon:
        mask_low = img_float < LP
        if np.any(mask_low):
            slope_SP = T_prime(SP, D, b) / T_1
            slope_LP = slope_SP * (LP / SP)
            img_stretched[mask_low] = slope_LP * img_float[mask_low]

    # Zone 2: LP <= x < SP (transformation hyperbolic)
    mask_mid_low = (img_float >= LP) & (img_float < SP)
    if np.any(mask_mid_low):
        x_norm = img_float[mask_mid_low] / SP
        T_x = T_base(x_norm, D, b) / T_1
        img_stretched[mask_mid_low] = SP * T_x

    # Zone 3: x >= SP (symétrie miroir)
    mask_high = img_float >= SP
    if np.any(mask_high):
        if HP < 1.0 - epsilon:
            mask_mid_high = (img_float >= SP) & (img_float < HP)
            if np.any(mask_mid_high):
                x_mirror = 1.0 - img_float[mask_mid_high]
                x_norm_mirror = x_mirror / (1.0 - SP)
                T_mirror = T_base(x_norm_mirror, D, b) / T_1
                img_stretched[mask_mid_high] = 1.0 - (1.0 - SP) * T_mirror

            mask_very_high = img_float >= HP
            if np.any(mask_very_high):
                slope_SP_high = T_prime(SP, D, b) / T_1
                slope_HP = slope_SP_high * ((1.0 - HP) / (1.0 - SP))
                T_HP_val = T_base((1.0 - HP) / (1.0 - SP), D, b) / T_1
                y_HP = 1.0 - (1.0 - SP) * T_HP_val
                img_stretched[mask_very_high] = y_HP + slope_HP * (img_float[mask_very_high] - HP)
        else:
            x_mirror = 1.0 - img_float[mask_high]
            x_norm_mirror = x_mirror / (1.0 - SP)
            T_mirror = T_base(x_norm_mirror, D, b) / T_1
            img_stretched[mask_high] = 1.0 - (1.0 - SP) * T_mirror

    return np.clip(img_stretched, 0.0, 1.0).astype(np.float32)


def apply_stretch(data, method=StretchMethod.ASINH, **params):
    """
    Applique la méthode d'étirement spécifiée

    Args:
        data: Image à étirer (float array)
        method: Méthode ('off', 'linear', 'asinh', 'log', 'sqrt', 'histogram', 'auto', 'ghs')
        **params: Paramètres (factor, clip_low, clip_high, ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP)

    Returns:
        Image étirée (0-1)
    """
    if method == StretchMethod.OFF:
        # Pas de stretch - juste clip [0, 1] SANS normalisation
        # IMPORTANT: Ne PAS normaliser pour préserver l'histogramme original
        return np.clip(data, 0, 1)

    elif method == StretchMethod.LINEAR:
        return stretch_linear(
            data,
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.ASINH:
        return stretch_asinh(
            data,
            factor=params.get('factor', 10.0),
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.LOG:
        return stretch_log(
            data,
            factor=params.get('factor', 100.0),
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.SQRT:
        return stretch_sqrt(
            data,
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.HISTOGRAM:
        return stretch_histogram(
            data,
            clip_low=params.get('clip_low', 1.0),
            clip_high=params.get('clip_high', 99.5)
        )

    elif method == StretchMethod.GHS:
        return stretch_ghs(
            data,
            D=params.get('ghs_D', 3.0),
            b=params.get('ghs_b', params.get('ghs_B', 0.13)),  # Rétro-compatibilité avec ghs_B
            SP=params.get('ghs_SP', 0.2),
            LP=params.get('ghs_LP', 0.0),
            HP=params.get('ghs_HP', 0.0),
            clip_low=params.get('clip_low', 0.0),
            clip_high=params.get('clip_high', 99.97)
        )

    else:  # AUTO
        return stretch_auto(
            data,
            clip_low=params.get('clip_low', 0.1),
            clip_high=params.get('clip_high', 99.9)
        )
