#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session de live stacking - API principale
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import StackingConfig
from .quality import QualityAnalyzer
from .aligner import AdvancedAligner
from .stacker import ImageStacker
from .io import load_image, save_fits, save_png_auto
from .stretch import apply_stretch
from .isp import ISP, ISPCalibrator


class LiveStackSession:
    """
    Session de live stacking - API principale
    
    Usage pour RPiCamera.py:
        config = StackingConfig()
        session = LiveStackSession(config)
        session.start()
        
        # Pour chaque frame de la caméra
        result = session.process_image_data(camera_array)
        
        # Sauvegarder
        session.save_result("output.fit")
        session.stop()
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: StackingConfig instance (ou None pour défaut)
        """
        self.config = config if config else StackingConfig()
        self.config.validate()

        # Composants
        self.quality_analyzer = QualityAnalyzer(self.config.quality)
        self.aligner = AdvancedAligner(self.config)
        self.stacker = ImageStacker(self.config)

        # ISP (Image Signal Processor)
        self.isp = None
        print(f"[DEBUG ISP INIT] isp_enable={self.config.isp_enable}, isp_config_path={self.config.isp_config_path}")
        if self.config.isp_enable:
            if self.config.isp_config_path:
                # Charger config ISP existante
                print(f"[DEBUG ISP] Chargement de {self.config.isp_config_path}...")
                try:
                    self.isp = ISP.load_config(self.config.isp_config_path)
                    print(f"[DEBUG ISP] ✓ Chargé avec succès")
                except Exception as e:
                    print(f"[DEBUG ISP] ✗ Erreur: {e}")
            else:
                print(f"[DEBUG ISP] Pas de chemin de config, isp_config_path est vide/None")
            # Sinon, calibration auto plus tard avec première frame
        else:
            print(f"[DEBUG ISP] ISP désactivé (isp_enable=False)")

        # État
        self.is_running = False
        self.files_processed = 0
        self.files_rejected = 0
        self.files_failed = 0
        self.is_color = None
        self.rotation_angles = []

        # Compteur pour rafraîchissement preview
        self._frame_count_since_refresh = 0
    
    def start(self):
        """Démarre la session"""
        print("\n" + "="*60)
        print(">>> LIBASTROSTACK SESSION")
        print("="*60)

        print("\n[CONFIG]")
        print(f"  - Mode alignement: {self.config.alignment_mode}")
        print(f"  - Contrôle qualité: {'OUI' if self.config.quality.enable else 'NON'}")
        print(f"  - ISP activé: {'OUI' if self.config.isp_enable else 'NON'}")
        if self.config.isp_enable:
            if self.isp:
                print(f"    • Config chargée: {self.config.isp_config_path}")
            else:
                print(f"    • Mode: Auto-calibration")
        print(f"  - Format vidéo: {self.config.video_format or 'Auto-détection'}")
        print(f"  - PNG bit depth: {self.config.png_bit_depth or 'Auto'}")
        print(f"  - Étirement PNG: {self.config.png_stretch_method}")
        print(f"  - FITS: {'Linéaire (RAW)' if self.config.fits_linear else 'Stretched'}")
        print(f"  - Preview refresh: toutes les {self.config.preview_refresh_interval} images")
        print(f"  - Save DNG: {self.config.save_dng_mode}")
        
        if self.config.quality.enable:
            print(f"\n[QC] Seuils:")
            print(f"  - FWHM max: {self.config.quality.max_fwhm} px")
            print(f"  - Ellipticité max: {self.config.quality.max_ellipticity}")
            print(f"  - Étoiles min: {self.config.quality.min_stars}")
            print(f"  - Drift max: {self.config.quality.max_drift} px")
            print(f"  - Netteté min: {self.config.quality.min_sharpness}")
        
        self.is_running = True
        print("\n[OK] Session prête !\n")
    
    def process_image_data(self, image_data):
        """
        Traite une image directement depuis un array (pour RPiCamera)
        
        Args:
            image_data: Array NumPy (float32) RGB ou MONO
        
        Returns:
            Image empilée courante (ou None si rejetée)
        """
        if not self.is_running:
            return None
        
        print(f"\n[IMG] Frame {self.files_processed + self.files_rejected + 1}")
        
        try:
            # Détecter type si première image
            if self.is_color is None:
                self.is_color = len(image_data.shape) == 3
                print(f"[DETECT] Mode {'RGB' if self.is_color else 'MONO'}")
            
            # 1. Contrôle qualité
            is_good, metrics, reason = self.quality_analyzer.analyze(image_data)
            
            if is_good:
                print(f"  [QC] OK - FWHM:{metrics.get('median_fwhm', 0):.2f}px, "
                      f"Ell:{metrics.get('median_ellipticity', 0):.2f}, "
                      f"Sharp:{metrics.get('sharpness', 0):.2f}, "
                      f"Stars:{metrics.get('num_stars', 0)}")
            else:
                print(f"  [QC] REJECT - {reason}")
                self.files_rejected += 1
                self.config.quality.rejected_images.append(f"frame_{self.files_processed + self.files_rejected}")
                self.config.quality.rejection_reasons[f"frame_{self.files_processed + self.files_rejected}"] = reason
                return None
            
            # 2. Alignement
            # Si mode OFF, sauter l'alignement
            if self.config.alignment_mode.upper() == "OFF" or self.config.alignment_mode.upper() == "NONE":
                print("  [ALIGN] Mode OFF - Pas d'alignement")
                aligned = image_data
                params = {'dx': 0, 'dy': 0, 'angle': 0, 'scale': 1.0}
                success = True
            else:
                print("  [ALIGN] Alignement...")
                aligned, params, success = self.aligner.align(image_data)

                if not success:
                    print("  [FAIL] Échec alignement")
                    self.files_rejected += 1
                    return None
            
            # Enregistrer rotation
            if 'angle' in params:
                self.rotation_angles.append(params['angle'])
            
            # 3. Empilement
            print("  [STACK] Empilement...")
            result = self.stacker.stack(aligned)
            
            self.files_processed += 1
            self._frame_count_since_refresh += 1
            
            self._print_stats()
            
            # Retourner résultat si rafraîchissement nécessaire
            if self._frame_count_since_refresh >= self.config.preview_refresh_interval:
                self._frame_count_since_refresh = 0
                return result
            
            return None  # Pas de rafraîchissement preview
            
        except Exception as e:
            print(f"[ERROR] {e}")
            self.files_failed += 1
            import traceback
            traceback.print_exc()
            return None
    
    def process_image_file(self, image_path):
        """
        Traite une image depuis un fichier (pour batch processing)
        
        Args:
            image_path: Chemin vers fichier image
        
        Returns:
            Image empilée courante (ou None si rejetée)
        """
        print(f"\n[IMG] {Path(image_path).name}")
        
        image_data = load_image(image_path)
        if image_data is None:
            self.files_failed += 1
            return None
        
        return self.process_image_data(image_data)
    
    def get_current_stack(self):
        """
        Retourne l'image empilée courante
        
        Returns:
            Image empilée (copie) ou None
        """
        return self.stacker.get_result()
    
    def get_preview_png(self):
        """
        Génère preview PNG étiré pour affichage

        Returns:
            Array uint8 ou uint16 (selon config) pour affichage/sauvegarde
        """
        print(f"\n[DEBUG get_preview_png] Début")
        print(f"  • ISP activé: {self.config.isp_enable}")
        print(f"  • ISP instance: {self.isp is not None}")
        print(f"  • Format vidéo: {self.config.video_format}")
        print(f"  • PNG bit depth config: {self.config.png_bit_depth}")

        result = self.stacker.get_result()
        if result is None:
            return None

        print(f"  • Stack result shape: {result.shape}, dtype: {result.dtype}")
        print(f"  • Stack result range: [{result.min():.3f}, {result.max():.3f}]")

        # NOUVEAU: Appliquer ISP AVANT stretch (si activé ET format RAW uniquement)
        # Pipeline optimal: Stack (linéaire) → ISP → Stretch → PNG
        # L'ISP ne doit s'appliquer QUE sur RAW12/RAW16, JAMAIS sur YUV420 (déjà traité par ISP hardware)
        is_raw_format = self.config.video_format in ['raw12', 'raw16']
        if self.config.isp_enable and self.isp is not None and is_raw_format:
            print(f"  → Application ISP (format RAW détecté)...")
            # swap_rb=True pour RAW car le débayeurisation inverse R/B (RPiCamera2.py:4931-4932)
            result = self.isp.process(result, return_uint8=False, swap_rb=True)  # Reste en float32
            print(f"  → Après ISP: shape={result.shape}, dtype={result.dtype}, range=[{result.min():.3f}, {result.max():.3f}]")
        elif self.config.isp_enable and not is_raw_format:
            print(f"  → ISP ignoré (format {self.config.video_format} déjà traité par camera ISP)")
        else:
            print(f"  → Pas d'ISP (enable={self.config.isp_enable}, isp={self.isp is not None})")

        # Appliquer étirement
        is_color = len(result.shape) == 3

        # CORRECTION: Normaliser [0-255] → [0-1] AVANT stretch pour éviter clip
        # La fonction stretch_ghs() attend des données en [0-1], pas [0-255]
        # Pour YUV420: normalisation simple /255 (comme RPiCamera2.py)
        # Pour RAW: normalisation par percentiles (pour images brutes)
        if is_raw_format:
            # RAW: normalisation par percentiles (data brute avec possibles pixels chauds)
            # Les données RAW passent par ISP qui normalise déjà à [0-1]
            clip_low = self.config.png_clip_low
            clip_high = self.config.png_clip_high
            print(f"  • Normalisation: Percentiles (RAW) - clip_low={clip_low}%, clip_high={clip_high}%")
        else:
            # YUV420: normalisation simple /255 (déjà traité par ISP caméra, pas de pixels chauds)
            # On normalise manuellement AVANT d'appeler stretch pour éviter le clip à [0-1]
            result = result / 255.0  # [0-255] → [0-1]
            result = np.clip(result, 0, 1)
            # Désactiver la normalisation par percentiles dans stretch (déjà normalisé)
            clip_low = 0.0
            clip_high = 100.0
            print(f"  • Normalisation: Simple /255 (YUV420 déjà traité par ISP caméra)")

        if is_color:
            stretched = np.zeros_like(result, dtype=np.float32)
            for i in range(3):
                stretched[:, :, i] = apply_stretch(
                    result[:, :, i],
                    method=self.config.png_stretch_method,
                    factor=self.config.png_stretch_factor,
                    clip_low=clip_low,
                    clip_high=clip_high,
                    ghs_D=getattr(self.config, 'ghs_D', 3.0),
                    ghs_b=getattr(self.config, 'ghs_b', getattr(self.config, 'ghs_B', 0.13)),
                    ghs_SP=getattr(self.config, 'ghs_SP', 0.2),
                    ghs_LP=getattr(self.config, 'ghs_LP', 0.0),
                    ghs_HP=getattr(self.config, 'ghs_HP', 0.0)
                )
        else:
            stretched = apply_stretch(
                result,
                method=self.config.png_stretch_method,
                factor=self.config.png_stretch_factor,
                clip_low=clip_low,
                clip_high=clip_high,
                ghs_D=getattr(self.config, 'ghs_D', 3.0),
                ghs_b=getattr(self.config, 'ghs_b', getattr(self.config, 'ghs_B', 0.13)),
                ghs_SP=getattr(self.config, 'ghs_SP', 0.2),
                ghs_LP=getattr(self.config, 'ghs_LP', 0.0),
                ghs_HP=getattr(self.config, 'ghs_HP', 0.0)
            )

        # Convertir en uint8 ou uint16 selon configuration
        # NOUVEAU: Support PNG 16-bit
        print(f"  • Stretched range: [{stretched.min():.3f}, {stretched.max():.3f}]")
        print(f"  → Sélection bit depth:")
        print(f"     config.png_bit_depth = {self.config.png_bit_depth}")
        print(f"     config.video_format = {self.config.video_format}")

        if self.config.png_bit_depth == 16:
            preview = (stretched * 65535).astype(np.uint16)
            print(f"     → 16-bit (forcé par config)")
        elif self.config.png_bit_depth == 8:
            preview = (stretched * 255).astype(np.uint8)
            print(f"     → 8-bit (forcé par config)")
        else:
            # Auto-détection intelligente selon stretch et format
            # CORRECTION: Toujours utiliser 16-bit si stretch activé (même pour YUV420)
            # Car le stretch crée des niveaux intermédiaires → besoin de 16-bit pour histogramme lisse
            if self.config.png_stretch_method != 'off':
                preview = (stretched * 65535).astype(np.uint16)
                print(f"     → 16-bit (auto: stretch '{self.config.png_stretch_method}' activé → préserve histogramme)")
            elif self.config.video_format and 'raw' in self.config.video_format.lower():
                preview = (stretched * 65535).astype(np.uint16)
                print(f"     → 16-bit (auto: format RAW)")
            else:
                preview = (stretched * 255).astype(np.uint8)
                print(f"     → 8-bit (auto: pas de stretch, pas de RAW)")

        print(f"  ✓ Preview final: dtype={preview.dtype}, shape={preview.shape}")
        print(f"     range=[{preview.min()}, {preview.max()}]")

        return preview
    
    def save_result(self, output_path, generate_png=None):
        """
        Sauvegarde résultat final
        
        Args:
            output_path: Chemin de sortie (.fit)
            generate_png: Générer PNG (défaut = config.auto_png)
        """
        result = self.stacker.get_result()
        if result is None:
            print("[WARN] Aucune image empilée")
            return
        
        output_path = Path(output_path)
        
        # Métadonnées FITS
        header_data = {
            'STACKED': self.config.num_stacked,
            'REJECTED': self.files_rejected,
            'ALIGNMOD': self.config.alignment_mode,
        }
        
        if self.rotation_angles:
            header_data['ROTMIN'] = np.min(self.rotation_angles)
            header_data['ROTMAX'] = np.max(self.rotation_angles)
            header_data['ROTMED'] = np.median(self.rotation_angles)
        
        # Sauvegarder FITS (linéaire ou stretched selon config)
        save_fits(result, output_path, header_data, linear=self.config.fits_linear)

        print(f"\n[SAVE] FITS: {output_path}")
        print(f"       Type: {'Linéaire (RAW)' if self.config.fits_linear else 'Stretched'}")
        print(f"       Acceptées: {self.config.num_stacked}, Rejetées: {self.files_rejected}")
        
        if self.rotation_angles:
            print(f"       Rotation: {np.min(self.rotation_angles):.2f}° à {np.max(self.rotation_angles):.2f}°")
        
        # PNG si demandé
        if generate_png is None:
            generate_png = self.config.auto_png
        
        if generate_png:
            png_path = output_path.with_suffix('.png')
            self._save_png(result, png_path)
        
        # Rapport qualité si rejets
        if self.files_rejected > 0 and self.config.save_rejected_list:
            report_path = output_path.with_suffix('.quality_report.txt')
            self._save_quality_report(report_path)
    
    def stop(self):
        """Arrête la session"""
        self.is_running = False
        print("\n" + "="*60)
        print("[STOP] SESSION TERMINÉE")
        print("="*60)
        self._print_final_stats()
    
    def reset(self):
        """Réinitialise la session (garde config)"""
        self.stacker.reset()
        self.aligner.reference_image = None
        self.aligner.reference_stars = None
        self.files_processed = 0
        self.files_rejected = 0
        self.files_failed = 0
        self.rotation_angles = []
        self._frame_count_since_refresh = 0
        self.config.quality.rejected_images = []
        self.config.quality.rejection_reasons = {}
    
    def _save_png(self, data, png_path):
        """Sauvegarde PNG avec détection automatique du bit depth"""
        preview = self.get_preview_png()
        if preview is None:
            return

        import cv2

        # Détecter le bit depth
        bit_depth = 16 if preview.dtype == np.uint16 else 8

        if len(preview.shape) == 3:
            # CORRECTION: Les données sont en RGB (de Picamera2)
            # Les fichiers PNG stockent RGB, donc écrire directement
            # Note: cv2.imwrite() n'effectue PAS de conversion BGR->RGB pour les PNG!
            # Il écrit les données telles quelles. Donc RGB→RGB = correct!
            cv2.imwrite(str(png_path), preview)
        else:
            # Grayscale image
            cv2.imwrite(str(png_path), preview)

        # Afficher infos
        file_size = Path(png_path).stat().st_size / 1024
        print(f"[OK] PNG: {png_path}")
        print(f"       Bit depth: {bit_depth}-bit, Taille: {file_size:.1f} KB")
    
    def _save_quality_report(self, report_path):
        """Sauvegarde rapport qualité"""
        with open(report_path, 'w') as f:
            f.write("=== RAPPORT DE CONTRÔLE QUALITÉ ===\n\n")
            f.write(f"Images acceptées: {self.config.num_stacked}\n")
            f.write(f"Images rejetées: {self.files_rejected}\n")
            
            total = self.config.num_stacked + self.files_rejected
            if total > 0:
                f.write(f"Taux d'acceptation: {100*self.config.num_stacked/total:.1f}%\n\n")
            
            f.write("Images rejetées:\n")
            for img_name in self.config.quality.rejected_images:
                reason = self.config.quality.rejection_reasons.get(img_name, "?")
                f.write(f"  - {img_name}: {reason}\n")
        
        print(f"[SAVE] Rapport: {report_path}")
    
    def _print_stats(self):
        """Affiche stats courantes"""
        print(f"  [STATS] Empilées: {self.config.num_stacked}, "
              f"Rejetées: {self.files_rejected}, Échecs: {self.files_failed}")
    
    def _print_final_stats(self):
        """Affiche stats finales"""
        print(f"\n[STATS] Final:")
        print(f"  * Type: {'RGB' if self.is_color else 'MONO'}")
        print(f"  * Mode: {self.config.alignment_mode}")
        print(f"  * Empilées: {self.config.num_stacked}")
        print(f"  * Rejetées: {self.files_rejected}")
        print(f"  * Échecs: {self.files_failed}")
        
        total = self.config.num_stacked + self.files_rejected
        if total > 0:
            print(f"  * Taux acceptation: {100*self.config.num_stacked/total:.1f}%")
        
        if self.rotation_angles:
            print(f"  * Rotation: {np.min(self.rotation_angles):.2f}° à {np.max(self.rotation_angles):.2f}°")
        
        if self.config.num_stacked > 0:
            snr_gain = self.stacker.get_snr_improvement()
            print(f"  * SNR gain: x{snr_gain:.2f}")
