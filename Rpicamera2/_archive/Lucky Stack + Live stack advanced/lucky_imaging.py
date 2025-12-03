#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lucky Imaging pour libastrostack
================================

Module de stacking planétaire haute vitesse basé sur le principe du Lucky Imaging :
1. Acquisition continue dans un buffer circulaire (ring buffer)
2. Notation rapide de chaque image (score de qualité)
3. Sélection des x% meilleures images
4. Alignement et empilement des images sélectionnées
5. Affichage du résultat

Optimisé pour :
- Haute cadence (>100 fps)
- Faible latence (<10ms par image pour le scoring)
- Imagerie planétaire (Soleil, Lune, planètes)

Paramètres réglables :
- buffer_size : Taille du buffer circulaire (ex: 100, 200, 500)
- keep_percent : Pourcentage d'images à garder (ex: 10%, 20%, 50%)
- score_method : Méthode de scoring (laplacian, gradient, sobel, tenengrad)
- stack_interval : Intervalle de stacking (toutes les N frames)
- min_score : Score minimum absolu pour accepter une image

Usage :
    from libastrostack.lucky_imaging import LuckyImagingStacker, LuckyConfig
    
    config = LuckyConfig(buffer_size=100, keep_percent=10.0)
    stacker = LuckyImagingStacker(config)
    
    for frame in camera_stream:
        stacker.add_frame(frame)
        
        if stacker.is_buffer_full():
            result = stacker.process_buffer()
            display(result)

Auteur: libastrostack Team
Version: 1.0.0
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor


class ScoreMethod(Enum):
    """Méthodes de calcul du score de qualité"""
    LAPLACIAN = "laplacian"       # Variance du Laplacien (rapide, recommandé)
    GRADIENT = "gradient"          # Magnitude du gradient
    SOBEL = "sobel"               # Filtre de Sobel
    TENENGRAD = "tenengrad"       # Tenengrad (Sobel au carré)
    BRENNER = "brenner"           # Gradient de Brenner
    FFT = "fft"                   # Analyse fréquentielle (plus lent)


class StackMethod(Enum):
    """Méthodes de combinaison des images"""
    MEAN = "mean"                 # Moyenne simple
    MEDIAN = "median"             # Médiane (plus robuste)
    SIGMA_CLIP = "sigma_clip"     # Moyenne avec rejection sigma


@dataclass
class LuckyConfig:
    """Configuration du Lucky Imaging"""
    
    # Buffer circulaire
    buffer_size: int = 100              # Nombre d'images dans le buffer
    
    # Sélection
    keep_percent: float = 10.0          # Pourcentage d'images à garder (1-100)
    keep_count: Optional[int] = None    # OU nombre fixe d'images (prioritaire sur %)
    min_score: float = 0.0              # Score minimum absolu (0 = désactivé)
    
    # Scoring
    score_method: ScoreMethod = ScoreMethod.LAPLACIAN
    score_roi: Optional[Tuple[int, int, int, int]] = None  # ROI pour scoring (x, y, w, h)
    score_roi_percent: float = 50.0     # OU % central de l'image pour scoring
    use_gpu: bool = False               # Utiliser GPU si disponible (OpenCV CUDA)
    
    # Stacking
    stack_method: StackMethod = StackMethod.MEAN
    stack_interval: int = 0             # 0 = stack quand buffer plein, N = toutes les N frames
    auto_stack: bool = True             # Stacker automatiquement quand buffer plein
    sigma_clip_kappa: float = 2.5       # Kappa pour sigma clipping
    
    # Alignement
    align_enabled: bool = True          # Activer l'alignement
    align_method: str = "phase"         # "phase" (FFT) ou "ecc" (Enhanced Correlation)
    align_roi_percent: float = 80.0     # % central pour alignement
    
    # Performance
    num_threads: int = 4                # Threads pour scoring parallèle
    downscale_scoring: float = 1.0      # Downscale pour scoring (1.0 = pas de downscale)
    
    # Statistiques
    keep_history: bool = True           # Garder historique des scores
    history_size: int = 1000            # Taille max de l'historique
    
    def validate(self) -> bool:
        """Valide la configuration"""
        errors = []
        
        if self.buffer_size < 10:
            errors.append("buffer_size doit être >= 10")
        
        if not 1.0 <= self.keep_percent <= 100.0:
            errors.append("keep_percent doit être entre 1 et 100")
        
        if self.keep_count is not None and self.keep_count < 1:
            errors.append("keep_count doit être >= 1")
        
        if not 0.0 <= self.score_roi_percent <= 100.0:
            errors.append("score_roi_percent doit être entre 0 et 100")
        
        if self.downscale_scoring <= 0 or self.downscale_scoring > 1.0:
            errors.append("downscale_scoring doit être entre 0 (exclu) et 1.0")
        
        if errors:
            raise ValueError("Config Lucky Imaging invalide: " + ", ".join(errors))
        
        return True
    
    def get_keep_count(self) -> int:
        """Retourne le nombre d'images à garder"""
        if self.keep_count is not None:
            return min(self.keep_count, self.buffer_size)
        return max(1, int(self.buffer_size * self.keep_percent / 100.0))


class FrameBuffer:
    """
    Buffer circulaire optimisé pour les images
    
    Stocke les images et leurs scores de manière efficace en mémoire.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.frames: deque = deque(maxlen=max_size)
        self.scores: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
        self.frame_count = 0
        self._lock = threading.Lock()
    
    def add(self, frame: np.ndarray, score: float = 0.0) -> int:
        """
        Ajoute une frame au buffer
        
        Returns:
            Index de la frame dans le buffer
        """
        with self._lock:
            self.frames.append(frame.copy())
            self.scores.append(score)
            self.timestamps.append(time.time())
            self.frame_count += 1
            return len(self.frames) - 1
    
    def update_score(self, index: int, score: float):
        """Met à jour le score d'une frame"""
        with self._lock:
            if 0 <= index < len(self.scores):
                self.scores[index] = score
    
    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Récupère une frame par index"""
        with self._lock:
            if 0 <= index < len(self.frames):
                return self.frames[index]
            return None
    
    def get_frames_by_indices(self, indices: List[int]) -> List[np.ndarray]:
        """Récupère plusieurs frames par indices"""
        with self._lock:
            return [self.frames[i] for i in indices if 0 <= i < len(self.frames)]
    
    def get_best_indices(self, count: int, min_score: float = 0.0) -> List[int]:
        """
        Retourne les indices des N meilleures frames
        
        Args:
            count: Nombre de frames à retourner
            min_score: Score minimum requis
        
        Returns:
            Liste des indices triés par score décroissant
        """
        with self._lock:
            # Créer liste (index, score) filtrée par min_score
            scored = [(i, s) for i, s in enumerate(self.scores) if s >= min_score]
            
            # Trier par score décroissant
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner les N premiers indices
            return [idx for idx, _ in scored[:count]]
    
    def get_statistics(self) -> Dict[str, float]:
        """Retourne statistiques sur les scores"""
        with self._lock:
            if not self.scores:
                return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'count': 0}
            
            scores_array = np.array(self.scores)
            return {
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'count': len(scores_array)
            }
    
    def is_full(self) -> bool:
        """Vérifie si le buffer est plein"""
        return len(self.frames) >= self.max_size
    
    def clear(self):
        """Vide le buffer"""
        with self._lock:
            self.frames.clear()
            self.scores.clear()
            self.timestamps.clear()
    
    def __len__(self) -> int:
        return len(self.frames)


class QualityScorer:
    """
    Calculateur de score de qualité ultra-rapide
    
    Optimisé pour haute cadence (>100 fps).
    Temps cible : <5ms par image.
    """
    
    def __init__(self, config: LuckyConfig):
        self.config = config
        self._roi_cache = None
        
        # Sélectionner la méthode de scoring
        self._score_func = self._get_score_function(config.score_method)
    
    def _get_score_function(self, method: ScoreMethod) -> Callable:
        """Retourne la fonction de scoring appropriée"""
        methods = {
            ScoreMethod.LAPLACIAN: self._score_laplacian,
            ScoreMethod.GRADIENT: self._score_gradient,
            ScoreMethod.SOBEL: self._score_sobel,
            ScoreMethod.TENENGRAD: self._score_tenengrad,
            ScoreMethod.BRENNER: self._score_brenner,
            ScoreMethod.FFT: self._score_fft,
        }
        return methods.get(method, self._score_laplacian)
    
    def score(self, image: np.ndarray) -> float:
        """
        Calcule le score de qualité d'une image
        
        Args:
            image: Image (grayscale ou RGB)
        
        Returns:
            Score de qualité (plus élevé = meilleure qualité)
        """
        # Convertir en grayscale si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8) if image.dtype != np.uint8 
                               else image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8) if image.dtype != np.uint8 else image
        
        # Downscale si configuré
        if self.config.downscale_scoring < 1.0:
            new_size = (int(gray.shape[1] * self.config.downscale_scoring),
                       int(gray.shape[0] * self.config.downscale_scoring))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        
        # Extraire ROI si configuré
        gray = self._extract_roi(gray)
        
        # Calculer score
        return self._score_func(gray)
    
    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """Extrait la région d'intérêt pour le scoring"""
        if self.config.score_roi is not None:
            x, y, w, h = self.config.score_roi
            h_img, w_img = image.shape[:2]
            x = min(x, w_img - 1)
            y = min(y, h_img - 1)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            return image[y:y+h, x:x+w]
        
        elif self.config.score_roi_percent < 100.0:
            h, w = image.shape[:2]
            margin_x = int(w * (100 - self.config.score_roi_percent) / 200)
            margin_y = int(h * (100 - self.config.score_roi_percent) / 200)
            return image[margin_y:h-margin_y, margin_x:w-margin_x]
        
        return image
    
    def _score_laplacian(self, gray: np.ndarray) -> float:
        """Score par variance du Laplacien (RECOMMANDÉ - très rapide)"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _score_gradient(self, gray: np.ndarray) -> float:
        """Score par magnitude du gradient"""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        return float(np.mean(magnitude))
    
    def _score_sobel(self, gray: np.ndarray) -> float:
        """Score par filtre de Sobel"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.abs(sobelx) + np.abs(sobely)))
    
    def _score_tenengrad(self, gray: np.ndarray) -> float:
        """Score Tenengrad (Sobel au carré) - très sensible au focus"""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx**2 + gy**2))
    
    def _score_brenner(self, gray: np.ndarray) -> float:
        """Score de Brenner (différence horizontale)"""
        diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
        return float(np.mean(diff**2))
    
    def _score_fft(self, gray: np.ndarray) -> float:
        """Score par analyse FFT (plus lent mais robuste)"""
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        # Score = énergie dans les hautes fréquences
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        # Masquer les basses fréquences (centre)
        mask_radius = min(h, w) // 8
        magnitude[cy-mask_radius:cy+mask_radius, cx-mask_radius:cx+mask_radius] = 0
        return float(np.mean(magnitude))


class FrameAligner:
    """
    Aligneur d'images rapide pour Lucky Imaging
    
    Utilise la corrélation de phase (FFT) pour un alignement sub-pixel rapide.
    """
    
    def __init__(self, config: LuckyConfig):
        self.config = config
        self.reference: Optional[np.ndarray] = None
    
    def set_reference(self, image: np.ndarray):
        """Définit l'image de référence"""
        if len(image.shape) == 3:
            self.reference = cv2.cvtColor(
                image.astype(np.uint8) if image.dtype != np.uint8 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            self.reference = image.astype(np.uint8) if image.dtype != np.uint8 else image
        
        # Extraire ROI pour alignement
        self.reference = self._extract_align_roi(self.reference)
    
    def _extract_align_roi(self, image: np.ndarray) -> np.ndarray:
        """Extrait ROI central pour alignement"""
        if self.config.align_roi_percent < 100.0:
            h, w = image.shape[:2]
            margin_x = int(w * (100 - self.config.align_roi_percent) / 200)
            margin_y = int(h * (100 - self.config.align_roi_percent) / 200)
            return image[margin_y:h-margin_y, margin_x:w-margin_x]
        return image
    
    def align(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aligne une image sur la référence
        
        Returns:
            (image_alignée, params)
        """
        if self.reference is None:
            self.set_reference(image)
            return image.copy(), {'dx': 0.0, 'dy': 0.0}
        
        # Convertir en grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image.astype(np.uint8) if image.dtype != np.uint8 else image,
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = image.astype(np.uint8) if image.dtype != np.uint8 else image
        
        # Extraire ROI
        gray_roi = self._extract_align_roi(gray)
        
        # Calculer décalage par corrélation de phase
        if self.config.align_method == "phase":
            dx, dy = self._phase_correlation(self.reference, gray_roi)
        else:  # ECC
            dx, dy = self._ecc_alignment(self.reference, gray_roi)
        
        # Appliquer translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        if len(image.shape) == 3:
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        else:
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        return aligned, {'dx': dx, 'dy': dy}
    
    def _phase_correlation(self, ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
        """Corrélation de phase pour trouver le décalage"""
        # Assurer mêmes dimensions
        h = min(ref.shape[0], img.shape[0])
        w = min(ref.shape[1], img.shape[1])
        ref = ref[:h, :w]
        img = img[:h, :w]
        
        # FFT
        ref_fft = np.fft.fft2(ref.astype(np.float64))
        img_fft = np.fft.fft2(img.astype(np.float64))
        
        # Cross-power spectrum
        cross_power = (ref_fft * np.conj(img_fft)) / (np.abs(ref_fft * np.conj(img_fft)) + 1e-10)
        
        # Inverse FFT
        correlation = np.real(np.fft.ifft2(cross_power))
        correlation = np.fft.fftshift(correlation)
        
        # Trouver le pic
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        dy = peak_idx[0] - h // 2
        dx = peak_idx[1] - w // 2
        
        return float(dx), float(dy)
    
    def _ecc_alignment(self, ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
        """Alignement par Enhanced Correlation Coefficient"""
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            
            _, warp_matrix = cv2.findTransformECC(
                ref.astype(np.float32),
                img.astype(np.float32),
                warp_matrix,
                cv2.MOTION_TRANSLATION,
                criteria
            )
            
            return float(warp_matrix[0, 2]), float(warp_matrix[1, 2])
        except cv2.error:
            return 0.0, 0.0
    
    def reset(self):
        """Réinitialise la référence"""
        self.reference = None


class LuckyImagingStacker:
    """
    Stacker Lucky Imaging principal
    
    Workflow:
    1. add_frame() : Ajoute frame au buffer + calcule score
    2. is_buffer_full() : Vérifie si buffer prêt
    3. process_buffer() : Sélectionne meilleures + aligne + stack
    4. get_result() : Récupère l'image finale
    """
    
    def __init__(self, config: Optional[LuckyConfig] = None):
        self.config = config if config else LuckyConfig()
        self.config.validate()
        
        # Composants
        self.buffer = FrameBuffer(self.config.buffer_size)
        self.scorer = QualityScorer(self.config)
        self.aligner = FrameAligner(self.config)
        
        # Thread pool pour scoring parallèle
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        # État
        self.is_running = False
        self.last_result: Optional[np.ndarray] = None
        self.last_stack_time = 0.0
        self.total_frames_processed = 0
        self.total_stacks_done = 0
        
        # Historique des scores
        self.score_history: deque = deque(maxlen=self.config.history_size)
        
        # Stats
        self.stats = {
            'frames_total': 0,
            'frames_selected': 0,
            'stacks_done': 0,
            'avg_score': 0.0,
            'best_score': 0.0,
            'worst_score': 0.0,
            'avg_scoring_time_ms': 0.0,
            'avg_stack_time_ms': 0.0,
            'selection_threshold': 0.0,
        }
        
        # Timing
        self._scoring_times: deque = deque(maxlen=100)
        self._stack_times: deque = deque(maxlen=100)
    
    def start(self):
        """Démarre le stacker"""
        self.is_running = True
        self.buffer.clear()
        self.aligner.reset()
        self.score_history.clear()
        self.total_frames_processed = 0
        self.total_stacks_done = 0
        print(f"[LUCKY] Démarré - Buffer: {self.config.buffer_size}, "
              f"Keep: {self.config.keep_percent}%")
    
    def stop(self):
        """Arrête le stacker"""
        self.is_running = False
        self.executor.shutdown(wait=False)
        print(f"[LUCKY] Arrêté - Frames: {self.total_frames_processed}, "
              f"Stacks: {self.total_stacks_done}")
    
    def add_frame(self, frame: np.ndarray) -> float:
        """
        Ajoute une frame au buffer et calcule son score
        
        Args:
            frame: Image (RGB ou grayscale, tout dtype)
        
        Returns:
            Score de qualité de la frame
        """
        if not self.is_running:
            self.start()
        
        # Normaliser l'image
        if frame.dtype != np.float32:
            if frame.dtype == np.uint8:
                frame_norm = frame.astype(np.float32)
            elif frame.dtype == np.uint16:
                frame_norm = (frame.astype(np.float32) / 256.0)
            else:
                frame_norm = frame.astype(np.float32)
        else:
            frame_norm = frame
        
        # Calculer score (chronométré)
        t0 = time.perf_counter()
        score = self.scorer.score(frame_norm)
        scoring_time = (time.perf_counter() - t0) * 1000
        self._scoring_times.append(scoring_time)
        
        # Ajouter au buffer
        self.buffer.add(frame_norm, score)
        
        # Historique
        if self.config.keep_history:
            self.score_history.append(score)
        
        self.total_frames_processed += 1
        self.stats['frames_total'] = self.total_frames_processed
        
        # Auto-stack si configuré
        if self.config.auto_stack and self.buffer.is_full():
            if self.config.stack_interval == 0 or \
               self.total_frames_processed % self.config.stack_interval == 0:
                self.process_buffer()
        
        return score
    
    def is_buffer_full(self) -> bool:
        """Vérifie si le buffer est plein"""
        return self.buffer.is_full()
    
    def get_buffer_fill(self) -> float:
        """Retourne le taux de remplissage du buffer (0-1)"""
        return len(self.buffer) / self.config.buffer_size
    
    def process_buffer(self) -> Optional[np.ndarray]:
        """
        Traite le buffer : sélectionne les meilleures et les stack
        
        Returns:
            Image stackée ou None si pas assez d'images
        """
        if len(self.buffer) < 2:
            return None
        
        t0 = time.perf_counter()
        
        # 1. Déterminer combien d'images garder
        keep_count = self.config.get_keep_count()
        keep_count = min(keep_count, len(self.buffer))
        
        # 2. Sélectionner les meilleures
        best_indices = self.buffer.get_best_indices(keep_count, self.config.min_score)
        
        if not best_indices:
            print(f"[LUCKY] Aucune image au-dessus du seuil min_score={self.config.min_score}")
            return None
        
        # Calculer le seuil de sélection (score de la dernière image sélectionnée)
        buffer_stats = self.buffer.get_statistics()
        if best_indices:
            scores_list = list(self.buffer.scores)
            selected_scores = [scores_list[i] for i in best_indices]
            self.stats['selection_threshold'] = min(selected_scores) if selected_scores else 0
        
        # 3. Récupérer les frames sélectionnées
        selected_frames = self.buffer.get_frames_by_indices(best_indices)
        
        print(f"[LUCKY] Sélection: {len(selected_frames)}/{len(self.buffer)} images "
              f"(seuil={self.stats['selection_threshold']:.1f})")
        
        # 4. Aligner les frames
        if self.config.align_enabled and len(selected_frames) > 1:
            aligned_frames = self._align_frames(selected_frames)
        else:
            aligned_frames = selected_frames
        
        # 5. Stacker
        result = self._stack_frames(aligned_frames)
        
        # Timing et stats
        stack_time = (time.perf_counter() - t0) * 1000
        self._stack_times.append(stack_time)
        
        self.last_result = result
        self.last_stack_time = time.time()
        self.total_stacks_done += 1
        
        # Mettre à jour stats
        self._update_stats(buffer_stats, len(selected_frames))
        
        print(f"[LUCKY] Stack #{self.total_stacks_done}: {len(selected_frames)} images, "
              f"{stack_time:.1f}ms")
        
        # Vider le buffer pour le prochain cycle
        self.buffer.clear()
        self.aligner.reset()
        
        return result
    
    def _align_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Aligne une liste de frames sur la première"""
        if not frames:
            return frames
        
        # Utiliser la première frame comme référence
        self.aligner.set_reference(frames[0])
        
        aligned = [frames[0]]  # La référence est déjà alignée
        
        for frame in frames[1:]:
            aligned_frame, params = self.aligner.align(frame)
            aligned.append(aligned_frame)
        
        return aligned
    
    def _stack_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Combine les frames selon la méthode configurée"""
        if not frames:
            return None
        
        if len(frames) == 1:
            return frames[0].copy()
        
        # Convertir en array 3D/4D
        stack = np.array(frames)
        
        if self.config.stack_method == StackMethod.MEAN:
            result = np.mean(stack, axis=0)
        
        elif self.config.stack_method == StackMethod.MEDIAN:
            result = np.median(stack, axis=0)
        
        elif self.config.stack_method == StackMethod.SIGMA_CLIP:
            result = self._sigma_clip_stack(stack)
        
        else:
            result = np.mean(stack, axis=0)
        
        return result.astype(np.float32)
    
    def _sigma_clip_stack(self, stack: np.ndarray) -> np.ndarray:
        """Stack avec rejection sigma-clipping"""
        kappa = self.config.sigma_clip_kappa
        
        # Calculer médiane et écart-type
        median = np.median(stack, axis=0)
        std = np.std(stack, axis=0)
        
        # Masquer les outliers
        lower = median - kappa * std
        upper = median + kappa * std
        
        # Créer masque
        mask = (stack >= lower) & (stack <= upper)
        
        # Moyenne des valeurs non-masquées
        masked_sum = np.sum(stack * mask, axis=0)
        masked_count = np.sum(mask, axis=0)
        masked_count = np.maximum(masked_count, 1)  # Éviter division par zéro
        
        return masked_sum / masked_count
    
    def _update_stats(self, buffer_stats: Dict, selected_count: int):
        """Met à jour les statistiques"""
        self.stats['frames_selected'] = selected_count
        self.stats['stacks_done'] = self.total_stacks_done
        self.stats['avg_score'] = buffer_stats['mean']
        self.stats['best_score'] = buffer_stats['max']
        self.stats['worst_score'] = buffer_stats['min']
        
        if self._scoring_times:
            self.stats['avg_scoring_time_ms'] = np.mean(list(self._scoring_times))
        if self._stack_times:
            self.stats['avg_stack_time_ms'] = np.mean(list(self._stack_times))
    
    def get_result(self) -> Optional[np.ndarray]:
        """
        Retourne le dernier résultat stacké (et le vide)
        Pattern 'pop': chaque résultat n'est retourné qu'une seule fois
        """
        result = self.last_result
        self.last_result = None  # Vider après récupération
        return result
    
    def get_preview(self, as_uint8: bool = True) -> Optional[np.ndarray]:
        """
        Retourne une preview du dernier résultat
        
        Args:
            as_uint8: Convertir en uint8 pour affichage
        """
        if self.last_result is None:
            return None
        
        if as_uint8:
            # Normaliser pour affichage
            result = self.last_result.copy()
            if result.max() > 255:
                result = (result / result.max() * 255)
            return np.clip(result, 0, 255).astype(np.uint8)
        
        return self.last_result.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques courantes"""
        stats = self.stats.copy()
        stats['buffer_fill'] = len(self.buffer)
        stats['buffer_size'] = self.config.buffer_size
        stats['buffer_percent'] = 100.0 * len(self.buffer) / self.config.buffer_size
        return stats
    
    def get_score_histogram(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne l'histogramme des scores"""
        if not self.score_history:
            return np.array([]), np.array([])
        
        scores = np.array(list(self.score_history))
        hist, bin_edges = np.histogram(scores, bins=bins)
        return hist, bin_edges
    
    def configure(self, **kwargs):
        """
        Configure les paramètres à la volée
        
        Paramètres modifiables:
        - buffer_size: int
        - keep_percent: float
        - keep_count: int
        - min_score: float
        - score_method: str ("laplacian", "gradient", "sobel", "tenengrad")
        - stack_method: str ("mean", "median", "sigma_clip")
        - align_enabled: bool
        - auto_stack: bool
        """
        if 'buffer_size' in kwargs:
            new_size = int(kwargs['buffer_size'])
            if new_size != self.config.buffer_size:
                self.config.buffer_size = new_size
                self.buffer = FrameBuffer(new_size)
        
        if 'keep_percent' in kwargs:
            self.config.keep_percent = float(kwargs['keep_percent'])
        
        if 'keep_count' in kwargs:
            self.config.keep_count = int(kwargs['keep_count']) if kwargs['keep_count'] else None
        
        if 'min_score' in kwargs:
            self.config.min_score = float(kwargs['min_score'])
        
        if 'score_method' in kwargs:
            method_str = kwargs['score_method'].lower()
            method_map = {
                'laplacian': ScoreMethod.LAPLACIAN,
                'gradient': ScoreMethod.GRADIENT,
                'sobel': ScoreMethod.SOBEL,
                'tenengrad': ScoreMethod.TENENGRAD,
                'brenner': ScoreMethod.BRENNER,
                'fft': ScoreMethod.FFT,
            }
            if method_str in method_map:
                self.config.score_method = method_map[method_str]
                self.scorer = QualityScorer(self.config)
        
        if 'stack_method' in kwargs:
            method_str = kwargs['stack_method'].lower()
            method_map = {
                'mean': StackMethod.MEAN,
                'median': StackMethod.MEDIAN,
                'sigma_clip': StackMethod.SIGMA_CLIP,
            }
            if method_str in method_map:
                self.config.stack_method = method_map[method_str]
        
        if 'align_enabled' in kwargs:
            self.config.align_enabled = bool(kwargs['align_enabled'])
        
        if 'auto_stack' in kwargs:
            self.config.auto_stack = bool(kwargs['auto_stack'])
        
        if 'score_roi_percent' in kwargs:
            self.config.score_roi_percent = float(kwargs['score_roi_percent'])
        
        if 'sigma_clip_kappa' in kwargs:
            self.config.sigma_clip_kappa = float(kwargs['sigma_clip_kappa'])
    
    def reset(self):
        """Réinitialise le stacker (garde la config)"""
        self.buffer.clear()
        self.aligner.reset()
        self.score_history.clear()
        self.last_result = None
        self._scoring_times.clear()
        self._stack_times.clear()
        self.stats = {
            'frames_total': 0,
            'frames_selected': 0,
            'stacks_done': 0,
            'avg_score': 0.0,
            'best_score': 0.0,
            'worst_score': 0.0,
            'avg_scoring_time_ms': 0.0,
            'avg_stack_time_ms': 0.0,
            'selection_threshold': 0.0,
        }
        print("[LUCKY] Reset")


# =============================================================================
# Wrapper pour intégration RPiCamera
# =============================================================================

class RPiCameraLuckyImaging:
    """
    Wrapper Lucky Imaging pour intégration dans RPiCamera.py
    
    Gère:
    - Configuration via paramètres simples
    - Preview pour affichage PyGame
    - Statistiques pour OSD
    """
    
    def __init__(self, output_dir: str = "/media/admin/THKAILAR/Lucky"):
        self.output_dir = output_dir
        self.stacker: Optional[LuckyImagingStacker] = None
        self.config = LuckyConfig()
        
        # État
        self.is_running = False
        self.frame_count = 0
        self.last_preview = None
        self.start_time = None
    
    def configure(self, **kwargs):
        """
        Configure le Lucky Imaging
        
        Paramètres:
        - buffer_size: int (défaut: 100)
        - keep_percent: float (défaut: 10.0)
        - keep_count: int (optionnel, prioritaire sur keep_percent)
        - min_score: float (défaut: 0)
        - score_method: str ("laplacian", "gradient", "sobel", "tenengrad")
        - stack_method: str ("mean", "median", "sigma_clip")
        - align_enabled: bool (défaut: True)
        - auto_stack: bool (défaut: True)
        - score_roi_percent: float (défaut: 50.0)
        """
        if 'buffer_size' in kwargs:
            self.config.buffer_size = int(kwargs['buffer_size'])
        if 'keep_percent' in kwargs:
            self.config.keep_percent = float(kwargs['keep_percent'])
        if 'keep_count' in kwargs:
            self.config.keep_count = int(kwargs['keep_count']) if kwargs['keep_count'] else None
        if 'min_score' in kwargs:
            self.config.min_score = float(kwargs['min_score'])
        if 'score_method' in kwargs:
            method_map = {
                'laplacian': ScoreMethod.LAPLACIAN,
                'gradient': ScoreMethod.GRADIENT,
                'sobel': ScoreMethod.SOBEL,
                'tenengrad': ScoreMethod.TENENGRAD,
            }
            self.config.score_method = method_map.get(kwargs['score_method'].lower(), 
                                                       ScoreMethod.LAPLACIAN)
        if 'stack_method' in kwargs:
            method_map = {
                'mean': StackMethod.MEAN,
                'median': StackMethod.MEDIAN,
                'sigma_clip': StackMethod.SIGMA_CLIP,
            }
            self.config.stack_method = method_map.get(kwargs['stack_method'].lower(),
                                                       StackMethod.MEAN)
        if 'align_enabled' in kwargs:
            self.config.align_enabled = bool(kwargs['align_enabled'])
        if 'auto_stack' in kwargs:
            self.config.auto_stack = bool(kwargs['auto_stack'])
        if 'score_roi_percent' in kwargs:
            self.config.score_roi_percent = float(kwargs['score_roi_percent'])
    
    def start(self):
        """Démarre le Lucky Imaging"""
        self.config.validate()
        self.stacker = LuckyImagingStacker(self.config)
        self.stacker.start()
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"[LUCKY] Session démarrée")
        print(f"  Buffer: {self.config.buffer_size} images")
        print(f"  Garder: {self.config.keep_percent}%")
        print(f"  Méthode score: {self.config.score_method.value}")
        print(f"  Méthode stack: {self.config.stack_method.value}")
    
    def stop(self):
        """Arrête le Lucky Imaging"""
        if self.stacker:
            self.stacker.stop()
        self.is_running = False
        print(f"[LUCKY] Session arrêtée - {self.frame_count} frames traitées")
    
    def process_frame(self, image_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Traite une frame
        
        Args:
            image_data: Image de la caméra
        
        Returns:
            Image stackée si disponible, None sinon
        """
        if not self.is_running or self.stacker is None:
            return None
        
        # Ajouter la frame
        score = self.stacker.add_frame(image_data)
        self.frame_count += 1
        
        # Mettre à jour preview si nouveau résultat
        if self.stacker.last_result is not None:
            self.last_preview = self.stacker.get_preview(as_uint8=True)
            return self.last_preview
        
        return None
    
    def get_preview_surface(self, pygame_module, target_size=None):
        """
        Retourne une surface PyGame pour affichage
        
        Args:
            pygame_module: Module pygame
            target_size: (width, height) optionnel
        
        Returns:
            pygame.Surface ou None
        """
        if self.last_preview is None:
            return None
        
        try:
            preview = self.last_preview
            
            if len(preview.shape) == 3:
                surface = pygame_module.surfarray.make_surface(
                    preview.transpose(1, 0, 2)
                )
            else:
                # Grayscale -> RGB
                preview_rgb = np.stack([preview, preview, preview], axis=-1)
                surface = pygame_module.surfarray.make_surface(
                    preview_rgb.transpose(1, 0, 2)
                )
            
            if target_size:
                surface = pygame_module.transform.scale(surface, target_size)
            
            return surface
        except Exception as e:
            print(f"[LUCKY] Erreur preview: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques pour affichage OSD"""
        if self.stacker is None:
            return {}
        
        stats = self.stacker.get_stats()
        
        # Ajouter infos supplémentaires
        elapsed = time.time() - self.start_time if self.start_time else 0
        stats['elapsed_time'] = elapsed
        stats['fps'] = self.frame_count / elapsed if elapsed > 0 else 0
        
        return stats
    
    def get_osd_text(self) -> List[str]:
        """Retourne les lignes de texte pour l'OSD"""
        stats = self.get_stats()
        
        lines = [
            f"LUCKY: {stats.get('buffer_fill', 0)}/{stats.get('buffer_size', 0)}",
            f"Score: {stats.get('avg_score', 0):.0f} (>{stats.get('selection_threshold', 0):.0f})",
            f"Stacks: {stats.get('stacks_done', 0)}",
            f"FPS: {stats.get('fps', 0):.1f}",
        ]
        
        return lines
    
    def save_result(self, filename: str = None):
        """Sauvegarde le dernier résultat"""
        if self.stacker is None or self.stacker.last_result is None:
            print("[LUCKY] Pas de résultat à sauvegarder")
            return
        
        import os
        from datetime import datetime
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lucky_{timestamp}"
        
        # Sauvegarder en PNG
        result = self.stacker.last_result
        if result.max() > 255:
            result_8bit = (result / result.max() * 255).astype(np.uint8)
        else:
            result_8bit = result.astype(np.uint8)
        
        png_path = os.path.join(self.output_dir, f"{filename}.png")
        
        if len(result_8bit.shape) == 3:
            cv2.imwrite(png_path, cv2.cvtColor(result_8bit, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(png_path, result_8bit)
        
        print(f"[LUCKY] Sauvegardé: {png_path}")
        
        # Sauvegarder stats
        stats_path = os.path.join(self.output_dir, f"{filename}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write("=== LUCKY IMAGING STATS ===\n\n")
            for key, value in self.get_stats().items():
                f.write(f"{key}: {value}\n")
        
        return png_path
    
    def reset(self):
        """Réinitialise la session"""
        if self.stacker:
            self.stacker.reset()
        self.frame_count = 0
        self.last_preview = None
        self.start_time = time.time()


# =============================================================================
# Factory function
# =============================================================================

def create_lucky_session(preset: str = 'default', **kwargs) -> RPiCameraLuckyImaging:
    """
    Crée une session Lucky Imaging avec preset
    
    Presets:
    - 'default': Équilibré (100 images, 10%)
    - 'fast': Rapide (50 images, 20%)
    - 'quality': Haute qualité (200 images, 5%)
    - 'aggressive': Très sélectif (200 images, 1%)
    
    Args:
        preset: Nom du preset
        **kwargs: Paramètres supplémentaires
    
    Returns:
        RPiCameraLuckyImaging configuré
    """
    session = RPiCameraLuckyImaging()
    
    # Appliquer preset
    if preset == 'fast':
        session.configure(
            buffer_size=50,
            keep_percent=20.0,
            score_method='laplacian',
            stack_method='mean',
        )
    elif preset == 'quality':
        session.configure(
            buffer_size=200,
            keep_percent=5.0,
            score_method='tenengrad',
            stack_method='sigma_clip',
        )
    elif preset == 'aggressive':
        session.configure(
            buffer_size=200,
            keep_percent=1.0,
            score_method='tenengrad',
            stack_method='sigma_clip',
        )
    else:  # default
        session.configure(
            buffer_size=100,
            keep_percent=10.0,
            score_method='laplacian',
            stack_method='mean',
        )
    
    # Appliquer paramètres supplémentaires
    session.configure(**kwargs)
    
    return session


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    print("=== Test Lucky Imaging ===\n")
    
    np.random.seed(42)
    
    # Créer images de test avec qualité variable
    def create_test_image(quality: float) -> np.ndarray:
        """Crée une image test avec netteté variable"""
        h, w = 256, 256
        
        # Image de base : disque avec détails
        y, x = np.ogrid[:h, :w]
        center = (128, 128)
        radius = 80
        
        disk = ((x - center[0])**2 + (y - center[1])**2 <= radius**2).astype(np.float32)
        
        # Ajouter détails (taches)
        for _ in range(5):
            tx = np.random.randint(60, 196)
            ty = np.random.randint(60, 196)
            tr = np.random.randint(5, 15)
            spot = ((x - tx)**2 + (y - ty)**2 <= tr**2).astype(np.float32)
            disk -= spot * 0.3
        
        disk = np.clip(disk * 200 + 30, 0, 255)
        
        # Appliquer flou selon qualité (moins de flou = meilleure qualité)
        blur_sigma = (1.0 - quality) * 5 + 0.5
        disk = cv2.GaussianBlur(disk, (0, 0), blur_sigma)
        
        # Ajouter bruit
        noise = np.random.normal(0, 5, disk.shape)
        disk = np.clip(disk + noise, 0, 255)
        
        return disk.astype(np.float32)
    
    # Test 1: Scoring
    print("--- Test Scoring ---")
    config = LuckyConfig()
    scorer = QualityScorer(config)
    
    for q in [0.2, 0.5, 0.8, 1.0]:
        img = create_test_image(q)
        score = scorer.score(img)
        print(f"  Qualité={q:.1f} -> Score={score:.1f}")
    
    # Test 2: Buffer et sélection
    print("\n--- Test Buffer et Sélection ---")
    config = LuckyConfig(buffer_size=20, keep_percent=25.0)
    stacker = LuckyImagingStacker(config)
    stacker.start()
    
    # Ajouter 20 images avec qualité aléatoire
    qualities = np.random.uniform(0.2, 1.0, 20)
    for i, q in enumerate(qualities):
        img = create_test_image(q)
        score = stacker.add_frame(img)
        print(f"  Frame {i+1}: qualité={q:.2f}, score={score:.1f}")
    
    # Le buffer devrait être plein et auto-stacké
    stats = stacker.get_stats()
    print(f"\n  Résultat: {stats['frames_selected']}/{stats['buffer_fill']} images sélectionnées")
    print(f"  Seuil de sélection: {stats['selection_threshold']:.1f}")
    print(f"  Score moyen: {stats['avg_score']:.1f}")
    print(f"  Temps scoring: {stats['avg_scoring_time_ms']:.2f}ms")
    print(f"  Temps stack: {stats['avg_stack_time_ms']:.2f}ms")
    
    # Test 3: Wrapper RPiCamera
    print("\n--- Test Wrapper RPiCamera ---")
    session = create_lucky_session('quality')
    session.start()
    
    for i in range(50):
        q = np.random.uniform(0.3, 1.0)
        img = create_test_image(q)
        result = session.process_frame(img)
        
        if result is not None:
            print(f"  Stack généré à la frame {i+1}")
    
    osd = session.get_osd_text()
    print(f"\n  OSD: {osd}")
    
    session.stop()
    
    print("\n=== Tests terminés ===")
