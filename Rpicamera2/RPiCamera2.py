#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Gordon999
# SPDX-License-Identifier: MIT

"""Copyright (c) 2025
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import time
import pygame
from pygame.locals import *
import os, sys
import datetime
import subprocess
import signal
import cv2
import glob
from datetime import timedelta
import numpy as np
import math
from pathlib import Path
from gpiozero import Button
from gpiozero import LED
import struct
from collections import deque

# Cache global pour les polices pygame (optimisation performance)
_font_cache = {}
import threading
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage pour éviter conflits avec pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO

#!/usr/bin/env python3
import sys
import os

# Configuration IMX585 - À mettre TOUT EN HAUT avant tous les imports
sys.path.insert(0, '/usr/local/lib/aarch64-linux-gnu/python3.11/site-packages')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib/aarch64-linux-gnu'
os.environ['LIBCAMERA_RPI_CONFIG_FILE'] = ''

# Imports Picamera2 (après configuration IMX585)
from picamera2 import Picamera2
from libcamera import controls, Transform

# Import Live Stack modules (dans libastrostack/)
from libastrostack.rpicamera_livestack import create_livestack_session
from libastrostack.rpicamera_livestack_advanced import create_advanced_livestack_session

# Import Allsky module (dans libastrostack/)
from libastrostack.allsky import AllskyMeanController

# ============================================================================
# Helper pour tuer le subprocess rpicam-vid (mode non-Picamera2)
# ============================================================================
def kill_preview_process():
    """Tue le processus rpicam-vid si actif (mode non-Picamera2)"""
    global p
    if not use_picamera2 and p is not None:
        try:
            poll = p.poll()
            if poll is None:
                os.killpg(p.pid, signal.SIGTERM)
        except:
            pass

# ============================================================================
# Post-traitement vidéo pour correction des timestamps (Pi 5)
# ============================================================================
def fix_video_timestamps(input_file, fps_value, quality_preset="ultrafast"):
    """
    Corrige les timestamps des vidéos MP4/H264 sur Pi 5 via ffmpeg.

    Problème: rpicam-vid sur Pi 5 génère des fichiers avec timestamps incorrects
    Solution: Réencodage avec ffmpeg pour recalculer les timestamps

    Args:
        input_file: Chemin du fichier vidéo brut (avec timestamps incorrects)
        fps_value: Framerate de la vidéo (utilisé pour recalculer les timestamps)
        quality_preset: Preset ffmpeg (ultrafast, veryfast, medium, slow)

    Returns:
        True si la correction a réussi, False sinon
    """
    import os
    import subprocess

    # Créer le nom du fichier temporaire
    temp_file = input_file.replace(".mp4", "_temp.mp4").replace(".h264", "_temp.h264")

    # Renommer le fichier original en temporaire
    try:
        os.rename(input_file, temp_file)
    except Exception as e:
        print(f"Erreur lors du renommage: {e}")
        return False

    # Construire la commande ffmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", temp_file,
        "-vf", f"setpts=N/{fps_value}/TB",
        "-r", str(fps_value),
        "-c:v", "libx264",
        "-preset", quality_preset,
        input_file,
        "-y",
        "-loglevel", "error"
    ]

    try:
        # Exécuter ffmpeg
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Succès - supprimer le fichier temporaire
            try:
                os.remove(temp_file)
            except:
                pass
            return True
        else:
            # Échec - restaurer le fichier original
            print(f"Erreur ffmpeg: {result.stderr}")
            try:
                os.rename(temp_file, input_file)
            except:
                pass
            return False

    except subprocess.TimeoutExpired:
        print("Timeout ffmpeg - fichier trop long")
        try:
            os.rename(temp_file, input_file)
        except:
            pass
        return False
    except Exception as e:
        print(f"Erreur lors de l'exécution de ffmpeg: {e}")
        try:
            os.rename(temp_file, input_file)
        except:
            pass
        return False

# ============================================================================
# Helpers pour Allsky Timelapse
# ============================================================================
def assemble_allsky_video(pic_dir, timestamp, fps, output_filename):
    """
    Assemble une séquence JPEG en vidéo MP4 via FFmpeg

    Args:
        pic_dir: Répertoire contenant les JPEGs
        timestamp: Préfixe timestamp des fichiers (ex: "250112123045")
        fps: Framerate de la vidéo finale
        output_filename: Chemin complet du fichier MP4 de sortie

    Returns:
        bool: True si succès, False sinon

    Pattern fichiers : timestamp_%04d.jpg (0001, 0002, ...)
    Codec : libx264, preset medium, CRF 23, yuv420p
    """
    import subprocess
    import os

    try:
        input_pattern = os.path.join(pic_dir, f"{timestamp}_%04d.jpg")

        ffmpeg_cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",  # Qualité (18-28, plus bas = meilleure qualité)
            "-pix_fmt", "yuv420p",  # Compatibilité maximale
            "-y",  # Overwrite output file
            "-loglevel", "error",
            output_filename
        ]

        print(f"[Allsky] Assemblage vidéo: {output_filename}")
        print(f"[Allsky] Commande FFmpeg: {' '.join(ffmpeg_cmd)}")

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print(f"[Allsky] ✓ Vidéo assemblée avec succès")
            return True
        else:
            print(f"[Allsky] ✗ Erreur FFmpeg: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("[Allsky] ✗ Timeout FFmpeg - vidéo trop longue ou processus bloqué")
        return False
    except Exception as e:
        print(f"[Allsky] ✗ Erreur assemblage vidéo: {e}")
        return False


def apply_stretch_to_jpeg(jpeg_path, stretch_preset):
    """
    Applique astro_stretch() à un JPEG et le sauvegarde

    Args:
        jpeg_path: Chemin du fichier JPEG
        stretch_preset: Preset de stretch (0=OFF, 1=GHS, 2=Arcsinh)

    Returns:
        bool: True si succès, False sinon

    Utilise les paramètres stretch globaux (ghs_D, ghs_b, etc.)
    Qualité JPEG = 95 pour minimiser les artefacts de compression
    """
    import cv2

    try:
        # Skip si stretch OFF
        if stretch_preset == 0:
            return True

        # Charger JPEG
        img = cv2.imread(jpeg_path)
        if img is None:
            print(f"[Allsky] ✗ Échec chargement image: {jpeg_path}")
            return False

        # Convertir BGR (OpenCV) -> RGB (astro_stretch)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Appliquer stretch (utilise fonction existante et paramètres globaux)
        stretched = astro_stretch(img_rgb)

        # Convertir RGB -> BGR pour sauvegarde
        stretched_bgr = cv2.cvtColor(stretched, cv2.COLOR_RGB2BGR)

        # Sauvegarder avec qualité élevée
        cv2.imwrite(jpeg_path, stretched_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return True

    except Exception as e:
        print(f"[Allsky] ✗ Erreur stretch {jpeg_path}: {e}")
        return False


# ============================================================================
# Helpers pour gérer Picamera2 temporairement pendant rpicam-vid
# ============================================================================
def pause_picamera2():
    """
    Ferme complètement Picamera2 pour libérer la caméra.
    IMPORTANT: Doit être suivi de resume_picamera2() pour recréer Picamera2.
    """
    global picam2, use_picamera2, capture_thread

    print(f"[DEBUG] pause_picamera2() called - use_picamera2={use_picamera2}, picam2={picam2 is not None}")

    if use_picamera2 and picam2 is not None:
        try:
            # Arrêter le thread de capture d'abord
            if capture_thread is not None:
                print("[DEBUG] Stopping capture thread...")
                capture_thread.stop()
                capture_thread = None

            # Fermer complètement Picamera2 pour libérer le pipeline
            print("[DEBUG] Closing Picamera2 completely...")
            picam2.stop()
            picam2.close()
            print("[DEBUG] Picamera2 closed, waiting for camera to be released...")
            import time
            time.sleep(1.0)  # Attendre que la caméra soit vraiment libérée
            print("[DEBUG] Picamera2 closed successfully - camera should now be free")
            return True
        except Exception as e:
            print(f"[DEBUG] Erreur lors de la fermeture de Picamera2: {e}")
            import traceback
            traceback.print_exc()
            return False
    print("[DEBUG] Picamera2 not paused (not in use or None)")
    return False

def resume_picamera2():
    """
    Recrée et redémarre Picamera2 après une pause temporaire (après close()).
    """
    global picam2, use_picamera2

    print(f"[DEBUG] resume_picamera2() called - use_picamera2={use_picamera2}")

    if use_picamera2:
        try:
            print("[DEBUG] Recreating Picamera2 via preview() function...")
            preview()  # Appelle la fonction qui crée et configure Picamera2
            print("[DEBUG] Picamera2 recreated and started successfully")
            return True
        except Exception as e:
            print(f"[DEBUG] Erreur lors de la reprise de Picamera2: {e}")
            import traceback
            traceback.print_exc()
            return False
    print("[DEBUG] Picamera2 not resumed (not in use)")
    return False


def apply_controls_immediately(exposure_time=None, gain_value=None):
    """
    Applique immédiatement les changements de contrôles (exposition, gain)
    sans recréer la caméra. Annule la capture en cours pour appliquer
    les nouveaux paramètres plus rapidement.

    Args:
        exposure_time: Temps d'exposition en microsecondes (None = ne pas changer)
        gain_value: Valeur de gain (None = ne pas changer)

    Returns:
        True si les contrôles ont été appliqués, False sinon
    """
    global picam2, capture_thread, Pi_Cam, max_shutters, livestack_active, luckystack_active, raw_format

    if not use_picamera2 or picam2 is None:
        return False

    try:
        controls_to_apply = {}

        # Préparer les contrôles à appliquer
        if exposure_time is not None:
            max_exposure_seconds = max_shutters[Pi_Cam]
            max_frame_duration = int(max_exposure_seconds * 1_000_000)
            min_frame_duration = 11415 if Pi_Cam == 10 else 100

            controls_to_apply["FrameDurationLimits"] = (min_frame_duration, max(max_frame_duration, exposure_time))
            controls_to_apply["ExposureTime"] = exposure_time

        if gain_value is not None:
            controls_to_apply["AnalogueGain"] = float(gain_value)

        # Appliquer les contrôles
        if controls_to_apply:
            picam2.set_controls(controls_to_apply)

            # IMPORTANT: Redémarrer le thread pour que les nouveaux paramètres
            # s'appliquent immédiatement (sinon le thread continue avec les anciens)
            if capture_thread is not None:
                if show_cmds == 1:
                    print(f"[AsyncCapture] Redémarrage du thread pour nouveaux paramètres")

                # Arrêter le thread actuel
                capture_thread.stop()

                # Recréer et redémarrer avec les mêmes paramètres de type
                new_thread = AsyncCaptureThread(picam2)
                if (livestack_active or luckystack_active) and raw_format >= 2:
                    new_thread.set_capture_params({'type': 'raw'})
                else:
                    new_thread.set_capture_params({'type': 'main'})
                new_thread.start()

                # Remplacer l'ancien thread
                capture_thread = new_thread

            if show_cmds == 1:
                print(f"[AsyncCapture] Contrôles appliqués immédiatement: {controls_to_apply}")

            return True

    except Exception as e:
        print(f"[AsyncCapture] Erreur lors de l'application des contrôles: {e}")
        return False

    return False


# ============================================================================
# Capture asynchrone pour éviter le blocage de l'interface
# ============================================================================
class AsyncCaptureThread:
    """
    Thread de capture asynchrone pour Picamera2.
    Permet de capturer des images en arrière-plan sans bloquer l'interface.
    """
    def __init__(self, picam2_instance):
        self.picam2 = picam2_instance
        self.thread = None
        self.running = False
        self.capturing = False
        self.latest_frame = None
        self.latest_metadata = None
        self.frame_lock = threading.Lock()
        self.cancel_current = False  # Flag pour ignorer la capture en cours
        self.capture_params = {}  # Paramètres de la capture en cours

    def start(self):
        """Démarre le thread de capture"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("[AsyncCapture] Thread de capture démarré")

    def stop(self):
        """Arrête le thread de capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            print("[AsyncCapture] Thread de capture arrêté")

    def _capture_loop(self):
        """Boucle de capture en arrière-plan"""
        while self.running:
            try:
                if self.picam2 is None:
                    time.sleep(0.1)
                    continue

                self.capturing = True
                self.cancel_current = False

                # Capturer l'image (BLOQUANT mais dans le thread)
                # On ne peut pas vraiment interrompre l'intégration hardware
                frame = None
                metadata = None

                try:
                    # Capturer selon le type demandé
                    capture_type = self.capture_params.get('type', 'main')
                    if capture_type == 'raw':
                        frame = self.picam2.capture_array("raw")
                    else:
                        frame = self.picam2.capture_array("main")
                    metadata = self.picam2.capture_metadata()
                except Exception as e:
                    print(f"[AsyncCapture] Erreur capture: {e}")
                    self.capturing = False
                    time.sleep(0.1)
                    continue

                # Si la capture a été annulée pendant l'acquisition, on ignore le résultat
                if self.cancel_current:
                    print(f"[AsyncCapture] Capture annulée (nouveaux paramètres appliqués)")
                    self.capturing = False
                    continue

                # Stocker la frame capturée
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_metadata = metadata

                self.capturing = False

            except Exception as e:
                print(f"[AsyncCapture] Erreur dans la boucle de capture: {e}")
                self.capturing = False
                time.sleep(0.1)

    def get_latest_frame(self):
        """
        Récupère la dernière frame capturée (non-bloquant).
        Retourne (frame, metadata) ou (None, None) si aucune frame disponible.
        """
        with self.frame_lock:
            frame = self.latest_frame
            metadata = self.latest_metadata
            # Réinitialiser pour éviter de réutiliser la même frame
            self.latest_frame = None
            self.latest_metadata = None
            return frame, metadata

    def cancel_capture(self):
        """
        Marque la capture en cours comme obsolète.
        Elle continuera physiquement (limitation hardware) mais sera ignorée.
        """
        self.cancel_current = True
        print(f"[AsyncCapture] Annulation de la capture en cours demandée")

    def set_capture_params(self, params):
        """Définit les paramètres pour la prochaine capture"""
        self.capture_params = params.copy()

    def is_capturing(self):
        """Retourne True si une capture est en cours"""
        return self.capturing


def create_ser_header(width, height, pixel_depth=8, color_id=100):
    """
    Crée l'en-tête d'un fichier SER
    color_id: 0=MONO, 8=BAYER_RGGB, 9=BAYER_GRBG, 10=BAYER_GBRG, 11=BAYER_BGGR, 
              100=RGB, 101=BGR
    """
    header = bytearray(178)
    
    # Signature "LUCAM-RECORDER"
    header[0:14] = b'LUCAM-RECORDER'
    
    # LuID (4 bytes) - peut être 0
    struct.pack_into('<I', header, 14, 0)
    
    # ColorID (4 bytes)
    struct.pack_into('<I', header, 18, color_id)
    
    # LittleEndian (4 bytes) - 0 pour little endian
    struct.pack_into('<I', header, 22, 0)
    
    # ImageWidth (4 bytes)
    struct.pack_into('<I', header, 26, width)
    
    # ImageHeight (4 bytes)
    struct.pack_into('<I', header, 30, height)
    
    # PixelDepth (4 bytes) - bits per pixel per channel
    struct.pack_into('<I', header, 34, pixel_depth)
    
    # FrameCount (4 bytes) - sera mis à jour à la fin
    struct.pack_into('<I', header, 38, 0)
    
    # Observer (40 bytes)
    observer = b'RPiCamGUI'
    header[42:42+len(observer)] = observer
    
    # Instrument (40 bytes)
    instrument = b'Raspberry Pi Camera'
    header[82:82+len(instrument)] = instrument
    
    # Telescope (40 bytes)
    telescope = b''
    header[122:162] = telescope.ljust(40, b'\x00')
    
    # DateTime (8 bytes) - timestamp en microseconds depuis epoch
    timestamp = int(time.time() * 1000000)
    struct.pack_into('<Q', header, 162, timestamp)
    
    # DateTime_UTC (8 bytes)
    struct.pack_into('<Q', header, 170, timestamp)
    
    return bytes(header)

def update_ser_frame_count(filename, frame_count):
    """Met à jour le nombre de frames dans l'en-tête SER"""
    with open(filename, 'r+b') as f:
        f.seek(38)
        f.write(struct.pack('<I', frame_count))

def convert_raw_to_ser(raw_input, ser_output, width, height, fps=None, bit_depth=8, progress_callback=None):
    """
    Convertit un fichier .raw (RGB) en fichier SER

    Args:
        raw_input: Chemin du fichier .raw d'entrée
        ser_output: Chemin du fichier .ser de sortie
        width: Largeur des frames
        height: Hauteur des frames
        fps: Framerate de la vidéo (optionnel, défaut=25). Utilisé pour calculer les timestamps
        bit_depth: Profondeur en bits (8 ou 16, défaut=8)
        progress_callback: Fonction appelée avec (frame_num, total_frames) pour afficher la progression

    Returns:
        (success, frame_count, message)
    """

    if not os.path.exists(raw_input):
        return False, 0, f"Fichier d'entrée introuvable: {raw_input}"

    # Utiliser fps par défaut si non spécifié
    if fps is None:
        fps = 25

    file_size = os.path.getsize(raw_input)

    # Calculer bytes par frame selon la profondeur
    if bit_depth == 16:
        bytes_per_frame = width * height * 6  # RGB48 (16-bit par canal)
    else:
        bytes_per_frame = width * height * 3  # RGB24 (8-bit par canal)

    # Calculer le nombre de frames complètes
    total_frames = file_size // bytes_per_frame
    remaining_bytes = file_size % bytes_per_frame

    if total_frames == 0:
        return False, 0, "Aucune frame complète dans le fichier"

    print(f"📊 Informations:")
    print(f"   Fichier entrée: {raw_input}")
    print(f"   Taille: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
    print(f"   Résolution: {width}x{height}")
    print(f"   Profondeur: {bit_depth}-bit")
    print(f"   Frames complètes: {total_frames}")
    print(f"   Framerate: {fps} fps")
    if remaining_bytes > 0:
        print(f"   ⚠  Bytes ignorés en fin de fichier: {remaining_bytes}")
    print()

    try:
        frame_count = 0
        frame_timestamps = []

        # Calculer le timestamp de départ (maintenant)
        start_time = datetime.datetime.utcnow()
        start_timestamp_us = int(start_time.timestamp() * 1000000)

        # Calculer l'intervalle entre frames en microsecondes
        frame_interval_us = int(1000000 / fps)

        with open(ser_output, 'wb') as ser_file:
            # Écrire l'en-tête SER avec la bonne profondeur
            header = create_ser_header(width, height, bit_depth, 100)  # RGB, bit_depth
            ser_file.write(header)

            # Ouvrir le fichier RAW et convertir frame par frame
            with open(raw_input, 'rb') as raw_file:
                for i in range(total_frames):
                    # Lire une frame RGB
                    frame_data = raw_file.read(bytes_per_frame)

                    if len(frame_data) != bytes_per_frame:
                        print(f"⚠  Frame {i+1}: données incomplètes, arrêt")
                        break

                    # Calculer le timestamp de cette frame
                    frame_timestamp = start_timestamp_us + (i * frame_interval_us)
                    frame_timestamps.append(frame_timestamp)

                    # Écrire directement dans le fichier SER
                    # (les données RGB sont déjà au bon format)
                    ser_file.write(frame_data)
                    frame_count += 1

                    # Afficher la progression
                    if progress_callback:
                        progress_callback(i + 1, total_frames)
                    elif (i + 1) % 50 == 0 or i == 0:
                        print(f"   Conversion: {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.1f}%)")

            # Écrire le trailer avec les timestamps (SER v3)
            # Chaque timestamp est un entier 64-bit little-endian (microsecondes depuis epoch)
            for timestamp in frame_timestamps:
                ser_file.write(struct.pack('<Q', timestamp))

        # Mettre à jour le nombre de frames dans l'en-tête
        update_ser_frame_count(ser_output, frame_count)

        output_size = os.path.getsize(ser_output)

        print()
        print(f"✓ Conversion terminée!")
        print(f"   Fichier sortie: {ser_output}")
        print(f"   Frames converties: {frame_count}")
        print(f"   Taille: {output_size:,} bytes ({output_size/(1024*1024):.2f} MB)")
        print(f"   Taille par frame: {output_size/frame_count/1024:.2f} KB")

        return True, frame_count, "Conversion réussie"

    except Exception as e:
        return False, 0, f"Erreur: {str(e)}"

def yuv420_to_rgb(yuv_data, width, height):
    """
    Convertit une frame YUV420 en RGB

    YUV420 format:
    - Y plane: width * height bytes
    - U plane: (width/2) * (height/2) bytes
    - V plane: (width/2) * (height/2) bytes

    Note: Les dimensions doivent être paires pour YUV420
    """
    # S'assurer que les dimensions sont paires pour YUV420
    # Si elles sont impaires, on les ajuste (rare mais possible)
    actual_width = width
    actual_height = height

    # Calculer les dimensions UV (sous-échantillonnées par 2)
    uv_width = (width + 1) // 2  # Arrondi supérieur si impair
    uv_height = (height + 1) // 2

    y_size = width * height
    uv_size = uv_width * uv_height

    try:
        # Extraire les plans Y, U, V
        y_plane = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape((height, width))

        # Vérifier qu'on a assez de données pour U et V
        if len(yuv_data) < y_size + 2 * uv_size:
            # Fallback: créer des plans UV neutres
            print(f"   ⚠️  Données UV incomplètes, utilisation de valeurs neutres")
            u_plane = np.full((uv_height, uv_width), 128, dtype=np.uint8)
            v_plane = np.full((uv_height, uv_width), 128, dtype=np.uint8)
        else:
            u_plane = np.frombuffer(yuv_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((uv_height, uv_width))
            v_plane = np.frombuffer(yuv_data[y_size + uv_size:y_size + 2 * uv_size], dtype=np.uint8).reshape((uv_height, uv_width))

        # Upscale U et V pour correspondre à Y (utiliser la taille exacte de Y)
        u_upscale = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
        v_upscale = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

        # Recombiner en YUV
        yuv_image = np.stack([y_plane, u_upscale, v_upscale], axis=-1).astype(np.uint8)

        # Convertir YUV → RGB
        rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

        return rgb_image

    except Exception as e:
        print(f"   ❌ Erreur conversion YUV->RGB: {e}")
        print(f"      Dimensions: {width}x{height}, UV: {uv_width}x{uv_height}")
        print(f"      Taille données: {len(yuv_data)}, Y: {y_size}, UV: {uv_size}")
        # Retourner une image noire en cas d'erreur
        return np.zeros((height, width, 3), dtype=np.uint8)

def convert_yuv420_to_ser(yuv_file, ser_file, width, height, fps=25):
    """
    Convertit un fichier YUV420 en SER avec timestamps
    Optimisé pour éviter la saturation RAM - traite les frames par lots

    Args:
        yuv_file: Fichier .yuv d'entrée
        ser_file: Fichier .ser de sortie
        width: Largeur
        height: Hauteur
        fps: Framerate pour les timestamps

    Returns:
        (success, frame_count, message)
    """

    if not os.path.exists(yuv_file):
        return False, 0, f"Fichier YUV introuvable: {yuv_file}"

    # Calculer la taille d'une frame YUV420 en tenant compte des dimensions exactes
    # Y plane: width * height
    # U plane: ((width+1)//2) * ((height+1)//2)
    # V plane: ((width+1)//2) * ((height+1)//2)
    y_size = width * height
    uv_width = (width + 1) // 2
    uv_height = (height + 1) // 2
    uv_size = uv_width * uv_height
    yuv_frame_size = y_size + 2 * uv_size
    rgb_frame_size = width * height * 3

    file_size = os.path.getsize(yuv_file)
    total_frames = file_size // yuv_frame_size

    if total_frames == 0:
        return False, 0, "Aucune frame YUV420 trouvée"

    print(f"📹 Conversion YUV420 → SER (optimisée RAM)")
    print(f"   Fichier: {yuv_file}")
    print(f"   Résolution: {width}x{height}")
    print(f"   Taille frame YUV: {yuv_frame_size} bytes (Y:{y_size}, UV:{uv_size}x2)")
    print(f"   Frames: {total_frames}")
    print(f"   FPS: {fps}")
    print()

    try:
        frame_count = 0
        frame_timestamps = []

        # Calculer le timestamp de départ
        start_time = datetime.datetime.utcnow()
        start_timestamp_us = int(start_time.timestamp() * 1000000)
        frame_interval_us = int(1000000 / fps)

        with open(ser_file, 'wb') as serf:
            # Écrire l'en-tête SER
            header = create_ser_header(width, height, 8, 100)  # RGB, 8-bit
            serf.write(header)

            # Ouvrir le fichier YUV et traiter frame par frame
            with open(yuv_file, 'rb') as yuvf:
                for i in range(total_frames):
                    # Lire frame YUV420
                    yuv_data = yuvf.read(yuv_frame_size)
                    if len(yuv_data) < yuv_frame_size:
                        print(f"   ⚠️  Frame {i+1} incomplète ({len(yuv_data)}/{yuv_frame_size} bytes)")
                        break

                    # Convertir YUV420 → RGB
                    rgb = yuv420_to_rgb(yuv_data, width, height)

                    if rgb is None:
                        print(f"   ❌ Erreur conversion frame {i+1}")
                        continue

                    # Calculer timestamp
                    frame_timestamp = start_timestamp_us + (i * frame_interval_us)
                    frame_timestamps.append(frame_timestamp)

                    # Écrire directement dans le SER
                    serf.write(rgb.tobytes())
                    frame_count += 1

                    # Afficher progression
                    if (i + 1) % 50 == 0 or i == 0:
                        print(f"   Conversion: {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.1f}%)")

            # Écrire les timestamps (SER v3)
            for timestamp in frame_timestamps:
                serf.write(struct.pack('<Q', timestamp))

        # Mettre à jour le nombre de frames dans l'en-tête
        update_ser_frame_count(ser_file, frame_count)

        output_size = os.path.getsize(ser_file)
        print()
        print(f"✓ Conversion terminée!")
        print(f"   Fichier sortie: {ser_file}")
        print(f"   Frames converties: {frame_count}")
        print(f"   Taille: {output_size:,} bytes ({output_size/(1024*1024):.2f} MB)")

        return True, frame_count, "Conversion réussie"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, 0, f"Erreur: {str(e)}"

def convert_rgb888_to_ser(rgb_file, ser_file, width, height, fps=25, bytes_per_pixel=3):
    """
    Convertit un fichier RGB888 ou XRGB8888 brut en SER avec timestamps
    Optimisé pour éviter la saturation RAM - traite les frames une par une

    Args:
        rgb_file: Fichier .rgb/.raw d'entrée
        ser_file: Fichier .ser de sortie
        width: Largeur
        height: Hauteur
        fps: Framerate pour les timestamps
        bytes_per_pixel: 3 pour RGB888, 4 pour XRGB8888

    Returns:
        (success, frame_count, message)
    """

    if not os.path.exists(rgb_file):
        return False, 0, f"Fichier RGB introuvable: {rgb_file}"

    # Calculer la taille d'une frame
    frame_size = width * height * bytes_per_pixel
    file_size = os.path.getsize(rgb_file)
    total_frames = file_size // frame_size

    if total_frames == 0:
        return False, 0, "Aucune frame RGB trouvée"

    format_name = "XRGB8888" if bytes_per_pixel == 4 else "RGB888"
    print(f"📹 Conversion {format_name} → SER (optimisée RAM)")
    print(f"   Fichier: {rgb_file}")
    print(f"   Résolution: {width}x{height}")
    print(f"   Taille frame: {frame_size} bytes ({bytes_per_pixel} bytes/pixel)")
    print(f"   Frames: {total_frames}")
    print(f"   FPS: {fps}")
    print()

    # Si XRGB8888, convertir frame par frame directement dans SER
    if bytes_per_pixel == 4:
        try:
            frame_count = 0
            frame_timestamps = []

            # Calculer le timestamp de départ
            start_time = datetime.datetime.utcnow()
            start_timestamp_us = int(start_time.timestamp() * 1000000)
            frame_interval_us = int(1000000 / fps)

            with open(ser_file, 'wb') as serf:
                # Écrire l'en-tête SER
                header = create_ser_header(width, height, 8, 100)  # RGB, 8-bit
                serf.write(header)

                # Ouvrir le fichier XRGB et traiter frame par frame
                with open(rgb_file, 'rb') as xrgbf:
                    for i in range(total_frames):
                        # Lire frame XRGB8888
                        xrgb_data = xrgbf.read(frame_size)
                        if len(xrgb_data) < frame_size:
                            print(f"   ⚠️  Frame {i+1} incomplète")
                            break

                        # Convertir XRGB8888 → RGB888 (enlever le canal X)
                        xrgb = np.frombuffer(xrgb_data, dtype=np.uint8).reshape(height, width, 4)
                        rgb = xrgb[:, :, :3]  # Prendre seulement RGB, ignorer X

                        # Calculer timestamp
                        frame_timestamp = start_timestamp_us + (i * frame_interval_us)
                        frame_timestamps.append(frame_timestamp)

                        # Écrire directement dans le SER
                        serf.write(rgb.tobytes())
                        frame_count += 1

                        # Afficher progression
                        if (i + 1) % 50 == 0 or i == 0:
                            print(f"   Conversion: {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.1f}%)")

                # Écrire les timestamps (SER v3)
                for timestamp in frame_timestamps:
                    serf.write(struct.pack('<Q', timestamp))

            # Mettre à jour le nombre de frames dans l'en-tête
            update_ser_frame_count(ser_file, frame_count)

            output_size = os.path.getsize(ser_file)
            print()
            print(f"✓ Conversion terminée!")
            print(f"   Fichier sortie: {ser_file}")
            print(f"   Frames converties: {frame_count}")
            print(f"   Taille: {output_size:,} bytes ({output_size/(1024*1024):.2f} MB)")

            return True, frame_count, "Conversion réussie"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, 0, f"Erreur conversion XRGB8888: {str(e)}"
    else:
        # RGB888 - inverser RGB→BGR pendant la conversion
        try:
            frame_count = 0
            frame_timestamps = []

            # Calculer le timestamp de départ
            start_time = datetime.datetime.utcnow()
            start_timestamp_us = int(start_time.timestamp() * 1000000)
            frame_interval_us = int(1000000 / fps)

            with open(ser_file, 'wb') as serf:
                # Écrire l'en-tête SER
                header = create_ser_header(width, height, 8, 100)  # RGB, 8-bit
                serf.write(header)

                # Ouvrir le fichier RGB et traiter frame par frame
                with open(rgb_file, 'rb') as rgbf:
                    for i in range(total_frames):
                        # Lire frame RGB888
                        rgb_data = rgbf.read(frame_size)
                        if len(rgb_data) < frame_size:
                            print(f"   ⚠️  Frame {i+1} incomplète")
                            break

                        # Inverser RGB → BGR (SER attend BGR)
                        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape(height, width, 3)
                        bgr = rgb[:, :, ::-1]  # Inverser les canaux

                        # Calculer timestamp
                        frame_timestamp = start_timestamp_us + (i * frame_interval_us)
                        frame_timestamps.append(frame_timestamp)

                        # Écrire dans le SER
                        serf.write(bgr.tobytes())
                        frame_count += 1

                        # Afficher progression
                        if (i + 1) % 50 == 0 or i == 0:
                            print(f"   Conversion: {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.1f}%)")

                # Écrire les timestamps (SER v3)
                for timestamp in frame_timestamps:
                    serf.write(struct.pack('<Q', timestamp))

            # Mettre à jour le nombre de frames dans l'en-tête
            update_ser_frame_count(ser_file, frame_count)

            output_size = os.path.getsize(ser_file)
            print()
            print(f"✓ Conversion terminée!")
            print(f"   Fichier sortie: {ser_file}")
            print(f"   Frames converties: {frame_count}")
            print(f"   Taille: {output_size:,} bytes ({output_size/(1024*1024):.2f} MB)")

            return True, frame_count, "Conversion réussie"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, 0, f"Erreur conversion RGB888: {str(e)}"

def auto_fix_bad_pixels_bayer(raw_bayer, sigma=5.0, min_threshold_adu=20.0, blend_factor=0.7):
    """
    Détection et correction automatique des pixels morts/chauds en respectant le pattern Bayer.
    VERSION AMÉLIORÉE POUR FAIBLE LUMINOSITÉ : Détection asymétrique + correction partielle.

    Améliorations par rapport à la version précédente :
    1. Détection ASYMÉTRIQUE : corrige uniquement les pixels CHAUDS (plus brillants que voisins)
       → Préserve les détails sombres et les étoiles faibles
    2. Seuil ADAPTATIF LOCAL par blocs (pas global)
       → Plus robuste aux variations de luminosité dans l'image
    3. Correction PARTIELLE (70% médiane + 30% original)
       → Moins destructif, préserve mieux les gradients fins
    4. Seuil MINIMUM ABSOLU pour ignorer le bruit faible
       → Évite de corriger le bruit naturel en faible luminosité

    Args:
        raw_bayer: Array RAW Bayer uint16 (height, width)
        sigma: Seuil de détection en nombre d'écarts-types (défaut=5.0)
               Plus élevé = moins agressif, moins de risque de lisser les détails
        min_threshold_adu: Seuil minimum absolu en ADU (défaut=20.0)
                           Ignore déviations < cette valeur pour préserver bruit faible
                           Mettre à 0 pour désactiver (plus agressif)
        blend_factor: Facteur de mélange pour correction (défaut=0.7)
                      0.7 = 70% médiane + 30% original (doux)
                      0.95 = 95% médiane + 5% original (agressif)

    Returns:
        Array RAW Bayer corrigé uint16
    """
    import cv2

    # IMPORTANT: Toujours traiter par canal Bayer pour éviter biais colorimétrique
    fixed = raw_bayer.astype(np.float32)

    # Traiter les 4 canaux Bayer séparément avec médiane Bayer-aware
    for start_y in range(2):
        for start_x in range(2):
            channel = fixed[start_y::2, start_x::2].copy()

            # Médiane locale 3x3 (8 voisins + pixel central)
            median_filtered = cv2.medianBlur(channel.astype(np.uint16), 3).astype(np.float32)

            # AMÉLIORATION 1 : Détection ASYMÉTRIQUE (seulement pixels CHAUDS)
            # Pixels chauds = plus brillants que médiane locale
            deviation_hot = channel - median_filtered  # Positif si pixel > médiane
            deviation_hot = np.maximum(deviation_hot, 0)  # Ignorer pixels plus sombres

            # AMÉLIORATION 2 : Seuil ADAPTATIF LOCAL par blocs 64×64
            # (Au lieu d'un seuil global qui échoue en faible luminosité)
            block_size = 64
            h, w = channel.shape
            threshold_map = np.zeros_like(channel)

            for by in range(0, h, block_size):
                for bx in range(0, w, block_size):
                    # Extraire bloc
                    y_end = min(by + block_size, h)
                    x_end = min(bx + block_size, w)
                    block_dev = deviation_hot[by:y_end, bx:x_end]

                    # MAD local du bloc
                    mad_local = np.median(block_dev[block_dev > 0]) if np.any(block_dev > 0) else 0

                    # Seuil local (1.4826 = conversion MAD → std)
                    threshold_local = sigma * mad_local * 1.4826

                    # Fallback sur percentile 95 si MAD trop faible
                    if mad_local < 1e-6:
                        threshold_local = sigma * np.percentile(block_dev, 95)

                    # AMÉLIORATION 4 : Seuil MINIMUM ABSOLU (ignorer bruit faible)
                    # Cela évite de corriger le bruit thermique naturel en faible luminosité
                    # Note: min_threshold_adu est passé en paramètre, peut être 0 pour désactiver
                    if min_threshold_adu > 0:
                        threshold_local = max(threshold_local, min_threshold_adu)

                    # Remplir la carte de seuils
                    threshold_map[by:y_end, bx:x_end] = threshold_local

            # Détecter pixels chauds (dépassent seuil local)
            is_hot_pixel = deviation_hot > threshold_map

            # Ignorer les bords (1 pixel de marge)
            is_hot_pixel[0, :] = False
            is_hot_pixel[-1, :] = False
            is_hot_pixel[:, 0] = False
            is_hot_pixel[:, -1] = False

            # AMÉLIORATION 3 : Correction PARTIELLE (blending configurable)
            # Au lieu de remplacer 100% par médiane, mélanger selon blend_factor
            # → Préserve les gradients fins et les détails faibles
            # blend_factor passé en paramètre (défaut 0.7 = 70% médiane, 30% original)
            channel[is_hot_pixel] = (
                blend_factor * median_filtered[is_hot_pixel] +
                (1 - blend_factor) * channel[is_hot_pixel]
            )

            fixed[start_y::2, start_x::2] = channel

    return np.clip(fixed, 0, 65535).astype(np.uint16)


def debayer_raw_array(raw_array, raw_format_str, red_gain=1.0, blue_gain=1.0, apply_denoise=True, swap_rb=False, fix_bad_pixels=False, sigma_threshold=5.0, min_adu_threshold=20.0):
    """
    Débayérise un array RAW Bayer (SRGGB12 ou SRGGB16) en RGB uint16 avec balance des blancs.

    IMPORTANT: Préserve la dynamique complète 12/16-bit en retournant du uint16

    Args:
        raw_array: Array numpy uint8 (height, stride_bytes) depuis capture_array("raw")
        raw_format_str: "SRGGB12" ou "SRGGB16" (utilisé pour info uniquement)
        red_gain: Gain du canal rouge (AWB, défaut=1.0)
        blue_gain: Gain du canal bleu (AWB, défaut=1.0)
        apply_denoise: Appliquer débruitage logiciel pour compenser l'absence d'ISP (défaut=True)
        swap_rb: Inverser les gains rouge et bleu (défaut=False)
        fix_bad_pixels: Activer la correction automatique des pixels morts (défaut=False)
        sigma_threshold: Seuil de détection des pixels morts en sigma (défaut=5.0)
        min_adu_threshold: Seuil minimum absolu en ADU (défaut=20.0, 0=désactivé)

    Returns:
        Array numpy uint16 (height, width, 3) RGB [0-65535]
    """
    try:
        # *** VALIDATION: Vérifier que l'array est bien 2D (format RAW Bayer) ***
        if len(raw_array.shape) != 2:
            print(f"[WARNING] debayer_raw_array: Array reçu n'est PAS du format RAW Bayer!")
            print(f"  Shape reçu: {raw_array.shape} (attendu: 2D)")
            print(f"  dtype: {raw_array.dtype}")
            if len(raw_array.shape) == 3:
                print(f"  → Array 3D détecté - probablement une image ISP du stream MAIN au lieu de RAW")
                # Retourner l'array tel quel s'il est déjà RGB/XRGB (pour éviter crash)
                if raw_array.shape[2] == 4:
                    # XRGB8888 - extraire les 3 premiers canaux (BGR)
                    print(f"  → Extraction BGR depuis XRGB (4 canaux)")
                    return (raw_array[:, :, :3].astype(np.float32) * 256).astype(np.uint16)
                elif raw_array.shape[2] == 3:
                    # RGB/BGR - convertir en uint16 pour compatibilité
                    print(f"  → Conversion RGB en uint16")
                    return (raw_array.astype(np.float32) * 256).astype(np.uint16)
            raise ValueError(f"Format RAW attendu (2D), reçu: {raw_array.shape}")

        height, dim2 = raw_array.shape

        # Détecter si les données sont déjà uint16 (unpacked) ou uint8 (packed)
        if raw_array.dtype == np.uint16:
            # Données déjà en uint16 (unpacked=True dans la config)
            # dim2 est la vraie largeur en pixels
            width = dim2
            raw_image = raw_array
        else:
            # Données en uint8 (packed) - conversion nécessaire
            # dim2 est le stride en bytes (2 bytes par pixel)
            raw_uint16 = raw_array.view(np.uint16)
            width = dim2 // 2
            raw_image = raw_uint16.reshape(height, -1)[:, :width]

        # *** CORRECTION PIXELS MORTS (automatique par sigma-clipping) ***
        if fix_bad_pixels:
            raw_image = auto_fix_bad_pixels_bayer(raw_image, sigma=sigma_threshold, min_threshold_adu=min_adu_threshold)

        # *** DÉBRUITAGE PRÉ-DEBAYER (optionnel) ***
        # DÉSACTIVÉ car cause écran noir pendant le stack (trop lent)
        # if apply_denoise:
        #     raw_image = cv2.fastNlMeansDenoising(raw_image, None, h=3, templateWindowSize=7, searchWindowSize=21)

        # Débayériser DIRECTEMENT en uint16 (préserve la dynamique complète)
        # OpenCV supporte le débayérisation sur uint16
        # Pattern pour IMX585 : RGGB (pattern original - BG inversait les couleurs)
        # Patterns possibles : BG=BGGR, GB=GBRG, RG=RGGB, GR=GRBG
        rgb_uint16 = cv2.cvtColor(raw_image, cv2.COLOR_BayerRG2RGB)

        # *** NOUVEAU : Appliquer la balance des blancs (AWB gains) ***
        # Le gain vert est normalisé à 1.0, donc on applique uniquement red_gain et blue_gain
        # Convertir en float32 pour appliquer les gains sans overflow
        rgb_float = rgb_uint16.astype(np.float32)

        # ATTENTION: pygame fait un swap R↔B pour l'affichage ([:,:,[2,1,0]])
        # Donc on applique red_gain sur canal 2 et blue_gain sur canal 0 pour que
        # l'effet corresponde visuellement au label du slider
        if swap_rb:
            # Mode swap_rb activé (rare) - inverser l'inversion
            rgb_float[:, :, 0] *= red_gain   # Canal 0 → Rouge à l'écran (après double swap)
            rgb_float[:, :, 2] *= blue_gain  # Canal 2 → Bleu à l'écran (après double swap)
        else:
            # Mode normal - INVERSÉ pour compenser le swap pygame [:,:,[2,1,0]]
            rgb_float[:, :, 2] *= red_gain   # Canal 2 natif → Rouge à l'écran
            rgb_float[:, :, 0] *= blue_gain  # Canal 0 natif → Bleu à l'écran
        # Canal Vert (index 1) : gain = 1.0, pas de modification

        # Cliper et reconvertir en uint16
        rgb_uint16 = np.clip(rgb_float, 0, 65535).astype(np.uint16)

        # Retourner en uint16 pour préserver la dynamique 12/16-bit
        # et être compatible avec libastrostack
        # Note: Pour SRGGB12, les valeurs sont déjà dans la bonne plage (0-65520)
        # car le capteur shifte de 4 bits à gauche.
        return rgb_uint16

    except Exception as e:
        print(f"[ERROR] Débayérisation échouée: {e}")
        import traceback
        traceback.print_exc()
        # Retourner une image noire en cas d'erreur (uint16 pour cohérence)
        # Dimensions approximatives si height/width non disponibles
        try:
            return np.zeros((height, width, 3), dtype=np.uint16)
        except:
            return np.zeros((1090, 1928, 3), dtype=np.uint16)


def calculate_snr(image):
    """
    Calcule le rapport signal/bruit (SNR) d'une image
    Retourne le SNR en ratio linéaire (signal/bruit)
    """
    try:
        # Convertir en niveaux de gris si l'image est en couleur
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculer le signal (moyenne)
        signal = np.mean(gray)
        
        # Calculer le bruit (écart-type)
        noise = np.std(gray)
        
        # Éviter la division par zéro
        if noise == 0:
            return 999.9
        
        # Calculer le SNR en ratio linéaire
        snr = signal / noise
        
        return snr
    except:
        return 0.0

def calculate_focus(gray_image, method):
    """
    Calcule le focus selon différentes méthodes

    Args:
        gray_image: image en niveaux de gris (numpy array)
        method: 0=OFF, 1=Laplacian, 2=Gradient, 3=Sobel, 4=Tenengrad

    Returns:
        valeur de focus (float) ou 0.0 si erreur/OFF
    """
    if method == 0:  # OFF
        return 0.0

    try:
        if method == 1:  # Laplacian variance
            return cv2.Laplacian(gray_image, cv2.CV_64F).var()

        elif method == 2:  # Gradient magnitude variance
            gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            return gradient_mag.var()

        elif method == 3:  # Sobel variance
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.abs(sobelx) + np.abs(sobely)
            return sobel.var()

        elif method == 4:  # Tenengrad (normalized)
            gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = gx**2 + gy**2
            # Pour cohérence avec les autres méthodes qui utilisent .var(),
            # on calcule la variance de sqrt(tenengrad) = variance de la magnitude
            return np.sqrt(tenengrad).var()

        else:
            return 0.0
    except:
        return 0.0

def calculate_hfr(image_surface, center_x, center_y, area_size):
    """
    Calcule le HFR (Half Flux Radius) - rayon contenant 50% du flux
    Plus robuste aux aigrettes que le FWHM
    Retourne le HFR en pixels
    """
    try:
        # Convertir la surface pygame en array
        image_array = pygame.surfarray.array3d(image_surface)
        
        # Extraire la région d'intérêt
        y1 = max(0, center_y - area_size)
        y2 = min(image_array.shape[1], center_y + area_size)
        x1 = max(0, center_x - area_size)
        x2 = min(image_array.shape[0], center_x + area_size)
        
        # Transposer pour obtenir (height, width, channels)
        roi = image_array[x1:x2, y1:y2, :]
        roi = np.transpose(roi, (1, 0, 2))
        
        # Convertir en niveaux de gris
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Trouver le centroïde (centre de masse)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        
        # Vérifier qu'il y a un contraste suffisant
        if max_val <= min_val or (max_val - min_val) / max_val < 0.1:
            return None
        
        # Soustraire le fond (minimum local)
        gray_sub = gray.astype(float) - min_val

        # CORRECTION : Utiliser un seuil pour ignorer le bruit
        # Ne considérer que les pixels avec au moins 30% de l'intensité maximale
        # Seuil plus élevé pour éviter les variations erratiques (2 à 60)
        threshold = (max_val - min_val) * 0.30

        # Créer un masque pour les pixels significatifs
        significant_mask = gray_sub >= threshold

        # Si pas assez de pixels significatifs, retourner None
        # Minimum de 10 pixels requis pour un calcul fiable
        if np.sum(significant_mask) < 10:
            return None

        # Calculer le flux total (uniquement les pixels significatifs)
        total_flux = np.sum(gray_sub[significant_mask])

        if total_flux <= 0:
            return None

        # Calculer le centroïde (uniquement sur les pixels significatifs)
        height, width = gray_sub.shape
        y_indices, x_indices = np.mgrid[0:height, 0:width]

        cx = np.sum(x_indices[significant_mask] * gray_sub[significant_mask]) / total_flux
        cy = np.sum(y_indices[significant_mask] * gray_sub[significant_mask]) / total_flux

        # Calculer la distance de chaque pixel significatif au centroïde
        distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)

        # Extraire seulement les pixels significatifs
        sig_distances = distances[significant_mask]
        sig_flux = gray_sub[significant_mask]

        # Trier les pixels par distance croissante
        sorted_indices = np.argsort(sig_distances)
        sorted_distances = sig_distances[sorted_indices]
        sorted_flux = sig_flux[sorted_indices]

        # Calculer le flux cumulé
        cumulative_flux = np.cumsum(sorted_flux)
        half_flux = total_flux / 2.0

        # Trouver le rayon contenant 50% du flux
        idx = np.searchsorted(cumulative_flux, half_flux)
        
        if idx >= len(sorted_distances):
            return None
        
        hfr = sorted_distances[idx]
        
        return float(hfr)
        
    except Exception as e:
        return None

version = 1.07

# streaming parameters
stream_type = 2             # 0 = TCP, 1 = UDP, 2 = RTSP
stream_port = 5000          # set video streaming port number
udp_ip_addr = "10.42.0.52"  # IP address of the client for UDP streaming

# Set displayed preview image size (must be less than screen size to allow for the menu!!)
# Optomised for Pi 7" v1 screen

preview_width  = 880
preview_height = 580
fullscreen     = 0   # set to 1 for FULLSCREEN
frame          = 0   # set to 0 for NO frame 
FUP            = 21  # Pi v3 camera Focus UP GPIO button
FDN            = 16  # Pi v3 camera Focus DN GPIO button
sw_ir          = 26  # Waveshare IR Filter switch GPIO
STR            = 12  # external GPIO trigger for capture

# set default values (see limits below)
camera      = 0    # choose camera to use, usually 0 unless using a Pi5 or multiswitcher
mode        = 1    # set camera mode ['manual','normal','sport'] 
speed       = 16   # position in shutters list (16 = 1/125th)
custom_sspeed = 0  # valeur d'exposition personnalisée en microsecondes (0 = utiliser shutters[speed])
gain        = 0    # set gain , 0 = AUTO
brightness  = 0    # set camera brightness
contrast    = 70   # set camera contrast 
ev          = 0    # eV correction 
blue        = 12   # blue balance 
red         = 15   # red balance 
extn        = 0    # still file type  (0 = jpg), see extns below
vlen        = 10   # video length in seconds
fps         = 100  # video fps - Optimisé pour IMX585 (max 178 fps en 800x600)
vformat     = 10   # set video format (10 = 1920x1080), see vwidths & vheights below
codec       = 0    # set video codec  (0 = h264), see codecs below
ser_format  = 1    # set SER capture format (0 = YUV420, 1 = RGB888, 2 = XRGB8888), see ser_formats below
flicker     = 0    # anti-flicker mode (0=OFF, 1=50Hz, 2=60Hz, 3=AUTO)
tinterval   = 5.0   # time between timelapse shots in seconds
tshots      = 10   # number of timelapse shots
saturation  = 10   # picture colour saturation
meter       = 2    # metering mode (2 = average), see meters below
awb         = 1    # auto white balance mode, off, auto etc (1 = auto), see awbs below
sharpness   = 15   # set sharpness level
denoise     = 1    # set denoise level, see denoises below
fix_bad_pixels = 0  # auto-fix bad/hot pixels in RAW mode (0=off, 1=on)
fix_bad_pixels_sigma = 40  # threshold for bad pixel detection (×10, 50=5.0 sigma)
fix_bad_pixels_min_adu = 100  # seuil minimum absolu en ADU (×10, 200=20 ADU, 0=désactivé)
quality     = 93   # set quality level
profile     = 0    # set h264 profile, see h264profiles below
level       = 0    # set h264 level
histogram   = 5    # OFF = 0, 1 = red, 2 = green, 3 = blue, 4 = luminance, 5 = ALL
histarea    = 50   # set histogram area size
v3_f_mode   = 0    # v3 focus mode,  see v3_f_modes below
v3_f_range  = 0    # v3 focus range, see v3_f_ranges below
v3_f_speed  = 0    # v3 focus speed, see v3_f_speeds below
IRF         = 0    # Waveshare imx290-83 IR filter, 1 = ON
str_cap     = 0    # 0 = STILL, see strs below
v3_hdr      = 0    # HDR (v3 camera or Pi5 ONLY), see v3_hdrs below
timet       = 100  # -t setting when capturing STILLS (délai stabilisation AE/AWB)
vflip       = 0    # set to 1 to vertically flip images
hflip       = 0    # set tp 1 tp horizontally flip images
# NOTE if you change any of the above defaults you need to delete the con_file and restart.

# default directories and files
pic         = "Pictures"
vid         = "Videos"
con_file    = "PiLCConfig104.txt"

# setup directories
Home_Files  = []
Home_Files.append(os.getlogin())
pic_dir     = "/home/" + Home_Files[0] + "/images/"
vid_dir     = "/home/" + Home_Files[0]+ "/" + vid + "/"
# Lire le fichier de config dans le même répertoire que le programme
script_dir  = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, con_file)

# inital parameters
prev_fps    = 60  # Optimisé pour IMX585 (haute performance)
focus_fps   = 60  # Optimisé pour IMX585 (haute performance)
focus       = 700
foc_man     = 0
focus_mode  = 0
v3_focus    = 480
v3_hdr      = 0
vpreview    = 1
scientific  = 0
scientif    = 0
zx          = int(preview_width/2)
zy          = int(preview_height/2)
fxz         = 1
zoom        = 0
igw         = 2592
igh         = 1944
zwidth      = igw 
zheight     = igh
buttonFUP   = Button(FUP)
buttonFDN   = Button(FDN)
buttonSTR   = Button(STR)
led_sw_ir   = LED(sw_ir)
str_btn     = 0
lo_res      = 1
show_cmds   = 1  # Debug: afficher les commandes
v3_af       = 1
v5_af       = 1
menu        = 0
menu_page   = {6: 1}  # Suivi des pages pour les menus multi-pages (menu 6: TIMELAPSE, page 1 par défaut)
alt_dis     = 0
rotate      = 0
still       = 0
video       = 0
timelapse   = 0
stream      = 0
stretch_mode = 0  # Mode stretch astro pour le preview
stretch_adjust_mode = 0  # Mode ajustement des paramètres stretch en plein écran (0=off, 1=on)
_stretch_slider_rects = {}  # Rectangles des sliders stretch pour détection clics
stretch_p_low = 0    # Percentile bas pour stretch (0% à 0.2%, stocké x10 pour slider)
stretch_p_high = 9998 # Percentile haut pour stretch (99.95% à 100%, stocké x100 pour slider)
stretch_factor = 25 # Facteur de stretch (0 à 8, stocké x10 pour slider)
stretch_preset = 0   # Préréglage stretch: 0=OFF, 1=GHS, 2=Arcsinh
# GHS Parameters (conforme Siril/PixInsight) - Phase 2
ghs_D = 31     # Stretch factor: -10 à 100 -> -1.0 à 10.0 (défaut 3.1 optimisé tests)
ghs_b = 1      # Local intensity: -50 à 150 -> -5.0 à 15.0 (défaut 0.1 optimisé tests)
ghs_SP = 19    # Symmetry point: 0-100 -> 0.0-1.0 (défaut 0.19 optimisé tests)
ghs_LP = 0     # Protect shadows: 0-100 -> 0.0-1.0 (défaut 0.0)
ghs_HP = 0     # Protect highlights: 0-100 -> 0.0-1.0 (défaut 0.0 optimisé tests)
ghs_preset = 1 # 0=Manual, 1=Galaxies, 2=Nébuleuses, 3=Étirement initial
ghs_presets = ['Manual', 'Galaxies', 'Nebulae', 'Initial']
# METRICS Settings - Phase 2 Demande 5 et 6
focus_method = 1        # 0=OFF, 1=Laplacian, 2=Gradient, 3=Sobel, 4=Tenengrad
star_metric = 1         # 0=OFF, 1=HFR, 2=FWHM
snr_display = 0         # 0=OFF, 1=ON
metrics_interval = 3    # 1-10 frames (calcul métriques tous les N frames)
focus_methods = ['OFF', 'Laplacian', 'Gradient', 'Sobel', 'Tenengrad']
star_metrics = ['OFF', 'HFR', 'FWHM']
metrics_frame_counter = 0  # Compteur pour calcul métriques tous les N frames
fwhm_history = deque(maxlen=240)
fwhm_times = deque(maxlen=240)
fwhm_start_time = 0
fwhm_fig = None
fwhm_ax = None
hfr_history = deque(maxlen=240)
hfr_times = deque(maxlen=240)
hfr_start_time = 0
hfr_fig = None
hfr_ax = None
focus_history = deque(maxlen=240)
focus_times = deque(maxlen=240)
focus_start_time = 0
focus_fig = None
focus_ax = None
# Optimisation matplotlib - Phase 2 Demande 7
_focus_frame_counter = 0
_hfr_fwhm_frame_counter = 0
_graphs_update_interval = 2  # Mettre à jour 1 frame sur 2 (réduction mémoire ~80%)
p = None  # Subprocess rpicam-vid (None en mode Picamera2)

# Picamera2 variables
picam2 = None  # Instance Picamera2
capture_thread = None  # Thread de capture asynchrone
use_picamera2 = True  # Flag pour activer Picamera2 (False = utiliser rpicam-vid)
use_native_sensor_mode = 0  # 0=binning rapide (1080p), 1=résolution native complète du capteur

# Live Stack variables
livestack = None  # Instance RPiCameraLiveStack
livestack_active = False  # Mode Live Stack actif
luckystack = None  # Instance RPiCameraLiveStackAdvanced (mode Lucky)
luckystack_active = False  # Mode Lucky Stack actif

# Live Stack parameters
ls_preview_refresh = 5  # Rafraîchir preview toutes les N images (1-10)
ls_alignment_mode = 2  # 0=OFF/none, 1=translation, 2=rotation, 3=affine
ls_alignment_modes = ["OFF", "translation", "rotation", "affine"]
ls_enable_qc = 0  # Contrôle qualité désactivé par défaut (0=OFF, 1=ON)
ls_max_fwhm = 170  # FWHM max x10 (0=OFF, 100-250 = 10.0-25.0)
ls_min_sharpness = 70  # Netteté min x1000 (0=OFF, 30-150 = 0.030-0.150)
ls_max_drift = 2500  # Dérive max en pixels (0=OFF, 500-5000)
ls_min_stars = 10  # Nombre min d'étoiles (0=OFF, 1-20)

# Stacker Advanced parameters
ls_stack_method = 0  # 0=mean, 1=median, 2=kappa_sigma, 3=winsorized, 4=weighted
ls_stack_kappa = 25  # Kappa x10 (valeur réelle: 2.5)
ls_stack_iterations = 3  # Itérations sigma-clip
stack_methods = ['Mean', 'Median', 'Kappa-Sigma', 'Winsorized', 'Weighted']

# Planetary Alignment parameters
ls_planetary_enable = 0  # 0=off, 1=on
ls_planetary_mode = 1  # 0=disk, 1=surface, 2=hybrid
ls_planetary_disk_min = 50  # Rayon min (pixels)
ls_planetary_disk_max = 500  # Rayon max (pixels)
ls_planetary_threshold = 30  # Seuil Canny
ls_planetary_margin = 10  # Marge disque (pixels)
ls_planetary_ellipse = 0  # 0=cercle, 1=ellipse
ls_planetary_window = 1  # 0=128, 1=256, 2=512
ls_planetary_upsample = 10  # Précision sub-pixel
ls_planetary_highpass = 1  # Filtre passe-haut
ls_planetary_roi_center = 1  # ROI au centre
ls_planetary_corr = 30  # Corrélation min x100 (valeur réelle: 0.30)
ls_planetary_max_shift = 100  # Décalage max
planetary_modes = ['Disk', 'Surface', 'Hybrid']
planetary_windows = [128, 256, 512]

# Lucky Imaging parameters
ls_lucky_buffer = 10  # Taille buffer (10-200)
ls_lucky_keep = 10  # % à garder (1-50)
ls_lucky_score = 0  # 0=laplacian, 1=gradient, 2=sobel, 3=tenengrad
ls_lucky_stack = 0  # 0=mean, 1=median, 2=sigma_clip
ls_lucky_align = 1  # 0=off, 1=on
ls_lucky_roi = 50  # % ROI scoring (20-100)
ls_lucky_save_progress = 0  # 0=off, 1=on (sauvegarde FITS+PNG tous les 2 stacks)
lucky_score_methods = ['Laplacian', 'Gradient', 'Sobel', 'Tenengrad']
lucky_stack_methods = ['Mean', 'Median', 'Sigma-Clip']

# RAW Format parameters (pour Lucky/Live Stack et vidéo RAW)
raw_format = 1  # 0=YUV420, 1=XRGB8888, 2=SRGGB12, 3=SRGGB16
raw_formats = ['YUV420 8bit', 'XRGB8888 ISP', 'RAW12 Bayer', 'RAW16 Clear HDR']
raw_swap_rb = 0  # Inverser rouge et bleu en mode RAW (0=non, 1=oui)
raw_stream_size = (1928, 1090)  # Taille du flux RAW (sera calculée dans preview())
capture_size = (1928, 1090)  # Taille de capture/ROI (sera calculée dans preview())

# ISP (Image Signal Processor) parameters
isp_enable = 0  # 0=off, 1=on (utilise isp_config_imx585.json)

# RAW Adjustment Panel (mode plein écran)
raw_adjust_mode = 0           # 0=off, 1=on
raw_adjust_tab = 0            # 0=ISP, 1=Stretch
_raw_slider_rects = {}        # Rectangles pour clics sliders RAW

# Paramètres ISP GUI (stockés x100 pour sliders, sauf black_level)
isp_wb_red = 100              # 0.5-2.0 → 50-200 (défaut 1.0)
isp_wb_green = 100            # 0.5-2.0 → 50-200 (défaut 1.0)
isp_wb_blue = 100             # 0.5-2.0 → 50-200 (défaut 1.0)
isp_gamma = 100               # 0.5-3.0 → 50-300 (défaut 1.0, correspond à gamma 2.2)
isp_black_level = 64          # 0-500 direct (défaut 64)
isp_brightness = 0            # -0.5-0.5 → -50-50 (défaut 0)
isp_contrast = 100            # 0.5-2.0 → 50-200 (défaut 1.0)
isp_saturation = 100          # 0.0-2.0 → 0-200 (défaut 1.0)
isp_sharpening = 0            # 0.0-2.0 → 0-200 (défaut 0)

# set button sizes
bw = int(preview_width/5.66)
bh = int(preview_height/10)
ft = int(preview_width/46)
fv = int(preview_width/46)

if tinterval > 0:
    tduration  = tshots * tinterval
else:
    tduration = 5

dis_height = preview_height
dis_width  = preview_width
    
# data
cameras      = [  '', 'Pi v1', 'Pi v2', 'Pi v3', 'Pi HQ','Ard 16MP','Hawkeye', 'Pi GS','Owlsight',"imx290",'imx585','imx293','imx294','imx283','imx500','ov9281']
camids       = [  '','ov5647','imx219','imx708','imx477',  'imx519', 'arduca','imx296',  'ov64a4','imx290','imx585','imx293','imx294','imx283','imx500','ov9281']
x_sens       = [   0,    2592,    3280,    4608,    4056,      4656,     9152,    1456,      9248,    1920,    3856,    3856,    4168,    5472,    4056,    1280]
y_sens       = [   0,    1944,    2464,    2592,    3040,      3496,     6944,    1088,      6944,    1080,    2180,    2180,    2824,    3648,    3040,     800]
max_gains    = [  64,     255,      40,      64,      88,        64,       64,      64,        64,      64,    3000,      64,      64,      64,      64,      64]  # IMX585: max 3000 avec courbe non-linéaire
max_shutters = [ 100,       1,      11,     112,     650,       200,      435,      15,       435,     100,     163,     100,     100,     100,     100,     100]
max_vfs      = [  10,      15,      16,      21,      20,        15,       22,       7,        22,      10,      18,      18,      18,      23,      20,       3]
modes        = ['manual','normal','sport']
extns        = ['jpg','png','bmp','rgb','yuv420','raw']
extns2       = ['jpg','png','bmp','data','data','dng']
vwidths      = [640,720,800,1280,1280,1296,1332,1456,1536,1640,1920,1928,2028,2028,2304,2880,2592,3280,3840,3856,4032,4056,4608,4656,5472,8000,9152,9248]
vheights     = [480,540,600, 720, 960, 972, 990,1088, 864,1232,1080,1090,1080,1520,1296,2160,1944,2464,2160,2180,3024,3040,2592,3496,3648,6000,6944,6944]
v_max_fps    = [200,120, 40,  40,  40,  30,  60,  30,  30,  30,  30,  30,  50,  40,  25,  40,  20,  20,  20,  20,  20,  10,  20,  20,  20,  20,  20,  20]
v3_max_fps   = [200,120,125, 120, 120, 120, 120, 120, 120, 100, 100,  50, 100,  56,  56,  40,  20,  20,  20,  20,  20,  15,  20,  20,  20  ,20,  20,  20]
v9_max_fps   = [240, 200, 150, 120, 100, 100, 80, 60, 60, 60, 60]
v10_max_fps  = [178,150,178, 150, 150, 150, 150, 150, 150, 150, 100, 50, 100,  60,  56,  51,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43,  43]  # IMX585 modes réels (178fps@800x600, 150fps@720p, 100fps@1080p, 51fps@2.8K, 43fps@4K)
v15_max_fps  = [240,200,200, 130]
zwidths      = [640,800,1280,2592,3280,4056,4656,9152]
zheights     = [480,600, 960,1944,2464,3040,3496,6944]
zfs          = [1, 0.5, 0.333333, 0.25, 0.2, 0.166666]  # Zoom: 1x, 2x, 3x, 4x, 5x, 6x

# NOUVEAU: Mapping zoom → modes IMX585 hardware crop
# Pour IMX585 uniquement (Pi_Cam == 10)
# ORDRE DÉCROISSANT: résolutions de la plus grande à la plus petite
imx585_crop_modes = {
    # zoom_level: (width, height, mode_name, max_fps_estimate)
    0: None,  # Full frame - déterminé par use_native_sensor_mode
    1: (2880, 2160, "Mode 2 Crop", 40),   # Hardware 2.8K crop (résolution la plus haute)
    2: (1920, 1080, "Mode 3 Crop", 90),   # Hardware 1080p crop
    3: (1280, 720, "Mode 4 Crop", 120),   # Hardware 720p crop
    4: (800, 600, "Mode 5 Crop", 150),    # Hardware 800x600 crop
    5: (800, 600, "Mode 5 Crop", 150)     # Identique à zoom 4
}

# Résolutions RAW validées pour IMX585 (testées et fonctionnelles)
# Format: (width, height): "description"
imx585_validated_raw_modes = {
    (3856, 2180): "Full Native 4K (avec unpacked=True)",
    (1928, 1090): "Binning 2x2",
    (1920, 1080): "FHD Crop (Mode 3)",
    # Autres modes à valider par tests:
    # (2880, 2160): "2.8K Crop (Mode 2)",
    # (1280, 720): "HD Crop (Mode 4)",
    # (800, 600): "SVGA Crop (Mode 5)",
}

# Labels de résolution pour le slider de zoom (ordre décroissant)
zoom_res_labels = {
    0: "",
    1: "2880x2160",  # 2x zoom - Résolution la plus haute
    2: "1920x1080",  # 3x zoom
    3: "1280x720",   # 4x zoom
    4: "800x600",    # 5x zoom
    5: "800x600"     # 6x zoom
}

# FPS optimaux pour chaque niveau de zoom (ROI permet des FPS plus élevés)
# Format: {zoom_level: (fps_standard, fps_v3, fps_v9, fps_imx585)}
zoom_optimal_fps = {
    1: (30, 50, 60, 51),     # 2880x2160 - Mode 2 hardware (IMX585: 51.14 fps)
    2: (60, 100, 120, 100),  # 1920x1080 - Mode 3 hardware (IMX585: 100 fps)
    3: (120, 120, 150, 150), # 1280x720  - Mode 4 hardware (IMX585: 150.02 fps)
    4: (200, 200, 240, 178), # 800x600   - Mode 5 hardware (IMX585: 178.57 fps)
    5: (200, 200, 240, 178)  # 800x600   - Mode 5 hardware (IMX585: 178.57 fps)
}

def sync_video_resolution_with_zoom():
    """
    Synchronise vwidth, vheight, vformat ET fps avec la résolution du zoom actuel.
    Appelée automatiquement quand le zoom change pour éviter les incohérences.
    Optimise aussi les FPS pour tirer parti des résolutions ROI plus petites.
    """
    global zoom, vwidth, vheight, vformat, vwidths, vheights, fps, video_limits, Pi_Cam

    # Résolutions correspondant aux niveaux de zoom
    if Pi_Cam == 10:  # IMX585 - Utiliser modes hardware crop (ordre décroissant)
        zoom_resolutions = {
            1: (2880, 2160),  # Mode 2 crop (résolution la plus haute)
            2: (1920, 1080),  # Mode 3 crop
            3: (1280, 720),   # Mode 4 crop
            4: (800, 600),    # Mode 5 crop
            5: (800, 600)     # Mode 5 crop
        }
    else:
        # Autres caméras: résolutions standards supportées (ordre décroissant)
        zoom_resolutions = {
            1: (2880, 2160),  # 2x zoom (résolution la plus haute si supportée)
            2: (1920, 1080),  # 3x zoom
            3: (1280, 720),   # 4x zoom
            4: (800, 600),    # 5x zoom
            5: (800, 600)     # 6x zoom (même résolution que zoom 4)
        }

    # Si zoom actif, synchroniser avec la résolution du zoom
    if zoom > 0 and zoom in zoom_resolutions:
        target_width, target_height = zoom_resolutions[zoom]

        # Chercher l'index correspondant dans vwidths/vheights
        vformat_found = False
        for i in range(len(vwidths)):
            if vwidths[i] == target_width and vheights[i] == target_height:
                vformat = i
                vwidth = target_width
                vheight = target_height
                vformat_found = True
                break

        # Si pas trouvé dans la liste, mettre à jour directement vwidth/vheight
        if not vformat_found:
            vwidth = target_width
            vheight = target_height
            # vformat reste inchangé

        # Optimiser les FPS pour le zoom (ROI permet des FPS plus élevés)
        if zoom in zoom_optimal_fps:
            fps_standard, fps_v3, fps_v9, fps_imx585 = zoom_optimal_fps[zoom]

            # Choisir les FPS selon le type de caméra
            if Pi_Cam == 3:
                optimal_fps = fps_v3
            elif Pi_Cam == 9:
                optimal_fps = fps_v9
            elif Pi_Cam == 10:  # IMX585
                optimal_fps = fps_imx585
            elif Pi_Cam == 15:
                optimal_fps = fps_v9  # OV9281 utilise les mêmes que IMX290
            else:
                optimal_fps = fps_standard

            # Sauvegarder les FPS d'origine si première activation du zoom
            if not hasattr(sync_video_resolution_with_zoom, 'fps_backup'):
                sync_video_resolution_with_zoom.fps_backup = fps
                sync_video_resolution_with_zoom.vfps_backup = video_limits[5]

            # Mettre à jour fps et la limite max pour profiter du ROI
            # IMPORTANT: Ces variables sont déclarées global en haut de la fonction
            globals()['fps'] = optimal_fps  # Utiliser les FPS optimaux
            video_limits[5] = optimal_fps  # Nouvelle limite max
    else:
        # Zoom désactivé : restaurer les FPS d'origine si sauvegardés
        if hasattr(sync_video_resolution_with_zoom, 'fps_backup'):
            globals()['fps'] = sync_video_resolution_with_zoom.fps_backup
            video_limits[5] = sync_video_resolution_with_zoom.vfps_backup
            # Nettoyer les sauvegardes
            delattr(sync_video_resolution_with_zoom, 'fps_backup')
            delattr(sync_video_resolution_with_zoom, 'vfps_backup')

def get_imx585_sensor_mode(zoom_level, use_native=False):
    """
    Retourne la résolution du mode sensor IMX585 pour un niveau de zoom donné.

    Args:
        zoom_level: 0-5 (niveau de zoom actuel)
        use_native: Si True et zoom=0, utilise mode natif 4K au lieu de binning

    Returns:
        (width, height) du mode sensor à utiliser, ou None si pas IMX585

    Usage:
        sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode)
        if sensor_mode:
            config = picam2.create_preview_configuration(
                main={"size": output_size},
                raw={"size": sensor_mode}
            )
    """
    if Pi_Cam != 10:
        # Pas IMX585, retourner None (utiliser logique existante)
        return None

    # Zoom 0 (full frame): dépend de use_native_sensor_mode
    if zoom_level == 0:
        if use_native:
            return (3856, 2180)  # Mode 1: Native 4K
        else:
            return (1928, 1090)  # Mode 0: Binning 2x2

    # Zoom 1-5: utiliser les modes crop hardware
    if zoom_level in imx585_crop_modes:
        mode_info = imx585_crop_modes[zoom_level]
        if mode_info:
            return (mode_info[0], mode_info[1])

    # Fallback: mode binning
    return (1928, 1090)

shutters     = [-4000,-2000,-1600,-1250,-1000,-800,-640,-500,-400,-320,-288,-250,-240,-200,-160,-144,-125,-120,-100,-96,-80,-60,-50,-48,-40,-30,-25,
                -20,-15,-13,-10,-8,-6,-5,-4,-3,0.4,0.5,0.6,0.8,1,1.1,1.2,2,3,4,5,6,7,8,9,10,11,15,20,25,30,40,50,60,75,100,112,120,150,200,220,230,
                239,435,500,600,650,660,670]
codecs       = ['h264','mjpeg','yuv420','raw','ser_yuv','ser_rgb','ser_xrgb']
codecs2      = ['h264','mjpeg','data','raw','SER-YUV','SER-RGB','SER-XRGB']
ser_formats  = ['YUV420', 'RGB888', 'XRGB8888']
h264profiles = ['baseline 4','baseline 4.1','baseline 4.2','main 4','main 4.1','main 4.2','high 4','high 4.1','high 4.2']
meters       = ['centre','spot','average']
awbs         = ['off','auto','incandescent','tungsten','fluorescent','indoor','daylight','cloudy']
denoises     = ['off','cdn_off','cdn_fast','cdn_hq']
v3_f_modes   = ['auto','manual','continuous']
v3_f_ranges  = ['normal','macro','full']
v3_f_speeds  = ['normal','fast']
histograms   = ["OFF","Red","Green","Blue","Lum","ALL"]
strs         = ["Still","Video","Stream","Timelapse"]
stretch_presets = ['OFF','GHS','Arcsinh']
v3_hdrs      = ["OFF","Single-Exp","Multi-Exp","Night","Clear HDR 16bit"]
# Mapping pour rpicam-still/rpicam-vid (valeurs CLI attendues)
# Note: Night n'existe pas en CLI → fallback vers "off"
v3_hdrs_cli  = ["off","single-exp","auto","off","sensor"]  # off, single-exp, multi-exp→auto, night→off, sensor

#check linux version.
if os.path.exists ("/run/shm/lv.txt"): 
    os.remove("/run/shm/lv.txt")
os.system("cat /etc/os-release >> /run/shm/lv.txt")
with open("/run/shm/lv.txt", "r") as file:
    line = file.readline()
    while line:
       line = file.readline()
       if line[0:16] == "VERSION_CODENAME":
           lver = line
lvers = lver.split("=")
lver = lvers[1][0:6]
print(lver)

#check Pi model.
Pi = -1
if os.path.exists ('/run/shm/md.txt'): 
    os.remove("/run/shm/md.txt")
os.system("cat /proc/cpuinfo >> /run/shm/md.txt")
with open("/run/shm/md.txt", "r") as file:
    line = file.readline()
    while line:
       line = file.readline()
       if line[0:5] == "Model":
           model = line
mod = model.split(" ")
if mod[3] == "Compute":
    Pi = int(mod[5][0:1])
elif mod[3] == "Zero":
    Pi = 0
else:
    Pi = int(mod[3])
print("Pi:",Pi)
# Note : MP4 sur Pi 5 a des métadonnées de framerate incorrectes (bug rpicam-vid)
# mais reste utilisable car les frames sont encodées correctement
if Pi == 5:
    codecs.append('mp4')
    codecs2.append('mp4')

# ALLSKY TIMELAPSE PARAMETERS - Mode vidéo timelapse style Allsky
allsky_mode = 0              # 0=OFF, 1=ON (gain fixe), 2=Auto-Gain
allsky_mean_target = 30      # ×100 → 0.30 (cible de luminosité pour auto-gain)
allsky_mean_threshold = 5    # ×100 → 0.05 (tolérance autour de la cible)
allsky_video_fps = 25        # FPS pour la vidéo finale assemblée (15-60)
allsky_max_gain = 200        # Gain maximum autorisé en mode Auto-Gain (50-500)
allsky_apply_stretch = 1     # 0=OFF, 1=ON (appliquer stretch sur chaque frame)
allsky_cleanup_jpegs = 0     # 0=garder JPEG, 1=supprimer après assemblage vidéo
allsky_modes = ['OFF', 'ON', 'Auto-Gain']

# Options de sauvegarde LiveStack/LuckyStack
ls_save_progress = 1         # 0=OFF, 1=ON (sauvegarder PNG intermédiaires LiveStack toutes les frames)
ls_save_final = 1            # 0=OFF, 1=ON (sauvegarder PNG/FITS final LiveStack)
ls_lucky_save_final = 1      # 0=OFF, 1=ON (sauvegarder PNG/FITS final LuckyStack)

still_limits = ['mode',0,len(modes)-1,'speed',0,len(shutters)-1,'gain',0,30,'brightness',-100,100,'contrast',0,200,'ev',-10,10,'blue',1,80,'sharpness',0,30,
                'denoise',0,len(denoises)-1,'quality',0,100,'red',1,80,'extn',0,len(extns)-1,'saturation',0,20,'meter',0,len(meters)-1,'awb',0,len(awbs)-1,
                'histogram',0,len(histograms)-1,'v3_f_speed',0,len(v3_f_speeds)-1,'v3_hdr',0,len(v3_hdrs)-1,'focus_method',0,4,'star_metric',0,2,'snr_display',0,1,'metrics_interval',1,10]
video_limits = ['vlen',0,3600,'fps',1,180,'v5_focus',10,2500,'vformat',0,7,'0',0,0,'zoom',0,5,'Focus',0,1,'tduration',1,86400,'tinterval',0.01,10,'tshots',1,999,
                'flicker',0,3,'codec',0,len(codecs)-1,'ser_format',0,len(ser_formats)-1,'profile',0,len(h264profiles)-1,'v3_focus',10,2000,'histarea',10,300,'v3_f_range',0,len(v3_f_ranges)-1,
                'str_cap',0,len(strs)-1,'v6_focus',10,1020,'stretch_p_low',0,2,'stretch_p_high',9995,10000,'stretch_factor',0,80,'stretch_preset',0,2,
                'ghs_D',-10,100,'ghs_b',-300,100,'ghs_SP',0,100,'ghs_LP',0,100,'ghs_HP',0,100,'ghs_preset',0,3,'use_native_sensor_mode',0,1,
                'raw_format',0,3,'focus_method',0,4,'star_metric',0,2,'snr_display',0,1,'metrics_interval',1,10,
                'allsky_mode',0,2,'allsky_mean_target',10,60,'allsky_mean_threshold',2,15,'allsky_video_fps',15,60,
                'allsky_max_gain',50,500,'allsky_apply_stretch',0,1,'allsky_cleanup_jpegs',0,1]

livestack_limits = [
    # Existants
    'ls_preview_refresh',1,10,
    'ls_alignment_mode',0,len(ls_alignment_modes)-1,
    'ls_enable_qc',0,1,
    'ls_max_fwhm',0,250,
    'ls_min_sharpness',0,150,
    'ls_max_drift',0,5000,
    'ls_min_stars',0,20,
    # Stacker Advanced
    'ls_stack_method',0,4,
    'ls_stack_kappa',10,40,
    'ls_stack_iterations',1,10,
    # Planetary
    'ls_planetary_enable',0,1,
    'ls_planetary_mode',0,2,
    'ls_planetary_disk_min',20,500,
    'ls_planetary_disk_max',100,2000,
    'ls_planetary_threshold',10,100,
    'ls_planetary_margin',5,50,
    'ls_planetary_ellipse',0,1,
    'ls_planetary_window',0,2,
    'ls_planetary_upsample',1,20,
    'ls_planetary_highpass',0,1,
    'ls_planetary_roi_center',0,1,
    'ls_planetary_corr',10,90,
    'ls_planetary_max_shift',10,200,
    # Lucky Imaging
    'ls_lucky_buffer',10,200,
    'ls_lucky_keep',1,50,
    'ls_lucky_score',0,3,
    'ls_lucky_stack',0,2,
    'ls_lucky_align',0,1,
    'ls_lucky_roi',20,100
]

# check config_file exists, if not then write default values
titles = ['mode','speed','gain','brightness','contrast','frame','red','blue','ev','vlen','fps','vformat','codec','tinterval','tshots','extn','zx','zy','zoom','saturation',
          'meter','awb','sharpness','denoise','quality','profile','level','histogram','histarea','v3_f_speed','v3_f_range','rotate','IRF','str_cap','v3_hdr','raw_format','vflip','hflip',
          'stretch_p_low','stretch_p_high','stretch_factor','stretch_preset','ghs_D','ghs_b','ghs_SP','ghs_LP','ghs_HP','ghs_preset',
          'ls_preview_refresh','ls_alignment_mode','ls_enable_qc','ls_max_fwhm','ls_min_sharpness','ls_max_drift','ls_min_stars',
          'ls_stack_method','ls_stack_kappa','ls_stack_iterations',
          'ls_planetary_enable','ls_planetary_mode','ls_planetary_disk_min','ls_planetary_disk_max','ls_planetary_threshold','ls_planetary_margin','ls_planetary_ellipse','ls_planetary_window','ls_planetary_upsample','ls_planetary_highpass','ls_planetary_roi_center','ls_planetary_corr','ls_planetary_max_shift',
          'ls_lucky_buffer','ls_lucky_keep','ls_lucky_score','ls_lucky_stack','ls_lucky_align','ls_lucky_roi','use_native_sensor_mode',
          'focus_method','star_metric','snr_display','metrics_interval','ls_lucky_save_progress','isp_enable',
          'allsky_mode','allsky_mean_target','allsky_mean_threshold','allsky_video_fps','allsky_max_gain','allsky_apply_stretch','allsky_cleanup_jpegs',
          'ls_save_progress','ls_save_final','ls_lucky_save_final',
          'fix_bad_pixels','fix_bad_pixels_sigma','fix_bad_pixels_min_adu']
points = [mode,speed,gain,brightness,contrast,frame,red,blue,ev,vlen,fps,vformat,codec,tinterval,tshots,extn,zx,zy,zoom,saturation,
          meter,awb,sharpness,denoise,quality,profile,level,histogram,histarea,v3_f_speed,v3_f_range,rotate,IRF,str_cap,v3_hdr,raw_format,vflip,hflip,
          stretch_p_low,stretch_p_high,stretch_factor,stretch_preset,ghs_D,ghs_b,ghs_SP,ghs_LP,ghs_HP,ghs_preset,
          ls_preview_refresh,ls_alignment_mode,ls_enable_qc,ls_max_fwhm,ls_min_sharpness,ls_max_drift,ls_min_stars,
          ls_stack_method,ls_stack_kappa,ls_stack_iterations,
          ls_planetary_enable,ls_planetary_mode,ls_planetary_disk_min,ls_planetary_disk_max,ls_planetary_threshold,ls_planetary_margin,ls_planetary_ellipse,ls_planetary_window,ls_planetary_upsample,ls_planetary_highpass,ls_planetary_roi_center,ls_planetary_corr,ls_planetary_max_shift,
          ls_lucky_buffer,ls_lucky_keep,ls_lucky_score,ls_lucky_stack,ls_lucky_align,ls_lucky_roi,use_native_sensor_mode,
          focus_method,star_metric,snr_display,metrics_interval,ls_lucky_save_progress,isp_enable,
          allsky_mode,allsky_mean_target,allsky_mean_threshold,allsky_video_fps,allsky_max_gain,allsky_apply_stretch,allsky_cleanup_jpegs,
          ls_save_progress,ls_save_final,ls_lucky_save_final,
          fix_bad_pixels,fix_bad_pixels_sigma,fix_bad_pixels_min_adu]
if not os.path.exists(config_file):
    with open(config_file, 'w') as f:
        for item in range(0,len(titles)):
            f.write( titles[item] + " : " + str(points[item]) + "\n")

# read config_file
config = []
with open(config_file, "r") as file:
   line = file.readline()
   while line:
       line = line.strip()
       item = line.split(" : ")
       config.append(item[1])
       line = file.readline()
# Convertir d'abord en float, puis en int sauf pour tinterval (index 13)
config = list(map(float,config))
for i in range(len(config)):
    if i != 13:  # Garder tinterval comme float
        config[i] = int(config[i])

mode        = config[0]
speed       = config[1]
gain        = config[2]
brightness  = config[3]
contrast    = config[4]
red         = config[6]
blue        = config[7]
ev          = config[8]
vlen        = config[9]
fps         = config[10]
vformat     = config[11]
codec       = config[12]
tinterval   = config[13]
tshots      = config[14]
extn        = config[15]
zx          = config[16]
zy          = config[17]
zoom        = 0
saturation  = config[19]
meter       = config[20]
awb         = config[21]
sharpness   = config[22]
denoise     = config[23]
quality     = config[24]
profile     = config[25]
level       = config[26]
histogram   = config[27]
histarea    = config[28]
v3_f_speed  = config[29]
v3_f_range  = config[30]
rotate      = config[31]
IRF         = config[32]
str_cap     = config[33]
v3_hdr      = config[34]
raw_format  = config[35]  # Remplace timet (maintenant fixé à 100ms)
vflip       = config[36]
hflip       = config[37]

# Ajouter les nouveaux paramètres stretch si le fichier de config est ancien
if len(config) <= 38:
    config.append(0)     # stretch_p_low par défaut (0%)
if len(config) <= 39:
    config.append(9998)  # stretch_p_high par défaut (99.98%)
if len(config) <= 40:
    config.append(25)    # stretch_factor par défaut (2.5)
if len(config) <= 41:
    config.append(0)     # stretch_preset par défaut (OFF)
if len(config) <= 42:
    config.append(31)    # ghs_D par défaut (3.1 optimisé)
if len(config) <= 43:
    config.append(1)     # ghs_b par défaut (0.1 optimisé)
if len(config) <= 44:
    config.append(19)    # ghs_SP par défaut (0.19 optimisé)
if len(config) <= 45:
    config.append(0)     # ghs_LP par défaut (0.0)
if len(config) <= 46:
    config.append(0)     # ghs_HP par défaut (0.0 optimisé)
if len(config) <= 47:
    config.append(1)     # ghs_preset par défaut (Galaxies)

stretch_p_low    = config[38]
stretch_p_high   = config[39]
stretch_factor   = config[40]
stretch_preset   = config[41]
ghs_D            = config[42]
ghs_b            = config[43]
ghs_SP           = config[44]
ghs_LP           = config[45]
ghs_HP           = config[46]
ghs_preset       = config[47]

# Ajouter les paramètres livestack si le fichier de config est ancien (indices décalés de +3 à cause de ghs_LP, ghs_HP, ghs_preset)
if len(config) <= 48:
    config.append(5)     # ls_preview_refresh par défaut
if len(config) <= 49:
    config.append(2)     # ls_alignment_mode par défaut (rotation)
if len(config) <= 50:
    config.append(0)     # ls_enable_qc par défaut (désactivé)
if len(config) <= 51:
    config.append(170)   # ls_max_fwhm par défaut (17.0)
if len(config) <= 52:
    config.append(70)    # ls_min_sharpness par défaut (0.070)
if len(config) <= 53:
    config.append(2500)  # ls_max_drift par défaut
if len(config) <= 54:
    config.append(10)    # ls_min_stars par défaut

# Ajouter les nouveaux paramètres si le fichier de config est ancien
if len(config) <= 55:
    config.append(0)     # ls_stack_method par défaut (mean)
if len(config) <= 56:
    config.append(25)    # ls_stack_kappa par défaut (2.5)
if len(config) <= 57:
    config.append(3)     # ls_stack_iterations par défaut
if len(config) <= 58:
    config.append(0)     # ls_planetary_enable par défaut (off)
if len(config) <= 59:
    config.append(1)     # ls_planetary_mode par défaut (surface)
if len(config) <= 60:
    config.append(50)    # ls_planetary_disk_min
if len(config) <= 61:
    config.append(500)   # ls_planetary_disk_max
if len(config) <= 62:
    config.append(30)    # ls_planetary_threshold
if len(config) <= 63:
    config.append(10)    # ls_planetary_margin
if len(config) <= 64:
    config.append(0)     # ls_planetary_ellipse
if len(config) <= 65:
    config.append(1)     # ls_planetary_window (256)
if len(config) <= 66:
    config.append(10)    # ls_planetary_upsample
if len(config) <= 67:
    config.append(1)     # ls_planetary_highpass
if len(config) <= 68:
    config.append(1)     # ls_planetary_roi_center
if len(config) <= 69:
    config.append(30)    # ls_planetary_corr (0.30)
if len(config) <= 70:
    config.append(100)   # ls_planetary_max_shift
if len(config) <= 71:
    config.append(10)    # ls_lucky_buffer (10 images)
if len(config) <= 72:
    config.append(10)    # ls_lucky_keep (10%)
if len(config) <= 73:
    config.append(0)     # ls_lucky_score (laplacian)
if len(config) <= 74:
    config.append(0)     # ls_lucky_stack (mean)
if len(config) <= 75:
    config.append(1)     # ls_lucky_align (on)
if len(config) <= 76:
    config.append(50)    # ls_lucky_roi (50%)
if len(config) <= 77:
    config.append(0)     # use_native_sensor_mode par défaut (binning)

# Ajouter les paramètres METRICS si le fichier de config est ancien
if len(config) <= 78:
    config.append(1)     # focus_method par défaut (Laplacian)
if len(config) <= 79:
    config.append(1)     # star_metric par défaut (HFR)
if len(config) <= 80:
    config.append(0)     # snr_display par défaut (OFF)
if len(config) <= 81:
    config.append(3)     # metrics_interval par défaut (3 frames)

ls_preview_refresh = config[48]
ls_alignment_mode  = config[49]
ls_enable_qc       = config[50]
ls_max_fwhm        = config[51]
ls_min_sharpness   = config[52]
ls_max_drift       = config[53]
ls_min_stars       = config[54]

ls_stack_method    = config[55]
ls_stack_kappa     = config[56]
ls_stack_iterations = config[57]

ls_planetary_enable = config[58]
ls_planetary_mode   = config[59]
ls_planetary_disk_min = config[60]
ls_planetary_disk_max = config[61]
ls_planetary_threshold = config[62]
ls_planetary_margin = config[63]
ls_planetary_ellipse = config[64]
ls_planetary_window = config[65]
ls_planetary_upsample = config[66]
ls_planetary_highpass = config[67]
ls_planetary_roi_center = config[68]
ls_planetary_corr   = config[69]
ls_planetary_max_shift = config[70]

ls_lucky_buffer    = config[71]
ls_lucky_keep      = config[72]
ls_lucky_score     = config[73]
ls_lucky_stack     = config[74]
ls_lucky_align     = config[75]
ls_lucky_roi       = config[76]
use_native_sensor_mode = config[77] if len(config) > 77 else 0
ls_lucky_save_progress = config[82] if len(config) > 82 else 0

focus_method       = config[78] if len(config) > 78 else 1
star_metric        = config[79] if len(config) > 79 else 1
snr_display        = config[80] if len(config) > 80 else 0
metrics_interval   = config[81] if len(config) > 81 else 3
isp_enable         = config[83] if len(config) > 83 else 0

# Ajouter les paramètres ALLSKY si le fichier de config est ancien
if len(config) <= 84:
    config.append(0)     # allsky_mode par défaut (OFF)
if len(config) <= 85:
    config.append(30)    # allsky_mean_target par défaut (0.30)
if len(config) <= 86:
    config.append(5)     # allsky_mean_threshold par défaut (0.05)
if len(config) <= 87:
    config.append(25)    # allsky_video_fps par défaut (25 fps)
if len(config) <= 88:
    config.append(200)   # allsky_max_gain par défaut (200)
if len(config) <= 89:
    config.append(1)     # allsky_apply_stretch par défaut (ON)
if len(config) <= 90:
    config.append(0)     # allsky_cleanup_jpegs par défaut (keep JPEGs)
# Ajouter les paramètres de sauvegarde LiveStack/LuckyStack si le fichier de config est ancien
if len(config) <= 91:
    config.append(1)     # ls_save_progress par défaut (ON)
if len(config) <= 92:
    config.append(1)     # ls_save_final par défaut (ON)
if len(config) <= 93:
    config.append(1)     # ls_lucky_save_final par défaut (ON)

# Charger les paramètres ALLSKY
allsky_mode           = config[84] if len(config) > 84 else 0
allsky_mean_target    = config[85] if len(config) > 85 else 30
allsky_mean_threshold = config[86] if len(config) > 86 else 5
allsky_video_fps      = config[87] if len(config) > 87 else 25
allsky_max_gain       = config[88] if len(config) > 88 else 200
allsky_apply_stretch  = config[89] if len(config) > 89 else 1
allsky_cleanup_jpegs  = config[90] if len(config) > 90 else 0

# Options de sauvegarde LiveStack/LuckyStack
ls_save_progress      = config[91] if len(config) > 91 else 1
ls_save_final         = config[92] if len(config) > 92 else 1
ls_lucky_save_final   = config[93] if len(config) > 93 else 1

# Suppression des pixels chauds (hot pixels)
fix_bad_pixels        = config[94] if len(config) > 94 else 1
fix_bad_pixels_sigma  = config[95] if len(config) > 95 else 40
fix_bad_pixels_min_adu = config[96] if len(config) > 96 else 100

# ISP configuration (chemin en dur pour le fichier de config ISP)
# Config neutre (transparente) pour astrophotographie - n'affecte que les PNG de prévisualisation
isp_config_path = "isp_config_neutral.json"

# VALIDATION GLOBALE - S'assurer que TOUTES les valeurs sont dans les limites correctes
# pour éviter les IndexError et les valeurs invalides

# Live Stack parameters
ls_preview_refresh = max(1, min(ls_preview_refresh, 10))
ls_alignment_mode = max(0, min(ls_alignment_mode, 2))
ls_enable_qc = max(0, min(ls_enable_qc, 1))
ls_max_fwhm = max(0, min(ls_max_fwhm, 300))
ls_min_sharpness = max(0, min(ls_min_sharpness, 200))
ls_max_drift = max(0, min(ls_max_drift, 5000))
ls_min_stars = max(0, min(ls_min_stars, 20))

# Stacking parameters
ls_stack_method = max(0, min(ls_stack_method, 4))  # 0-4: mean/median/kappa/winsorized/weighted
ls_stack_kappa = max(10, min(ls_stack_kappa, 40))
ls_stack_iterations = max(1, min(ls_stack_iterations, 10))

# Planetary parameters
ls_planetary_enable = max(0, min(ls_planetary_enable, 1))
ls_planetary_mode = max(0, min(ls_planetary_mode, 2))  # 0-2: disk/surface/hybrid
ls_planetary_disk_min = max(10, min(ls_planetary_disk_min, 200))
ls_planetary_disk_max = max(50, min(ls_planetary_disk_max, 1000))
ls_planetary_threshold = max(10, min(ls_planetary_threshold, 100))
ls_planetary_margin = max(5, min(ls_planetary_margin, 50))
ls_planetary_ellipse = max(0, min(ls_planetary_ellipse, 1))
ls_planetary_window = max(0, min(ls_planetary_window, 2))  # 0-2: index dans planetary_windows
ls_planetary_upsample = max(5, min(ls_planetary_upsample, 20))
ls_planetary_highpass = max(0, min(ls_planetary_highpass, 1))
ls_planetary_roi_center = max(0, min(ls_planetary_roi_center, 1))
ls_planetary_corr = max(10, min(ls_planetary_corr, 100))
ls_planetary_max_shift = max(10, min(ls_planetary_max_shift, 500))

# Lucky Imaging parameters
ls_lucky_buffer = max(10, min(ls_lucky_buffer, 200))
ls_lucky_keep = max(1, min(ls_lucky_keep, 50))
ls_lucky_score = max(0, min(ls_lucky_score, 3))  # 0-3: laplacian/gradient/sobel/tenengrad
ls_lucky_stack = max(0, min(ls_lucky_stack, 2))  # 0-2: mean/median/sigma_clip
ls_lucky_align = max(0, min(ls_lucky_align, 1))
ls_lucky_roi = max(20, min(ls_lucky_roi, 100))
ls_lucky_save_progress = max(0, min(ls_lucky_save_progress, 1))  # 0-1: off/on

# Validation des options de sauvegarde LiveStack/LuckyStack
ls_save_progress = max(0, min(ls_save_progress, 1))  # 0-1: off/on
ls_save_final = max(0, min(ls_save_final, 1))  # 0-1: off/on
ls_lucky_save_final = max(0, min(ls_lucky_save_final, 1))  # 0-1: off/on

# Metrics parameters
focus_method = max(0, min(focus_method, 4))  # 0-4: OFF/Laplacian/Gradient/Sobel/Tenengrad
star_metric = max(0, min(star_metric, 2))    # 0-2: OFF/HFR/FWHM
snr_display = max(0, min(snr_display, 1))    # 0-1: OFF/ON

# RAW Format parameter
raw_format = max(0, min(raw_format, 3))      # 0-3: YUV420/XRGB8888/SRGGB12/SRGGB16
metrics_interval = max(1, min(metrics_interval, 10))  # 1-10 frames

# Sensor mode
use_native_sensor_mode = max(0, min(use_native_sensor_mode, 1))  # 0-1: Binning/Native

# ISP parameter
isp_enable = max(0, min(isp_enable, 1))  # 0-1: OFF/ON

# Hot pixels removal parameters
fix_bad_pixels = max(0, min(fix_bad_pixels, 1))  # 0-1: OFF/ON
fix_bad_pixels_sigma = max(10, min(fix_bad_pixels_sigma, 100))  # 1.0-10.0 sigma (×10)
fix_bad_pixels_min_adu = max(0, min(fix_bad_pixels_min_adu, 1000))  # 0-100 ADU (×10)

# Debug ISP
print(f"[DEBUG CONFIG] isp_enable = {isp_enable}, isp_config_path = {isp_config_path}")

# Debug Hot Pixels Removal
status = "ON" if fix_bad_pixels else "OFF"
print(f"[DEBUG CONFIG] Hot Pixels Removal: {status}, sigma={fix_bad_pixels_sigma/10.0}, min_adu={fix_bad_pixels_min_adu/10.0}")

# Étendre config à 97 éléments si nécessaire (pour fix_bad_pixels_min_adu à index 96)
while len(config) <= 96:
    config.append(0)

if codec > len(codecs)-1:
    codec = 0

def get_native_vformats():
    """
    Retourne la liste des index vformat correspondant aux résolutions natives du capteur.
    Utilise les MÊMES résolutions que le système de zoom pour cohérence.
    """
    global vwidths, vheights, Pi_Cam

    # Résolutions correspondant aux niveaux de zoom (IDENTIQUES à sync_video_resolution_with_zoom)
    if Pi_Cam == 10:  # IMX585 - Utiliser modes hardware crop (ordre décroissant)
        zoom_resolutions = [
            (3856, 2180),  # Mode natif full frame (zoom 0)
            (2880, 2160),  # Mode 2 crop (zoom 1)
            (1928, 1090),  # Mode binning 2x2
            (1920, 1080),  # Mode 3 crop (zoom 2)
            (1280, 720),   # Mode 4 crop (zoom 3)
            (800, 600),    # Mode 5 crop (zoom 4/5)
        ]
    else:
        # Autres caméras: résolutions standards supportées (ordre décroissant)
        zoom_resolutions = [
            (2880, 2160),  # zoom 1
            (1920, 1080),  # zoom 2
            (1280, 720),   # zoom 3
            (800, 600),    # zoom 4/5
        ]

    # Trouver les index vformat correspondant à ces résolutions
    native_vformats = []
    for res_width, res_height in zoom_resolutions:
        for i in range(len(vwidths)):
            if vwidths[i] == res_width and vheights[i] == res_height:
                if i not in native_vformats:  # Éviter les doublons
                    native_vformats.append(i)
                break

    return native_vformats

def get_next_native_vformat(current_vformat, direction=1):
    """
    Retourne le prochain vformat natif dans la direction spécifiée.
    direction = 1 pour suivant, -1 pour précédent.
    """
    native_vformats = get_native_vformats()

    if len(native_vformats) == 0:
        return current_vformat

    # Trouver l'index actuel dans la liste native
    if current_vformat in native_vformats:
        current_idx = native_vformats.index(current_vformat)
    else:
        # Si pas natif, trouver le plus proche
        if direction > 0:
            # Trouver le premier natif supérieur
            for nv in native_vformats:
                if nv > current_vformat:
                    return nv
            return native_vformats[-1]  # Retourner le dernier si aucun supérieur
        else:
            # Trouver le premier natif inférieur
            for nv in reversed(native_vformats):
                if nv < current_vformat:
                    return nv
            return native_vformats[0]  # Retourner le premier si aucun inférieur

    # Incrémenter/décrémenter dans la liste native
    new_idx = current_idx + direction

    # Contraindre aux limites
    if new_idx < 0:
        new_idx = 0
    elif new_idx >= len(native_vformats):
        new_idx = len(native_vformats) - 1

    return native_vformats[new_idx]

def setmaxvformat():
    # set max video format - UNIQUEMENT résolutions natives du capteur (comme le zoom)
    global codec,Pi_Cam,configtxt,max_vformat,max_vfs,vwidths,vheights,vwidths2,vheights2

    # Utiliser la liste de résolutions natives
    native_vformats = get_native_vformats()

    # Définir max_vformat comme le dernier index natif trouvé
    if len(native_vformats) > 0:
        max_vformat = native_vformats[-1]
    else:
        # Fallback vers l'ancienne méthode si aucune résolution native trouvée
        if codec > 0 and (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and ("dtoverlay=vc4-kms-v3d,cma-512" in configtxt):
            max_vformat = max_vfs[6]
        elif codec > 0 and (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
            max_vformat = max_vfs[5]
        elif codec > 0:
            max_vformat = max_vfs[Pi_Cam]
        elif Pi_Cam == 7 or Pi_Cam == 15:
            max_vformat = max_vfs[Pi_Cam]
        else:
            max_vformat = max_vfs[0]
        if Pi_Cam == 4 and codec == 0:
            max_vformat = 12
    
def slider_to_gain_nonlinear(slider_value, max_gain):
    """
    Convertit position slider linéaire (0-max_gain) en gain non-linéaire (1-max_gain)
    70% de la plage → gain 1-1000 (précis)
    30% de la plage → gain 1000-max_gain (moins précis)
    """
    if max_gain <= 1000:
        # Pour caméras avec gain max <= 1000, rester linéaire
        return max(1, slider_value)

    # Normaliser la position (0.0 à 1.0)
    slider_pos = slider_value / max_gain

    if slider_pos <= 0.7:
        # 70% du slider → gain 1 à 1000 (précis)
        return int(1 + (slider_pos / 0.7) * 999)
    else:
        # 30% du slider → gain 1000 à max_gain (moins précis)
        return int(1000 + ((slider_pos - 0.7) / 0.3) * (max_gain - 1000))

def gain_to_slider_nonlinear(gain_value, max_gain):
    """
    Inverse : convertit gain (1-max_gain) en position slider linéaire (0-max_gain)
    Pour ajustements +/- des boutons
    """
    if max_gain <= 1000:
        return gain_value

    if gain_value <= 1000:
        # gain 1-1000 → 0-70% du slider
        return int(((gain_value - 1) / 999) * 0.7 * max_gain)
    else:
        # gain 1000-max_gain → 70-100% du slider
        return int((0.7 + ((gain_value - 1000) / (max_gain - 1000)) * 0.3) * max_gain)

def Camera_Version():
    # Check for Pi Camera version
    global lver,v3_af,camera,vwidths2,vheights2,configtxt,mode,mag,max_gain,max_shutter,Pi_Cam,max_camera,same_cams,x_sens,y_sens,igw,igh
    global cam0,cam1,cam2,cam3,max_gains,max_shutters,scientif,max_vformat,vformat,vwidth,vheight,vfps,sspeed,tduration,video_limits,lo_res
    global speed,shutter,max_vf_7,max_vf_6,max_vf_5,max_vf_4,max_vf_3,max_vf_2,max_vf_1,max_vf_4a,max_vf_0,max_vf_8,max_vf_9,IRF,foc_sub3
    global foc_sub5,v3_hdr,windowSurfaceObj,cam1,custom_sspeed
    # DETERMINE NUMBER OF CAMERAS (FOR ARDUCAM MULITPLEXER or Pi5)
    if os.path.exists('rpicams.txt'):
        os.rename('rpicams.txt', 'oldrpicams.txt')
    if lver != "bookwo" and lver != "trixie":
        os.system("libcamera-vid --list-cameras >> rpicams.txt")
    else:
        os.system("rpicam-vid --list-cameras >> rpicams.txt")
    time.sleep(0.5)
    # read rpicams.txt file
    camstxt = []
    with open("rpicams.txt", "r") as file:
        line = file.readline()
        while line:
            camstxt.append(line.strip())
            line = file.readline()
    max_camera = 0
    same_cams  = 0
    lo_res = 1
    cam0 = "0"
    cam1 = "1"
    cam2 = "2"
    cam3 = "3"
    Pi_Cam = -1
    vwidths2  = []
    vheights2 = []
    vfps2 = []
    for x in range(0,len(camstxt)):
        # Determine camera models
        if camstxt[x][0:4] == "0 : ":
            cam0 = camstxt[x][4:10]
        if cam0 != "0" and cam1 == "1" and camera == 0:
            # determine native formats
            forms = camstxt[x].split(" ")
            for q in range(0,len(forms)):
                if "x" in forms[q] and "/" not in forms[q] and "m" not in forms[q] and "[" not in forms[q]:
                    qwidth,qheight = forms[q].split("x")
                    vwidths2.append(int(qwidth))
                    vheights2.append(int(qheight))
                if forms[q][0:1] == "[" and "x" not in forms[q]:
                    vfps2.append(int(float(forms[q][1:4])))
        if camstxt[x][0:4] == "1 : ":
            cam1 = camstxt[x][4:10]
        if cam0 != "0" and cam1 != "1" and camera == 1:
              # determine native formats
              forms = camstxt[x].split(" ")
              for q in range(0,len(forms)):
               if "x" in forms[q] and "/" not in forms[q] and "m" not in forms[q] and "[" not in forms[q]:
                  qwidth,qheight = forms[q].split("x")
                  vwidths2.append(int(qwidth))
                  vheights2.append(int(qheight))
               if forms[q][0:1] == "[" and "x" not in forms[q]:
                    vfps2.append(int(float(forms[q][1:4])))
        if camstxt[x][0:4] == "2 : ":
            cam2 = camstxt[x][4:10]
        if camstxt[x][0:4] == "3 : ":
            cam3 = camstxt[x][4:10]
        # Determine MAXIMUM number of cameras available 
        if camstxt[x][0:4]   == "3 : " and max_camera < 3:
            max_camera = 3
        elif camstxt[x][0:4] == "2 : " and max_camera < 2:
            max_camera = 2
        elif camstxt[x][0:4] == "1 : " and max_camera < 1:
            max_camera = 1
        pic = 0
        Pi_Cam = -1
        for x in range(0,len(camids)):
            if camera == 0:
                if cam0 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            elif camera == 1:
                if cam1 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            elif camera == 2:
                if cam2 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            elif camera == 3:
                if cam3 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            if pic == 1:
                max_shutter = max_shutters[Pi_Cam]
                max_gain = max_gains[Pi_Cam]
                mag = int(max_gain/4)
                still_limits[8] = max_gain
    if Pi_Cam != -1:
        print("Camera:",cameras[Pi_Cam])
    elif cam0 != "0" and camera == 0:
        Pi_Cam      = 0
        cameras[0]  = cam0
        camids[0]   = cam0[0:6]
        print("Camera:",cameras[Pi_Cam])
        max_shutter = max_shutters[Pi_Cam]
        max_gain    = max_gains[Pi_Cam]
        mag         = int(max_gain/4)
        still_limits[8] = max_gain
        x_sens[0] = vwidths2[len(vwidths2)-1]
        y_sens[0] = vheights2[len(vheights2)-1]
    elif cam1 != "1" and camera == 1:
        Pi_Cam      = 0
        cameras[0]  = cam1
        camids[0]   = cam1[0:6]
        print("Camera:",cameras[Pi_Cam])
        max_shutter = max_shutters[Pi_Cam]
        max_gain    = max_gains[Pi_Cam]
        mag         = int(max_gain/4)
        still_limits[8] = max_gain
        x_sens[0] = vwidths2[len(vwidths2)-1]
        y_sens[0] = vheights2[len(vheights2)-1]
    else:
        print("No Camera Found")
        pygame.display.quit()
        sys.exit()

    igw = x_sens[Pi_Cam]
    igh = y_sens[Pi_Cam]

    if igw/igh > 1.5:
        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,int(preview_height *.25) ))

            
    if max_camera == 1 and cam0 == cam1:
        same_cams = 1
    configtxt = []
    if Pi_Cam == 9:
        if IRF == 0:
            led_sw_ir.off()
        else:
            led_sw_ir.on()
    if Pi_Cam != 3 and v3_hdr > 0:
        v3_hdr = 1
    if Pi_Cam == 3 or Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8:
        # read /boot/config.txt file
        if lver != "bookwo" and lver != "trixie":
          with open("/boot/config.txt", "r") as file:
            line = file.readline()
            while line:
                configtxt.append(line.strip())
                line = file.readline()
        else:
          with open("/boot/firmware/config.txt", "r") as file:
            line = file.readline()
            while line:
                configtxt.append(line.strip())
                line = file.readline()
        # determine /dev/v4l-subdevX for Pi v3 and Arducam 16/64MP cameras
        foc_sub3 = -1
        foc_sub5 = -1
        if Pi_Cam == 3 or Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8: # AF cameras
          for x in range(0,10):
            if os.path.exists("ctrls1.txt"):
                os.remove("ctrls1.txt")
            os.system("v4l2-ctl -d /dev/v4l-subdev" + str(x) + " --list-ctrls >> ctrls1.txt")
            time.sleep(0.25)
            ctrlstxt = []
            with open("ctrls1.txt", "r") as file:
                line = file.readline()
                while line:
                    ctrlstxt.append(line.strip())
                    line = file.readline()
            for a in range(0,len(ctrlstxt)):
                if ctrlstxt[a][0:45] == "exposure 0x00980911 (int)    : min=9 max=7079" and foc_sub5 == -1 and Pi_Cam == 6: # arducam 64mp hawkeye
                    foc_sub5 = x + 1
                elif ctrlstxt[a][0:51] == "focus_absolute 0x009a090a (int)    : min=0 max=4095" and foc_sub5 == -1 and Pi_Cam == 5: # arducam 16mp
                    foc_sub5 = x
                elif ctrlstxt[a][0:45] == "exposure 0x00980911 (int)    : min=1 max=2602" and Pi_Cam == 3: # pi v3
                    foc_sub3 = x + 1
                elif ctrlstxt[a][0:37] == "exposure 0x00980911 (int)    : min=16" and Pi_Cam == 8: # arducam owlsight 64mp
                    foc_sub3 = x + 1
    if cam0 != "0" and cam1 != "1":
        pygame.display.set_caption('RPiCamera:  v' + str(version) + "  Pi: " + str(Pi) + "  Camera: "  + cameras[Pi_Cam] + " : " + str(camera))
    else:
        pygame.display.set_caption('RPiCamera:  v' + str(version) + "  Pi: " + str(Pi) + "  Camera: "  + cameras[Pi_Cam] )

    if (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and ("dtoverlay=vc4-kms-v3d,cma-512" in configtxt):
        lo_res = 0
    # set max video format
    setmaxvformat()
    if vformat > max_vformat:
        vformat = max_vformat
    if Pi_Cam == 4:  # Pi HQ
        if codec == 0:
            max_vformat = 12
        if ((Pi != 5 and os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json')) or (Pi == 5 and os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json'))):
            scientif = 1
        else:
            scientif = 0
    vwidth    = vwidths[vformat]
    vheight   = vheights[vformat]
    # set max fps
    if Pi_Cam == 3:
        vfps = v3_max_fps[vformat]
    elif Pi_Cam == 9:
        vfps = v9_max_fps[vformat]
    elif Pi_Cam == 10:
        vfps = v10_max_fps[vformat]
    elif Pi_Cam == 15:
        vfps = v15_max_fps[vformat]
    else:
        vfps = v_max_fps[vformat]
    video_limits[5] = vfps
    if tinterval > 0:
        tduration = tinterval * tshots
    else:
        tduration = 5
    # Calculer sspeed: utiliser custom_sspeed si défini, sinon depuis shutters[speed]
    if custom_sspeed > 0:
        sspeed = custom_sspeed
    else:
        shutter = shutters[speed]
        if shutter < 0:
            shutter = abs(1/shutter)
        sspeed = int(shutter * 1000000)
        if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
            sspeed +=1
    # determine max speed for camera
    max_speed = 0
    while max_shutter > shutters[max_speed]:
        max_speed +=1
    if speed > max_speed:
        speed = max_speed
        custom_sspeed = 0  # Réinitialiser car speed a changé
        shutter = shutters[speed]
        if shutter < 0:
            shutter = abs(1/shutter)
        sspeed = int(shutter * 1000000)
        if mode == 0:
            if shutters[speed] < 0:
                text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
            else:
                text(0,2,3,1,1,str(shutters[speed]),fv,10)
        else:
            if shutters[speed] < 0:
                text(0,2,0,1,1,"1/" + str(abs(shutters[speed])),fv,10)
            else:
                text(0,2,0,1,1,str(shutters[speed]),fv,10)

# ============================================================================
# MODE TEST RAW - Activation avec python3 RPiCamera2.py --test-raw
# ============================================================================
def test_raw_modes():
    """
    Mode test pour analyser les formats RAW disponibles
    S'exécute avant l'interface graphique
    """
    print("\n" + "="*70)
    print("🎥 MODE TEST RAW - Analyse des formats disponibles")
    print("="*70)

    try:
        # Initialiser la caméra
        print("\n📷 Initialisation caméra...")
        test_picam2 = Picamera2()

        # Informations capteur
        print("\n" + "="*70)
        print("INFORMATIONS CAPTEUR")
        print("="*70)
        camera_props = test_picam2.camera_properties
        print(f"Modèle : {camera_props.get('Model', 'N/A')}")
        print(f"Résolution : {camera_props.get('PixelArraySize', 'N/A')}")

        if 'ColorFilterArrangement' in camera_props:
            cfa = camera_props['ColorFilterArrangement']
            bayer_map = {0: 'RGGB', 1: 'GRBG', 2: 'GBRG', 3: 'BGGR', 4: 'MONO'}
            print(f"Matrice Bayer : {bayer_map.get(cfa, f'Unknown ({cfa})')}")

        # Test formats disponibles
        print("\n" + "="*70)
        print("TEST DES FORMATS DISPONIBLES")
        print("="*70)

        formats_to_test = [
            ("RGB888", "RGB 8-bit standard"),
            ("YUV420", "YUV420 non compressé"),
            ("XRGB8888", "XRGB 8-bit"),
            ("SRGGB10", "RAW Bayer 10-bit"),
            ("SRGGB10_CSI2P", "RAW Bayer 10-bit CSI2"),
            ("SRGGB12", "RAW Bayer 12-bit"),
            ("SRGGB16", "RAW Bayer 16-bit"),
        ]

        available = []
        for fmt, desc in formats_to_test:
            try:
                cfg = test_picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": fmt}
                )
                print(f"✅ {fmt:20s} - {desc}")
                available.append(fmt)
            except Exception as e:
                print(f"❌ {fmt:20s} - Non supporté")

        # Test capture RAW
        print("\n" + "="*70)
        print("TEST CAPTURE FLUX RAW")
        print("="*70)

        try:
            # Configuration avec RAW
            sensor_res = test_picam2.sensor_resolution
            raw_cfg = test_picam2.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                raw={"size": sensor_res}
            )

            print(f"Configuration : {raw_cfg}")
            test_picam2.configure(raw_cfg)
            test_picam2.start()
            time.sleep(1)

            # Tester capture RGB
            print("\n📸 Test flux 'main' (RGB)...")
            main_array = test_picam2.capture_array("main")
            print(f"   ✅ Shape: {main_array.shape}, dtype: {main_array.dtype}")

            # Tester capture RAW
            print("\n📸 Test flux 'raw' (Bayer)...")
            raw_array = test_picam2.capture_array("raw")
            print(f"   ✅ Shape: {raw_array.shape}, dtype: {raw_array.dtype}")
            print(f"   📊 Min/Max: {raw_array.min()} / {raw_array.max()}")
            print(f"   💾 Taille: {raw_array.nbytes / 1024 / 1024:.2f} MB")

            # Test de debayerisation
            print("\n🔬 Test debayerisation...")
            patterns = [
                (cv2.COLOR_BayerRG2RGB, "RGGB"),
                (cv2.COLOR_BayerGR2RGB, "GRBG"),
                (cv2.COLOR_BayerGB2RGB, "GBRG"),
                (cv2.COLOR_BayerBG2RGB, "BGGR"),
            ]

            # Convertir en uint8 si nécessaire
            if raw_array.dtype == np.uint16:
                test_array = (raw_array / 256).astype(np.uint8)
            else:
                test_array = raw_array.astype(np.uint8)

            for pattern_code, pattern_name in patterns:
                try:
                    rgb = cv2.cvtColor(test_array, pattern_code)
                    print(f"   ✅ {pattern_name} : OK ({rgb.shape})")
                except Exception as e:
                    print(f"   ❌ {pattern_name} : {e}")

            test_picam2.stop()

        except Exception as e:
            print(f"❌ Erreur test capture : {e}")
            import traceback
            traceback.print_exc()

        # Test performance
        print("\n" + "="*70)
        print("TEST PERFORMANCE CAPTURE")
        print("="*70)

        try:
            # Configuration RAW
            raw_cfg = test_picam2.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                raw={"size": test_picam2.sensor_resolution}
            )
            test_picam2.configure(raw_cfg)
            test_picam2.start()
            time.sleep(1)

            # Test RGB
            print("\n📊 Test RGB (50 frames)...")
            start = time.time()
            for i in range(50):
                _ = test_picam2.capture_array("main")
            rgb_time = time.time() - start
            rgb_fps = 50 / rgb_time
            print(f"   FPS: {rgb_fps:.1f}, Latence: {rgb_time/50*1000:.1f}ms/frame")

            # Test RAW
            print("\n📊 Test RAW (50 frames)...")
            start = time.time()
            for i in range(50):
                _ = test_picam2.capture_array("raw")
            raw_time = time.time() - start
            raw_fps = 50 / raw_time
            print(f"   FPS: {raw_fps:.1f}, Latence: {raw_time/50*1000:.1f}ms/frame")

            # Comparaison
            print("\n💡 RÉSUMÉ:")
            print(f"   RGB : {rgb_fps:.1f} fps")
            print(f"   RAW : {raw_fps:.1f} fps ({(raw_fps/rgb_fps*100):.0f}% de RGB)")

            if raw_fps >= 25:
                print("\n   ✅ RAW assez rapide pour Lucky Imaging !")
            else:
                print("\n   ⚠️  RAW trop lent pour Lucky Imaging haute cadence")

            test_picam2.stop()

        except Exception as e:
            print(f"❌ Erreur test performance : {e}")

        # Fermer
        test_picam2.close()

        print("\n" + "="*70)
        print("✅ Tests terminés ! Appuyez sur CTRL+C pour quitter.")
        print("="*70)

        input("\nAppuyez sur ENTRÉE pour continuer...")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrompus par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# ANALYSE PROFONDEUR DE BITS - Utilitaire d'analyse
# ============================================================================
def analyze_bit_depth(raw_array, format_name):
    """Analyse la profondeur de bits réelle d'un array brut."""

    print(f"\n{'='*60}")
    print(f"ANALYSE DE PROFONDEUR : {format_name}")
    print(f"{'='*60}\n")

    # Convertir en uint16 si nécessaire (données brutes en uint8)
    print(f"Array brut - Shape: {raw_array.shape}, Dtype: {raw_array.dtype}")

    if raw_array.dtype == np.uint8:
        # Les données sont en bytes, il faut les convertir en uint16
        # Format: little-endian, 2 bytes par pixel
        raw_uint16 = raw_array.view(np.uint16)

        # Calculer la largeur réelle (en excluant le padding du stride)
        height, stride_bytes = raw_array.shape
        width = stride_bytes // 2  # 2 bytes par pixel en 16-bit

        # Reshape et extraire seulement les pixels valides
        raw_array = raw_uint16.reshape(height, -1)[:, :width]

        print(f"Converti en uint16 - Shape: {raw_array.shape}, Dtype: {raw_array.dtype}")

    # Stats de base
    print(f"\nMin value: {raw_array.min()}")
    print(f"Max value: {raw_array.max()}")
    print(f"Mean value: {raw_array.mean():.2f}")
    print(f"Std dev: {raw_array.std():.2f}")

    # Analyse des bits utilisés
    print(f"\n--- ANALYSE DES BITS ---")

    # Vérifier si les valeurs sont multiples de 16 (12-bit shifté de 4 bits)
    multiples_of_16 = np.sum(raw_array % 16 == 0)
    total_pixels = raw_array.size
    percentage_mult_16 = (multiples_of_16 / total_pixels) * 100

    print(f"Valeurs multiples de 16: {multiples_of_16}/{total_pixels} ({percentage_mult_16:.2f}%)")

    # Vérifier si les valeurs sont multiples de autres puissances de 2
    for shift in [1, 2, 3, 4, 5, 6, 7, 8]:
        divisor = 2 ** shift
        multiples = np.sum(raw_array % divisor == 0)
        percentage = (multiples / total_pixels) * 100
        print(f"Valeurs multiples de {divisor:3d}: {multiples}/{total_pixels} ({percentage:.2f}%)")

    # Analyse des 4 bits de poids faible
    print(f"\n--- BITS DE POIDS FAIBLE ---")
    low_4_bits = raw_array & 0x0F  # Masque les 4 bits de poids faible
    unique_low_bits = np.unique(low_4_bits)
    print(f"Valeurs uniques dans les 4 bits de poids faible: {len(unique_low_bits)}")
    print(f"Valeurs: {unique_low_bits[:20]}...")  # Affiche les 20 premières

    if len(unique_low_bits) == 1 and unique_low_bits[0] == 0:
        print("⚠️  Les 4 bits de poids faible sont toujours à 0 → 12-bit codé sur 16-bit")
    else:
        print("✓ Les 4 bits de poids faible varient → Probablement du vrai 16-bit")

    # Analyse des 8 bits de poids faible
    print(f"\n--- BITS DE POIDS FAIBLE (8 bits) ---")
    low_8_bits = raw_array & 0xFF
    unique_low_8_bits = np.unique(low_8_bits)
    print(f"Valeurs uniques dans les 8 bits de poids faible: {len(unique_low_8_bits)}")

    # Histogramme des valeurs
    print(f"\n--- DISTRIBUTION DES VALEURS ---")
    hist, bin_edges = np.histogram(raw_array, bins=16)
    for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
        next_edge = bin_edges[i+1]
        print(f"  [{edge:6.0f} - {next_edge:6.0f}): {count:8d} pixels ({count/total_pixels*100:5.2f}%)")

    # Échantillon de valeurs brutes
    print(f"\n--- ÉCHANTILLON DE VALEURS BRUTES (50 pixels au centre) ---")
    center_y = raw_array.shape[0] // 2
    center_x = raw_array.shape[1] // 2
    sample = raw_array[center_y:center_y+10, center_x:center_x+5].flatten()

    print("Valeurs décimales:")
    print(sample)
    print("\nValeurs hexadécimales:")
    print([hex(v) for v in sample])
    print("\nValeurs binaires (4 bits de poids faible):")
    print([bin(v & 0x0F) for v in sample])

    # Conclusion
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print(f"{'='*60}")

    max_possible_12bit = 4095 << 4  # 12-bit shifté de 4 bits = 65520

    if percentage_mult_16 > 99.0:
        print("✓ Format: 12-bit codé sur 16-bit (shifté de 4 bits)")
        print(f"  Plage réelle: 0 - {max_possible_12bit} (0x0000 - 0xFFF0)")
        print(f"  Bits utilisés: [15:4], bits [3:0] toujours à 0")
        bit_depth = 12
    elif raw_array.max() > max_possible_12bit:
        print("✓ Format: Vrai 16-bit")
        print(f"  Plage réelle: 0 - 65535 (0x0000 - 0xFFFF)")
        print(f"  Tous les bits sont utilisés")
        bit_depth = 16
    elif len(unique_low_bits) > 1:
        print("✓ Format: Probablement vrai 16-bit (bits de poids faible variables)")
        bit_depth = 16
    else:
        print("? Format incertain, analyse manuelle recommandée")
        bit_depth = None

    print(f"{'='*60}\n")

    return bit_depth


# ============================================================================
# MODE TEST 16-BIT - Test de profondeur réelle SRGGB16 vs SRGGB12
# ============================================================================
def test_bit_depth_16():
    """
    Test pour vérifier la profondeur réelle du format SRGGB16.
    Détermine si c'est du 12-bit codé sur 16-bit ou du vrai 16-bit.
    """
    print("\n" + "="*70)
    print("🔬 TEST PROFONDEUR BITS - SRGGB16 vs SRGGB12")
    print("="*70)

    try:
        picam2 = Picamera2()

        # Test 1: SRGGB16
        print("\n\n### TEST 1: Format SRGGB16 ###")

        try:
            config = picam2.create_video_configuration(
                raw={"format": "SRGGB16", "size": (3856, 2180)},
                buffer_count=2
            )
            picam2.configure(config)

            print(f"Configuration appliquée: {config['raw']}")

            picam2.start()
            time.sleep(2)  # Laisser l'AE/AWB se stabiliser

            # Capturer
            print("Capture en cours...")
            start = time.time()
            raw_array = picam2.capture_array("raw")
            capture_time = (time.time() - start) * 1000

            print(f"Capture: {capture_time:.1f} ms")

            # Analyser
            bit_depth_16 = analyze_bit_depth(raw_array, "SRGGB16")

            picam2.stop()

        except Exception as e:
            print(f"❌ Erreur avec SRGGB16: {e}")
            bit_depth_16 = None

        # Test 2: SRGGB12 pour comparaison
        print("\n\n### TEST 2: Format SRGGB12 (pour comparaison) ###")

        try:
            config = picam2.create_video_configuration(
                raw={"format": "SRGGB12", "size": (3856, 2180)},
                buffer_count=2
            )
            picam2.configure(config)

            print(f"Configuration appliquée: {config['raw']}")

            picam2.start()
            time.sleep(2)

            print("Capture en cours...")
            start = time.time()
            raw_array = picam2.capture_array("raw")
            capture_time = (time.time() - start) * 1000

            print(f"Capture: {capture_time:.1f} ms")

            # Analyser
            bit_depth_12 = analyze_bit_depth(raw_array, "SRGGB12")

            picam2.stop()

        except Exception as e:
            print(f"❌ Erreur avec SRGGB12: {e}")
            bit_depth_12 = None

        picam2.close()

        # Résumé final
        print("\n" + "="*70)
        print("RÉSUMÉ FINAL")
        print("="*70)

        if bit_depth_16:
            print(f"SRGGB16: {bit_depth_16}-bit effectifs")
        else:
            print("SRGGB16: Non testé ou erreur")

        if bit_depth_12:
            print(f"SRGGB12: {bit_depth_12}-bit effectifs")
        else:
            print("SRGGB12: Non testé ou erreur")

        print("\nRECOMMANDATION:")
        if bit_depth_16 == 16 and bit_depth_12 == 12:
            print("✓ SRGGB16 offre plus de profondeur que SRGGB12")
            print("  → Utiliser SRGGB16 pour DSO/Lunaire (meilleure qualité)")
            print("  → Tester les performances pour voir si acceptable")
        elif bit_depth_16 == 12 and bit_depth_12 == 12:
            print("⚠️  SRGGB16 et SRGGB12 ont la même profondeur effective (12-bit)")
            print("  → Utiliser SRGGB12 (probablement plus rapide)")
        else:
            print("? Résultats inattendus, analyse manuelle recommandée")

        print("="*70)
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrompus")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# MODE TEST DEBAYER - Test de debayerisation SRGGB12
# ============================================================================
def test_debayer_srggb12():
    """
    Test complet de debayerisation SRGGB12
    Vérifie l'unpacking, le pattern Bayer et la qualité finale
    """
    print("\n" + "="*70)
    print("🔬 TEST DEBAYERISATION SRGGB12 - Point critique")
    print("="*70)

    try:
        # Créer répertoire de sortie
        test_dir = Path("./test_debayer_output")
        test_dir.mkdir(exist_ok=True)
        print(f"\n📁 Répertoire de sortie : {test_dir.absolute()}")

        # Initialiser caméra
        print("\n📷 Initialisation caméra IMX585...")
        test_picam2 = Picamera2()

        sensor_res = test_picam2.sensor_resolution
        print(f"   Résolution capteur : {sensor_res}")

        # ====================================================================
        # TEST 1 : Capture et analyse SRGGB12
        # ====================================================================
        print("\n" + "="*70)
        print("TEST 1 : CAPTURE SRGGB12 - Analyse de la structure")
        print("="*70)

        config_12bit = test_picam2.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"},
            raw={"size": sensor_res, "format": "SRGGB12"}
        )

        print(f"\n📋 Configuration SRGGB12 :")
        print(f"   Main : {config_12bit['main']}")
        print(f"   Raw  : {config_12bit['raw']}")

        test_picam2.configure(config_12bit)
        test_picam2.start()
        time.sleep(1)

        # Capturer RGB de référence
        print("\n📸 Capture RGB888 (référence)...")
        rgb_ref = test_picam2.capture_array("main")
        print(f"   ✅ RGB Shape: {rgb_ref.shape}, dtype: {rgb_ref.dtype}")
        print(f"   📊 RGB Min/Max: {rgb_ref.min()} / {rgb_ref.max()}")

        # Capturer RAW SRGGB12
        print("\n📸 Capture SRGGB12 (raw)...")
        raw_12bit = test_picam2.capture_array("raw")
        print(f"   ✅ RAW Shape: {raw_12bit.shape}, dtype: {raw_12bit.dtype}")
        print(f"   📊 RAW Min/Max: {raw_12bit.min()} / {raw_12bit.max()}")
        print(f"   💾 RAW Taille: {raw_12bit.nbytes / 1024 / 1024:.2f} MB")

        # Capturer les métadonnées pour comprendre le stride
        metadata = test_picam2.capture_metadata()
        print(f"\n📊 Métadonnées :")
        if 'stride' in metadata:
            print(f"   Stride : {metadata.get('stride', 'N/A')} bytes")

        # ====================================================================
        # TEST 2 : Unpacking des données SRGGB12
        # ====================================================================
        print("\n" + "="*70)
        print("TEST 2 : UNPACKING SRGGB12 - Extraction des pixels 12-bit")
        print("="*70)

        print("\n🔧 Analyse de la structure des données...")

        # Le format est généralement : (height, stride_in_bytes)
        height, stride_bytes = raw_12bit.shape
        print(f"   Hauteur : {height}")
        print(f"   Stride  : {stride_bytes} bytes")

        # Calculer la largeur réelle
        # SRGGB12 peut être packed (1.5 bytes par pixel) ou unpacked (2 bytes par pixel)
        width_packed = (stride_bytes * 2) // 3  # Si packed 12-bit
        width_unpacked = stride_bytes // 2       # Si unpacked (12-bit dans 16-bit)

        print(f"   Si packed   : largeur = {width_packed} pixels")
        print(f"   Si unpacked : largeur = {width_unpacked} pixels")
        print(f"   Attendu     : {sensor_res[0]} pixels")

        # Déterminer le bon format
        if abs(width_unpacked - sensor_res[0]) < 100:
            # Format unpacked (12-bit dans 16-bit container)
            print("\n   ✅ Format détecté : UNPACKED (12-bit dans 16-bit)")

            # Convertir en uint16
            raw_uint16 = raw_12bit.view(np.uint16)
            raw_image = raw_uint16[:, :sensor_res[0]]  # Prendre seulement la largeur réelle

        elif abs(width_packed - sensor_res[0]) < 100:
            # Format packed (vrai 12-bit packed)
            print("\n   ✅ Format détecté : PACKED (vrai 12-bit)")
            print("   ⚠️  Unpacking nécessaire...")

            # Unpacking 12-bit packed : 2 pixels = 3 bytes
            # [AAAAAAAA][AAAABBBB][BBBBBBBB]
            # Pixel A = byte0 + 4 bits MSB de byte1
            # Pixel B = 4 bits LSB de byte1 + byte2

            width = sensor_res[0]
            raw_image = np.zeros((height, width), dtype=np.uint16)

            for i in range(0, width, 2):
                if i + 1 < width:
                    byte_idx = (i * 3) // 2
                    if byte_idx + 2 < stride_bytes:
                        # Premier pixel (12 bits)
                        raw_image[:, i] = (raw_12bit[:, byte_idx].astype(np.uint16) << 4) | \
                                         ((raw_12bit[:, byte_idx + 1].astype(np.uint16) >> 4) & 0x0F)
                        # Deuxième pixel (12 bits)
                        raw_image[:, i + 1] = ((raw_12bit[:, byte_idx + 1].astype(np.uint16) & 0x0F) << 8) | \
                                              raw_12bit[:, byte_idx + 2].astype(np.uint16)
        else:
            print("\n   ❌ Format non reconnu, tentative avec reshape simple...")
            raw_image = raw_12bit.view(np.uint16).reshape(height, -1)[:, :sensor_res[0]]

        print(f"\n   ✅ Image unpacked :")
        print(f"      Shape : {raw_image.shape}")
        print(f"      dtype : {raw_image.dtype}")
        print(f"      Min/Max : {raw_image.min()} / {raw_image.max()}")
        print(f"      Profondeur réelle : {np.log2(raw_image.max() + 1):.1f} bits")

        # ====================================================================
        # TEST 3 : Debayerisation avec pattern RGGB
        # ====================================================================
        print("\n" + "="*70)
        print("TEST 3 : DEBAYERISATION - Application pattern RGGB")
        print("="*70)

        print("\n🎨 Test des 4 patterns Bayer...")

        patterns = [
            (cv2.COLOR_BayerRG2RGB, "RGGB", "Pattern correct IMX585"),
            (cv2.COLOR_BayerGR2RGB, "GRBG", "Pattern alternatif 1"),
            (cv2.COLOR_BayerGB2RGB, "GBRG", "Pattern alternatif 2"),
            (cv2.COLOR_BayerBG2RGB, "BGGR", "Pattern alternatif 3"),
        ]

        debayer_results = {}

        for pattern_code, pattern_name, description in patterns:
            try:
                print(f"\n   🔬 Test pattern {pattern_name} ({description})...")

                # Convertir 12-bit → 8-bit pour debayerisation OpenCV
                # Méthode 1 : Division simple (perd la dynamique)
                raw_8bit_simple = (raw_image / 16).astype(np.uint8)

                # Méthode 2 : Étirement auto (préserve la dynamique)
                raw_stretched = np.clip((raw_image - raw_image.min()) * 255.0 /
                                       (raw_image.max() - raw_image.min()), 0, 255).astype(np.uint8)

                # Debayerisation
                rgb_simple = cv2.cvtColor(raw_8bit_simple, pattern_code)
                rgb_stretched = cv2.cvtColor(raw_stretched, pattern_code)

                print(f"      ✅ Débayer OK : {rgb_simple.shape}")

                # Sauvegarder les images
                simple_path = test_dir / f"debayer_{pattern_name}_simple.png"
                stretched_path = test_dir / f"debayer_{pattern_name}_stretched.png"

                cv2.imwrite(str(simple_path), cv2.cvtColor(rgb_simple, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(stretched_path), cv2.cvtColor(rgb_stretched, cv2.COLOR_RGB2BGR))

                print(f"      💾 Sauvegardé : {simple_path.name}")
                print(f"      💾 Sauvegardé : {stretched_path.name}")

                debayer_results[pattern_name] = {
                    'simple': rgb_simple,
                    'stretched': rgb_stretched,
                    'path_simple': simple_path,
                    'path_stretched': stretched_path
                }

            except Exception as e:
                print(f"      ❌ Erreur : {e}")

        # ====================================================================
        # TEST 4 : Comparaison avec RGB888 de référence
        # ====================================================================
        print("\n" + "="*70)
        print("TEST 4 : VALIDATION - Comparaison avec RGB888")
        print("="*70)

        # Sauvegarder RGB référence
        rgb_ref_path = test_dir / "reference_RGB888.png"
        cv2.imwrite(str(rgb_ref_path), cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2BGR))
        print(f"\n💾 RGB888 référence : {rgb_ref_path.name}")

        # Comparer les couleurs moyennes
        print("\n📊 Comparaison des couleurs moyennes :")
        rgb_ref_mean = rgb_ref.mean(axis=(0,1))
        print(f"   RGB888 référence : R={rgb_ref_mean[0]:.1f}, G={rgb_ref_mean[1]:.1f}, B={rgb_ref_mean[2]:.1f}")

        for pattern_name, results in debayer_results.items():
            rgb_mean = results['stretched'].mean(axis=(0,1))
            diff = np.abs(rgb_mean - rgb_ref_mean)
            print(f"   {pattern_name:4s} stretched  : R={rgb_mean[0]:.1f}, G={rgb_mean[1]:.1f}, B={rgb_mean[2]:.1f} "
                  f"(Δ={diff.mean():.1f})")

        # ====================================================================
        # TEST 5 : Performance
        # ====================================================================
        print("\n" + "="*70)
        print("TEST 5 : PERFORMANCE - Vitesse de debayerisation")
        print("="*70)

        print("\n⏱️  Test de vitesse (20 frames)...")

        times = {
            'capture_rgb': [],
            'capture_raw': [],
            'unpack': [],
            'debayer_8bit': [],
            'debayer_16bit': []
        }

        for i in range(20):
            # Capture RGB
            t0 = time.time()
            _ = test_picam2.capture_array("main")
            times['capture_rgb'].append(time.time() - t0)

            # Capture RAW
            t0 = time.time()
            raw = test_picam2.capture_array("raw")
            times['capture_raw'].append(time.time() - t0)

            # Unpack
            t0 = time.time()
            raw_uint16 = raw.view(np.uint16)[:, :sensor_res[0]]
            times['unpack'].append(time.time() - t0)

            # Debayer 8-bit
            t0 = time.time()
            raw_8bit = (raw_uint16 / 16).astype(np.uint8)
            _ = cv2.cvtColor(raw_8bit, cv2.COLOR_BayerRG2RGB)
            times['debayer_8bit'].append(time.time() - t0)

            # Debayer 16-bit (si possible)
            # Note: OpenCV ne supporte pas directement 16-bit, on simule
            t0 = time.time()
            raw_stretched = np.clip((raw_uint16 - raw_uint16.min()) * 255.0 /
                                   (raw_uint16.max() - raw_uint16.min()), 0, 255).astype(np.uint8)
            _ = cv2.cvtColor(raw_stretched, cv2.COLOR_BayerRG2RGB)
            times['debayer_16bit'].append(time.time() - t0)

        print("\n📊 Résultats (moyenne sur 20 frames) :")
        print(f"   Capture RGB888     : {np.mean(times['capture_rgb'])*1000:.1f} ms ({1.0/np.mean(times['capture_rgb']):.1f} fps)")
        print(f"   Capture SRGGB12    : {np.mean(times['capture_raw'])*1000:.1f} ms ({1.0/np.mean(times['capture_raw']):.1f} fps)")
        print(f"   Unpack 12→16 bit   : {np.mean(times['unpack'])*1000:.1f} ms")
        print(f"   Debayer simple     : {np.mean(times['debayer_8bit'])*1000:.1f} ms")
        print(f"   Debayer stretched  : {np.mean(times['debayer_16bit'])*1000:.1f} ms")

        total_raw_pipeline = np.mean(times['capture_raw']) + np.mean(times['unpack']) + np.mean(times['debayer_8bit'])
        total_rgb_pipeline = np.mean(times['capture_rgb'])

        print(f"\n💡 Pipeline complet :")
        print(f"   RGB888 direct      : {total_rgb_pipeline*1000:.1f} ms ({1.0/total_rgb_pipeline:.1f} fps)")
        print(f"   RAW12→RGB pipeline : {total_raw_pipeline*1000:.1f} ms ({1.0/total_raw_pipeline:.1f} fps)")
        print(f"   Overhead RAW       : {(total_raw_pipeline/total_rgb_pipeline - 1)*100:.1f}%")

        if total_raw_pipeline < 1.0/25:  # < 40ms = > 25 fps
            print(f"\n   ✅ Performance suffisante pour Lucky Imaging (>25 fps) !")
        else:
            print(f"\n   ⚠️  Performance limite pour Lucky Imaging (<25 fps)")

        test_picam2.stop()
        test_picam2.close()

        # ====================================================================
        # RÉSUMÉ FINAL
        # ====================================================================
        print("\n" + "="*70)
        print("✅ RÉSUMÉ DES TESTS")
        print("="*70)

        print(f"\n📁 Images de test sauvegardées dans : {test_dir.absolute()}")
        print(f"\n💡 Pour vérifier visuellement :")
        print(f"   1. Ouvrez les images debayer_RGGB_*.png")
        print(f"   2. Comparez avec reference_RGB888.png")
        print(f"   3. Les couleurs doivent correspondre (ciel bleu, végétation verte)")
        print(f"   4. debayer_RGGB doit être la meilleure correspondance")

        print(f"\n🎯 Recommandation :")
        if 1.0/total_raw_pipeline >= 25:
            print(f"   ✅ SRGGB12 viable pour Lucky/Live Stack !")
            print(f"   ✅ Profondeur 12-bit préservée (vs 8-bit RGB)")
            print(f"   ✅ Dynamique 15× supérieure (4096 vs 256 niveaux)")
        else:
            print(f"   ⚠️  SRGGB12 plus lent que souhaité")
            print(f"   💡 Options : SRGGB10_CSI2P ou résolution réduite")

        print("\n" + "="*70)
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrompus")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Vérifier si mode test activé
if len(sys.argv) > 1:
    if sys.argv[1] == '--test-raw':
        test_raw_modes()
    elif sys.argv[1] == '--test-debayer':
        test_debayer_srggb12()
    elif sys.argv[1] == '--test-16bit':
        test_bit_depth_16()

pygame.init()

if frame == 1:
    if fullscreen == 1:
        windowSurfaceObj = pygame.display.set_mode((preview_width + bw,dis_height),pygame.FULLSCREEN, 24)
    elif fullscreen == 0:
        windowSurfaceObj = pygame.display.set_mode((preview_width + bw,dis_height),0,24)

else:
    windowSurfaceObj = pygame.display.set_mode((preview_width + bw,dis_height), pygame.NOFRAME,24)

Camera_Version()

global greyColor, redColor, greenColor, blueColor, dgryColor, lgrnColor, blackColor, whiteColor, purpleColor, yellowColor,lpurColor,lyelColor
# Palette de couleurs sobres et professionnelles
bredColor =   pygame.Color(180,  60,  60)  # Rouge atténué
lgrnColor =   pygame.Color(140, 160, 145)  # Vert grisâtre doux
lpurColor =   pygame.Color(150, 140, 160)  # Mauve discret
lyelColor =   pygame.Color(160, 155, 135)  # Beige/taupe
blackColor =  pygame.Color( 20,  20,  25)  # Noir légèrement bleuté
whiteColor =  pygame.Color(230, 230, 235)  # Blanc cassé
greyColor =   pygame.Color(110, 115, 120)  # Gris moyen neutre
dgryColor =   pygame.Color( 45,  48,  52)  # Gris foncé moderne
greenColor =  pygame.Color( 80, 160,  90)  # Vert modéré
purpleColor = pygame.Color(140, 100, 150)  # Violet sobre
yellowColor = pygame.Color(200, 180,  80)  # Jaune moutarde
blueColor =   pygame.Color(200, 100,  40)  # Orange foncé
redColor =    pygame.Color(160,  70,  70)  # Rouge brique
navyColor =   pygame.Color( 20,  40, 100)  # Bleu marine pour Lucky Stack

def button(col,row,bkgnd_Color,border_Color):
    global preview_width,bw,bh,alt_dis,preview_height,menu
    colors = [greyColor, dgryColor,yellowColor,purpleColor,greenColor,whiteColor,lgrnColor,lpurColor,lyelColor,blueColor,navyColor]
    Color = colors[bkgnd_Color]
    bx = preview_width + (col * bw)
    by = row * bh
    # but = pygame.image.load("button.jpg")  # Image des boutons désactivée
    pygame.draw.rect(windowSurfaceObj,Color,Rect(bx+1,by,bw-2,bh))
    pygame.draw.line(windowSurfaceObj,whiteColor,(bx,by),(bx,by+bh-1),2)
    pygame.draw.line(windowSurfaceObj,whiteColor,(bx,by),(bx+bw-1,by),1)
    pygame.draw.line(windowSurfaceObj,dgryColor,(bx,by+bh-1),(bx+bw-1,by+bh-1),1)
    pygame.draw.line(windowSurfaceObj,dgryColor,(bx+bw-2,by),(bx+bw-2,by+bh),2)
    # Images des boutons (Still, Video, Timelapse) désactivées pour un look plus épuré
    # if menu == 0 and row < 3:
    #     windowSurfaceObj.blit(but, (preview_width + 2,by + 2))
    pygame.display.update()

def ms_to_shutter_index(ms_value):
    """
    Convertit une valeur en millisecondes vers l'index le plus proche dans la liste shutters.

    Args:
        ms_value: Temps d'exposition en millisecondes

    Returns:
        L'index dans la liste shutters correspondant à la valeur la plus proche
    """
    global shutters, max_speed

    # Convertir ms en secondes
    seconds = ms_value / 1000.0

    best_index = 0
    best_diff = float('inf')

    for i in range(min(len(shutters), max_speed + 1)):
        shutter_val = shutters[i]
        # Convertir la valeur shutters en secondes
        if shutter_val < 0:
            shutter_seconds = abs(1.0 / shutter_val)
        else:
            shutter_seconds = shutter_val

        diff = abs(shutter_seconds - seconds)
        if diff < best_diff:
            best_diff = diff
            best_index = i

    return best_index

def shutter_index_to_ms(index):
    """
    Convertit un index shutters en millisecondes.

    Args:
        index: Index dans la liste shutters

    Returns:
        Temps d'exposition en millisecondes
    """
    global shutters

    if index < 0 or index >= len(shutters):
        return 0

    shutter_val = shutters[index]
    if shutter_val < 0:
        seconds = abs(1.0 / shutter_val)
    else:
        seconds = shutter_val

    return seconds * 1000.0

def get_sspeed():
    """
    Retourne la valeur d'exposition en microsecondes.
    Utilise custom_sspeed si défini (> 0), sinon calcule depuis shutters[speed].

    Returns:
        Temps d'exposition en microsecondes (int)
    """
    global custom_sspeed, speed, shutters

    if custom_sspeed > 0:
        return custom_sspeed
    else:
        shutter = shutters[speed]
        if shutter < 0:
            shutter = abs(1/shutter)
        sspeed_val = int(shutter * 1000000)
        if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
            sspeed_val += 1
        return sspeed_val

def update_sspeed():
    """
    Met à jour la variable globale sspeed en utilisant get_sspeed().
    Appeler cette fonction après avoir changé speed ou custom_sspeed.
    """
    global sspeed
    sspeed = get_sspeed()

def numeric_input_dialog(title, current_value, min_value, max_value):
    """
    Affiche une boîte de dialogue avec clavier virtuel pour saisir une valeur numérique.
    Fonctionne sans clavier physique.

    Args:
        title: Titre de la boîte de dialogue
        current_value: Valeur actuelle à afficher
        min_value: Valeur minimale autorisée
        max_value: Valeur maximale autorisée

    Returns:
        La nouvelle valeur saisie, ou None si annulé
    """
    global windowSurfaceObj, whiteColor, blackColor, greyColor, dgryColor, greenColor, redColor

    # Dimensions de la boîte de dialogue (augmentée pour éviter superposition)
    dialog_width = 320
    dialog_height = 420
    dialog_x = (windowSurfaceObj.get_width() - dialog_width) // 2
    dialog_y = (windowSurfaceObj.get_height() - dialog_height) // 2

    # Sauvegarder l'écran actuel
    screen_backup = windowSurfaceObj.copy()

    # Valeur en cours de saisie
    input_text = str(current_value)

    # Disposition du clavier numérique (touches plus compactes)
    keys = [
        ['7', '8', '9'],
        ['4', '5', '6'],
        ['1', '2', '3'],
        ['0', '.', 'Del']
    ]

    key_width = 70
    key_height = 45
    key_margin = 8
    keyboard_y = dialog_y + 100

    # Police pour l'affichage
    try:
        font_large = pygame.font.Font('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
        font_medium = pygame.font.Font('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
        font_small = pygame.font.Font('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except:
        font_large = pygame.font.Font(None, 32)
        font_medium = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 18)

    running = True
    result = None

    while running:
        # Fond semi-transparent
        s = pygame.Surface((windowSurfaceObj.get_width(), windowSurfaceObj.get_height()))
        s.set_alpha(180)
        s.fill((0, 0, 0))
        windowSurfaceObj.blit(screen_backup, (0, 0))
        windowSurfaceObj.blit(s, (0, 0))

        # Boîte de dialogue principale
        pygame.draw.rect(windowSurfaceObj, dgryColor, (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(windowSurfaceObj, whiteColor, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

        # Titre
        title_surface = font_medium.render(title, True, whiteColor)
        title_rect = title_surface.get_rect(center=(dialog_x + dialog_width // 2, dialog_y + 25))
        windowSurfaceObj.blit(title_surface, title_rect)

        # Affichage de la valeur saisie (plus compact)
        input_bg_rect = pygame.Rect(dialog_x + 20, dialog_y + 50, dialog_width - 40, 40)
        pygame.draw.rect(windowSurfaceObj, blackColor, input_bg_rect)
        pygame.draw.rect(windowSurfaceObj, greenColor, input_bg_rect, 2)

        input_surface = font_medium.render(input_text if input_text else "0", True, whiteColor)
        input_rect = input_surface.get_rect(center=input_bg_rect.center)
        windowSurfaceObj.blit(input_surface, input_rect)

        # Dessiner le clavier numérique
        key_buttons = []
        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                x = dialog_x + (dialog_width - (len(row) * key_width + (len(row) - 1) * key_margin)) // 2 + col_idx * (key_width + key_margin)
                y = keyboard_y + row_idx * (key_height + key_margin)

                key_rect = pygame.Rect(x, y, key_width, key_height)
                key_buttons.append((key_rect, key))

                # Couleur du bouton
                if key == 'Del':
                    color = redColor
                else:
                    color = greyColor

                pygame.draw.rect(windowSurfaceObj, color, key_rect)
                pygame.draw.rect(windowSurfaceObj, whiteColor, key_rect, 2)

                key_surface = font_medium.render(key, True, whiteColor)
                key_text_rect = key_surface.get_rect(center=key_rect.center)
                windowSurfaceObj.blit(key_surface, key_text_rect)

        # Boutons OK et Annuler (positionnés sous le clavier)
        button_y = dialog_y + dialog_height - 70
        button_width = 120
        button_spacing = 20
        ok_rect = pygame.Rect(dialog_x + (dialog_width // 2) - button_width - (button_spacing // 2), button_y, button_width, 45)
        cancel_rect = pygame.Rect(dialog_x + (dialog_width // 2) + (button_spacing // 2), button_y, button_width, 45)

        pygame.draw.rect(windowSurfaceObj, greenColor, ok_rect)
        pygame.draw.rect(windowSurfaceObj, whiteColor, ok_rect, 2)
        ok_surface = font_small.render("OK", True, whiteColor)
        ok_text_rect = ok_surface.get_rect(center=ok_rect.center)
        windowSurfaceObj.blit(ok_surface, ok_text_rect)

        pygame.draw.rect(windowSurfaceObj, redColor, cancel_rect)
        pygame.draw.rect(windowSurfaceObj, whiteColor, cancel_rect, 2)
        cancel_surface = font_small.render("Annuler", True, whiteColor)
        cancel_text_rect = cancel_surface.get_rect(center=cancel_rect.center)
        windowSurfaceObj.blit(cancel_surface, cancel_text_rect)

        pygame.display.flip()

        # Gestion des événements
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                result = None
            elif event.type == MOUSEBUTTONUP:
                mouse_pos = event.pos

                # Vérifier les touches du clavier
                for key_rect, key in key_buttons:
                    if key_rect.collidepoint(mouse_pos):
                        if key == 'Del':
                            input_text = input_text[:-1] if input_text else ""
                        elif key == '.':
                            if '.' not in input_text:
                                input_text += key
                        else:
                            input_text += key
                        break

                # Vérifier le bouton OK
                if ok_rect.collidepoint(mouse_pos):
                    try:
                        value = float(input_text) if input_text else current_value
                        if min_value <= value <= max_value:
                            result = value
                            running = False
                        else:
                            # Valeur hors limites, afficher un message rapide
                            input_text = str(current_value)
                    except ValueError:
                        input_text = str(current_value)

                # Vérifier le bouton Annuler
                if cancel_rect.collidepoint(mouse_pos):
                    running = False
                    result = None

    # Restaurer l'écran
    windowSurfaceObj.blit(screen_backup, (0, 0))
    pygame.display.flip()

    return result

def text(col,row,fColor,top,upd,msg,fsize,bkgnd_Color):
    global bh,preview_width,fv,tduration,menu
    colors =  [dgryColor, greenColor, yellowColor, redColor, purpleColor, blueColor, whiteColor, greyColor, blackColor, purpleColor,lgrnColor,lpurColor,lyelColor]
    Color  =  colors[fColor]
    bColor =  colors[bkgnd_Color]
    bx = preview_width + (col * bw)
    by = row * bh
    if menu == 0 and row < 3:
        by +=10
    global _font_cache
    
    # Utiliser le cache des polices (clé = taille) - optimisation performance
    cache_key = int(fsize)
    if cache_key not in _font_cache:
        # Polices modernes en ordre de préférence
        modern_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'
        ]
        
        for font_path in modern_fonts:
            if os.path.exists(font_path):
                _font_cache[cache_key] = pygame.font.Font(font_path, cache_key)
                break
        
        if cache_key not in _font_cache:
            _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    
    fontObj = _font_cache[cache_key]
    msgSurfaceObj = fontObj.render(msg, False, Color)
    msgRectobj = msgSurfaceObj.get_rect()
    if top == 0:
        if menu != 0:
             pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+1,by+int(bh/3),bw-2,int(bh/3)))
        msgRectobj.topleft = (bx + 5, by + int(bh/3)-int(preview_width/640))
    elif msg == "Config":
        if menu != 0:
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+1,by+int(bh/1.5),int(bw/2)-1,int(bh/3)-1))
        msgRectobj.topleft = (bx+5,  by + int(bh/1.5)-1)
    elif top == 1:
        if menu != 0 :
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+20,by+int(bh/1.5)-1,int(bw-20)-1,int(bh/3)))
        elif timelapse == 1:
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+20,by+int(bh/1.5)-1,int(bw-101),int(bh/5)))
        elif video == 1 or stream == 1:
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+20,by+int(bh/1.5)-1,int(bw-61)-1,int(bh/5)))
        msgRectobj.topleft = (bx + 20, by + int(bh/1.5)-int(preview_width/640)-1) 
    elif top == 2:
        if bkgnd_Color == 1:
            pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,row * fsize,preview_width,fv*2)) 
        msgRectobj.topleft = (0,row * fsize)
    windowSurfaceObj.blit(msgSurfaceObj, msgRectobj)
    if upd == 1 and top == 2:
        pygame.display.update(0,0,preview_width,fv*2)
    if upd == 1:
        pygame.display.update(bx, by, bw, bh)

def draw_bar(col,row,color,msg,value):
    global bw,bh,preview_width,still_limits,max_speed,v3_mag
    for f in range(0,len(still_limits)-1,3):
        if still_limits[f] == msg:
            pmin = still_limits[f+1]
            pmax = still_limits[f+2]
    if msg == "speed":
        pmax = max_speed

    # Pour le gain non-linéaire (IMX585), convertir gain → position slider
    display_value = value
    display_mag = mag
    if msg == "gain" and pmax > 1000 and value > 0:
        display_value = gain_to_slider_nonlinear(value, pmax)
        display_mag = gain_to_slider_nonlinear(mag, pmax)

    pygame.draw.rect(windowSurfaceObj,color,Rect(preview_width + col*bw,(row * bh) + 1,bw-2,int(bh/3)))
    if pmin > -1:
        j = display_value / (pmax - pmin)  * bw
        jag = display_mag / (pmax - pmin) * bw
    else:
        j = int(bw/2) + (display_value / (pmax - pmin)  * bw)
    j = min(j,bw-5)
    pygame.draw.rect(windowSurfaceObj,(80,140,90),Rect(int(preview_width + int(col*bw) + 2),int(row * bh)+1,int(j+1),int(bh/3)))
    if msg == "gain" and value > mag:
        pygame.draw.rect(windowSurfaceObj,(180,160,70),Rect(int(preview_width + int(col*bw) + 2 + jag),int(row * bh),int(j+1 - jag),int(bh/3)))
    pygame.draw.rect(windowSurfaceObj,(120,90,130),Rect(int(preview_width + int(col*bw) + j ),int(row * bh)+1,3,int(bh/3)))
    # Dessiner l'icône de main pour les sliders EV, SPEED et GAIN
    if msg in ["ev", "speed", "gain"]:
        draw_hand_icon(col, row)
    pygame.display.update()

def draw_hand_icon(col, row):
    """
    Dessine une icône de main au centre de la zone de click sous un slider.
    Cette zone permet d'activer le pavé numérique pour saisir une valeur.
    La main est dessinée dans la zone centrale (entre "-" et "+").
    """
    global bw, bh, preview_width, _font_cache

    # Position de la zone centrale (entre "-" et "+")
    # La zone de click est divisée en 3 parties: gauche (-), centre (main), droite (+)
    bx = preview_width + (col * bw)
    by = row * bh

    # Zone de l'icône : au centre de la zone bh/3 à bh*2/3
    # La zone de click totale va de bh/3 à bh*2/3 (hauteur bh/3)
    icon_y = by + int(bh/3) + int(bh/12)  # Un peu en dessous du haut de la zone de click
    icon_x = bx + int(bw/2) - int(bh/12)  # Centré horizontalement

    # Taille de l'icône basée sur bh
    icon_size = int(bh/6)

    # Dessiner un petit rectangle de fond pour l'icône
    icon_rect = pygame.Rect(icon_x - 2, icon_y - 2, icon_size + 4, icon_size + 4)
    pygame.draw.rect(windowSurfaceObj, (60, 65, 70), icon_rect)  # Fond légèrement plus clair
    pygame.draw.rect(windowSurfaceObj, (100, 110, 120), icon_rect, 1)  # Bordure

    # Dessiner le symbole de la main (utiliser un caractère unicode ou dessiner simplement)
    # On utilise le caractère unicode ✋ ou on dessine une main stylisée
    cache_key = int(icon_size + 2)
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)

    fontObj = _font_cache[cache_key]
    # Utiliser le symbole "☰" (menu) ou "#" comme alternative simple
    hand_surface = fontObj.render("#", True, (200, 200, 100))  # Jaune clair
    hand_rect = hand_surface.get_rect(center=icon_rect.center)
    windowSurfaceObj.blit(hand_surface, hand_rect)

def draw_Vbar(col,row,color,msg,value):
    global bw,bh,preview_width,video_limits,livestack_limits
    pmin = None
    pmax = None
    # Chercher d'abord dans video_limits
    for f in range(0,len(video_limits)-1,3):
        if video_limits[f] == msg:
            pmin = video_limits[f+1]
            pmax = video_limits[f+2]
            break
    # Si pas trouvé, chercher dans livestack_limits
    if pmin is None:
        for f in range(0,len(livestack_limits)-1,3):
            if livestack_limits[f] == msg:
                pmin = livestack_limits[f+1]
                pmax = livestack_limits[f+2]
                break

    # Pour vformat, utiliser les résolutions natives uniquement
    if msg == "vformat":
        native_vformats = get_native_vformats()
        if len(native_vformats) > 0:
            # Convertir vformat (index global) en position dans la liste native
            if value in native_vformats:
                native_idx = native_vformats.index(value)
                pmin = 0
                pmax = len(native_vformats) - 1
                value = native_idx
            else:
                # Si vformat n'est pas natif, trouver le plus proche
                pmin = 0
                pmax = len(native_vformats) - 1
                value = 0  # Par défaut première résolution
        else:
            # Fallback ancien système si aucune résolution native
            pmax = max_vformat

    if alt_dis == 0:
        pygame.draw.rect(windowSurfaceObj,color,Rect(preview_width + col*bw,(row * bh) +1,bw-2,int(bh/3)))
    else:
        if row < 8:
            if alt_dis == 1:
                 pygame.draw.rect(windowSurfaceObj,color,Rect(row*bw,preview_height + (bh*2),bw-1,int(bh/3)))
            else:
                 pygame.draw.rect(windowSurfaceObj,color,Rect(row*bw,int((preview_height *.75) + (bh*2)),bw-1,int(bh/3)))
        else:
            if alt_dis == 1:
                pygame.draw.rect(windowSurfaceObj,color,Rect((row-8)*bw,preview_height + (bh*3),bw-1,int(bh/3)))
            else:
                pygame.draw.rect(windowSurfaceObj,color,Rect((row-8)*bw,int((preview_height *.75) + (bh*3)),bw-1,int(bh/3)))
    if pmin > -1:
        j = value / (pmax - pmin)  * bw
    else:
        j = int(bw/2) + (value / (pmax - pmin)  * bw)
    j = min(j,bw-5)
    pygame.draw.rect(windowSurfaceObj,(120,110,130),Rect(int(preview_width + (col*bw) + 2),int(row * bh)+1,int(j+1),int(bh/3)))
    pygame.draw.rect(windowSurfaceObj,(120,90,130),Rect(int(preview_width + (col*bw) + j ),int(row * bh)+1,3,int(bh/3)))

    pygame.display.update()

# ============================================================================
# STRETCH FULLSCREEN CONTROLS - Mode ajustement interactif des paramètres
# ============================================================================

def draw_stretch_hand_icon(screen_width, screen_height, active=False, is_raw_mode=False):
    """
    Dessine l'icône de réglage en haut à droite de l'écran en mode stretch fullscreen.
    Cette icône permet d'activer/désactiver les contrôles de réglage.
    Le texte s'adapte selon le mode: "ISP" pour RAW, "ADJ" pour RGB/YUV.

    Args:
        screen_width: Largeur de l'écran
        screen_height: Hauteur de l'écran
        active: True si les contrôles sont actifs
        is_raw_mode: True si en mode RAW (affiche ISP), False pour RGB/YUV (affiche ADJ)
    """
    global windowSurfaceObj, _font_cache

    # Taille et position de l'icône (coin supérieur droit)
    icon_size = 50
    margin = 15
    icon_x = screen_width - icon_size - margin
    icon_y = margin

    # Couleur de fond selon l'état et le mode
    if active:
        if is_raw_mode:
            bg_color = (120, 80, 80)  # Rouge/brun si actif en mode RAW
            border_color = (180, 120, 120)
        else:
            bg_color = (80, 120, 80)  # Vert si actif en mode RGB/YUV
            border_color = (120, 180, 120)
    else:
        bg_color = (60, 60, 70)  # Gris si inactif
        border_color = (100, 100, 110)

    # Dessiner le fond arrondi (simulé par rect avec bordure)
    icon_rect = pygame.Rect(icon_x, icon_y, icon_size, icon_size)
    pygame.draw.rect(windowSurfaceObj, bg_color, icon_rect, border_radius=8)
    pygame.draw.rect(windowSurfaceObj, border_color, icon_rect, 2, border_radius=8)

    # Dessiner le texte selon le mode
    cache_key = 32
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    # Texte adapté au mode
    icon_text = "ISP" if is_raw_mode else "ADJ"
    text_color = (220, 220, 100) if active else (180, 180, 180)
    hand_text = fontObj.render(icon_text, True, text_color)
    hand_rect = hand_text.get_rect(center=icon_rect.center)
    windowSurfaceObj.blit(hand_text, hand_rect)

    return icon_rect  # Retourner le rect pour détecter les clics

def draw_stretch_slider(x, y, width, height, label, value, vmin, vmax, color=(100, 140, 180)):
    """
    Dessine un slider horizontal pour ajuster un paramètre stretch.

    Args:
        x, y: Position du slider
        width, height: Dimensions
        label: Texte du label
        value: Valeur actuelle
        vmin, vmax: Plage de valeurs
        color: Couleur du slider

    Returns:
        Rect du slider pour détection clics
    """
    global windowSurfaceObj, _font_cache

    # Fond semi-transparent
    slider_rect = pygame.Rect(x, y, width, height)
    s = pygame.Surface((width, height), pygame.SRCALPHA)
    s.fill((30, 30, 40, 200))
    windowSurfaceObj.blit(s, (x, y))

    # Label
    cache_key = 18
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    # Afficher label et valeur
    label_text = fontObj.render(f"{label}: {value:.2f}", True, (220, 220, 220))
    windowSurfaceObj.blit(label_text, (x + 5, y + 2))

    # Barre de progression
    bar_y = y + 20
    bar_height = height - 25
    pygame.draw.rect(windowSurfaceObj, (50, 50, 60), (x + 5, bar_y, width - 10, bar_height))

    # Position du curseur
    if vmax > vmin:
        ratio = (value - vmin) / (vmax - vmin)
    else:
        ratio = 0
    cursor_x = x + 5 + int(ratio * (width - 15))

    # Barre remplie
    pygame.draw.rect(windowSurfaceObj, color, (x + 5, bar_y, cursor_x - x - 5, bar_height))

    # Curseur
    pygame.draw.rect(windowSurfaceObj, (200, 200, 220), (cursor_x, bar_y - 2, 5, bar_height + 4))

    # Bordure
    pygame.draw.rect(windowSurfaceObj, (80, 80, 90), slider_rect, 1)

    return slider_rect

def draw_stretch_controls(screen_width, screen_height, image_array=None):
    """
    Dessine les contrôles de réglage stretch sur l'écran en mode plein écran.
    Affiche les sliders selon le preset actif (GHS ou Arcsinh).
    Inclut également les sliders de gain et exposition pour YUV/RGB.

    Args:
        screen_width: Largeur de l'écran
        screen_height: Hauteur de l'écran
        image_array: Image numpy pour l'histogramme (optionnel)

    Returns:
        dict avec les rects des sliders pour détection des clics
    """
    global stretch_preset, ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP
    global stretch_factor, stretch_p_low, stretch_p_high
    global windowSurfaceObj, _font_cache
    global gain, ev, still_limits, custom_sspeed, sspeed

    slider_rects = {}

    # Dimensions des sliders
    slider_width = 250
    slider_height = 45
    margin = 10
    start_x = 20
    start_y = 60

    # Titre du mode actif
    cache_key = 24
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    # Calculer la position Y pour les sliders Exposure (à droite)
    exp_start_x = screen_width - slider_width - 40
    exp_start_y = 60

    if stretch_preset == 1:
        # Mode GHS
        title = "GHS Stretch"
        title_color = (100, 200, 150)

        title_text = fontObj.render(title, True, title_color)
        windowSurfaceObj.blit(title_text, (start_x, 20))

        # Sliders GHS
        slider_rects['ghs_D'] = draw_stretch_slider(
            start_x, start_y, slider_width, slider_height,
            "D (Stretch)", ghs_D / 10.0, -1.0, 10.0, (100, 180, 140)
        )

        slider_rects['ghs_b'] = draw_stretch_slider(
            start_x, start_y + slider_height + margin, slider_width, slider_height,
            "b (Local Int)", ghs_b / 10.0, -30.0, 10.0, (100, 160, 180)
        )

        slider_rects['ghs_SP'] = draw_stretch_slider(
            start_x, start_y + 2*(slider_height + margin), slider_width, slider_height,
            "SP (Symmetry)", ghs_SP / 100.0, 0.0, 1.0, (180, 140, 100)
        )

        slider_rects['ghs_LP'] = draw_stretch_slider(
            start_x, start_y + 3*(slider_height + margin), slider_width, slider_height,
            "LP (Shadows)", ghs_LP / 100.0, 0.0, 1.0, (140, 100, 180)
        )

        slider_rects['ghs_HP'] = draw_stretch_slider(
            start_x, start_y + 4*(slider_height + margin), slider_width, slider_height,
            "HP (Highlights)", ghs_HP / 100.0, 0.0, 1.0, (180, 100, 140)
        )

    elif stretch_preset == 2:
        # Mode Arcsinh
        title = "Arcsinh Stretch"
        title_color = (200, 150, 100)

        title_text = fontObj.render(title, True, title_color)
        windowSurfaceObj.blit(title_text, (start_x, 20))

        # Sliders Arcsinh
        slider_rects['stretch_factor'] = draw_stretch_slider(
            start_x, start_y, slider_width, slider_height,
            "Factor", stretch_factor / 10.0, 0.0, 8.0, (200, 160, 100)
        )

        slider_rects['stretch_p_low'] = draw_stretch_slider(
            start_x, start_y + slider_height + margin, slider_width, slider_height,
            "Clip Low %", stretch_p_low / 10.0, 0.0, 2.0, (180, 140, 100)
        )

        slider_rects['stretch_p_high'] = draw_stretch_slider(
            start_x, start_y + 2*(slider_height + margin), slider_width, slider_height,
            "Clip High %", stretch_p_high / 100.0, 99.5, 100.0, (100, 180, 140)
        )
    else:
        # Mode OFF - pas de contrôles
        title = "Stretch OFF"
        title_color = (150, 150, 150)
        title_text = fontObj.render(title, True, title_color)
        windowSurfaceObj.blit(title_text, (start_x, 20))

    # =========================================================================
    # SECTION EXPOSURE (Gain + Temps d'exposition) - À droite de l'écran pour YUV/RGB
    # =========================================================================
    # Titre Exposure
    cache_key_exp = 20
    if cache_key_exp not in _font_cache:
        _font_cache[cache_key_exp] = pygame.font.Font(None, cache_key_exp)
    expFont = _font_cache[cache_key_exp]

    exp_title = expFont.render("Exposure (YUV/RGB)", True, (200, 180, 100))
    windowSurfaceObj.blit(exp_title, (exp_start_x, 20))

    # Slider Gain (0 = AUTO, 1-300 = valeur fixe)
    gain_max = 300  # Plage étendue pour astrophoto
    gain_label = "AUTO" if gain == 0 else f"{gain:.0f}"
    slider_rects['preview_gain'] = draw_stretch_slider(
        exp_start_x, exp_start_y, slider_width, slider_height,
        f"Gain ({gain_label})", float(gain), 0.0, float(gain_max), (180, 180, 100)
    )

    # Slider Temps d'exposition (0.001s à 20s) - échelle logarithmique
    # custom_sspeed est en microsecondes (1000 = 1ms, 1000000 = 1s)
    # Plage: 1000µs (1ms) à 20000000µs (20s)
    exp_min_us = 1000        # 1ms = 0.001s
    exp_max_us = 20000000    # 20s
    current_exp_us = custom_sspeed if custom_sspeed > 0 else sspeed
    current_exp_us = max(exp_min_us, min(exp_max_us, current_exp_us))

    # Afficher en format lisible
    if current_exp_us >= 1000000:
        exp_label = f"{current_exp_us / 1000000:.1f}s"
    elif current_exp_us >= 1000:
        exp_label = f"{current_exp_us / 1000:.0f}ms"
    else:
        exp_label = f"{current_exp_us}µs"

    # Position logarithmique pour le slider (0 à 1)
    import math
    log_min = math.log10(exp_min_us)
    log_max = math.log10(exp_max_us)
    log_current = math.log10(current_exp_us)
    exp_ratio = (log_current - log_min) / (log_max - log_min)

    slider_rects['preview_exposure'] = draw_stretch_slider(
        exp_start_x, exp_start_y + slider_height + margin, slider_width, slider_height,
        f"Expo ({exp_label})", exp_ratio, 0.0, 1.0, (100, 180, 180)
    )

    # Dessiner l'histogramme en bas de l'écran
    if image_array is not None:
        draw_stretch_histogram(screen_width, screen_height, image_array)

    return slider_rects

def draw_stretch_histogram(screen_width, screen_height, image_array):
    """
    Dessine l'histogramme de l'image en bas de l'écran en mode stretch.
    Version optimisée avec lissage et rendu rapide.
    Affiche également la courbe de transformation (GHS ou Arcsinh) en violet.

    Args:
        screen_width: Largeur de l'écran
        screen_height: Hauteur de l'écran
        image_array: Image numpy (H, W, 3) en RGB
    """
    global windowSurfaceObj
    global stretch_preset, stretch_factor, stretch_p_low, stretch_p_high
    global ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP

    # Dimensions de l'histogramme
    hist_height = 120
    hist_y = screen_height - hist_height - 10
    hist_width = screen_width - 40
    hist_x = 20

    # Fond semi-transparent
    s = pygame.Surface((hist_width, hist_height), pygame.SRCALPHA)
    s.fill((20, 20, 30, 220))
    windowSurfaceObj.blit(s, (hist_x, hist_y))

    # Bordure
    pygame.draw.rect(windowSurfaceObj, (80, 80, 100), (hist_x, hist_y, hist_width, hist_height), 1)

    try:
        # OPTIMISATION 1: Sous-échantillonner l'image (1 pixel sur 4)
        # Réduit drastiquement le temps de calcul sans perte de précision visible
        h, w = image_array.shape[:2]
        step = max(1, min(h, w) // 256)  # Adapter le pas à la taille de l'image
        img_sampled = image_array[::step, ::step]

        # S'assurer que l'image est en uint8
        if img_sampled.dtype == np.float32:
            img_sampled = np.clip(img_sampled, 0, 255).astype(np.uint8)

        # OPTIMISATION 2: Utiliser 64 bins au lieu de 256 (plus lisse naturellement)
        num_bins = 64
        bins = np.linspace(0, 256, num_bins + 1)

        # Calculer les histogrammes avec moins de bins
        # ATTENTION: L'image a les canaux inversés pour compenser le swap pygame [:,:,[2,1,0]]
        # Canal 0 = Bleu (affiché), Canal 2 = Rouge (affiché)
        hist_r, _ = np.histogram(img_sampled[:,:,2].ravel(), bins=bins)  # Canal 2 = Rouge affiché
        hist_g, _ = np.histogram(img_sampled[:,:,1].ravel(), bins=bins)
        hist_b, _ = np.histogram(img_sampled[:,:,0].ravel(), bins=bins)  # Canal 0 = Bleu affiché

        # Luminance optimisée (calcul vectorisé sur image sous-échantillonnée)
        # Utiliser une approximation rapide: (R + G + G + B) / 4
        # Adapter aux canaux inversés: canal 2=R, canal 0=B
        gray = ((img_sampled[:,:,2].astype(np.uint16) +
                 img_sampled[:,:,1].astype(np.uint16) * 2 +
                 img_sampled[:,:,0].astype(np.uint16)) >> 2).astype(np.uint8)
        hist_l, _ = np.histogram(gray.ravel(), bins=bins)

        # OPTIMISATION 3: Lissage par moyenne mobile (kernel size 3)
        def smooth_hist(h):
            # Padding pour éviter les effets de bord
            padded = np.concatenate([[h[0]], h, [h[-1]]])
            # Moyenne mobile de 3 points
            return ((padded[:-2] + padded[1:-1] + padded[2:]) / 3.0)

        hist_r = smooth_hist(hist_r.astype(np.float32))
        hist_g = smooth_hist(hist_g.astype(np.float32))
        hist_b = smooth_hist(hist_b.astype(np.float32))
        hist_l = smooth_hist(hist_l.astype(np.float32))

        # Normaliser avec échelle logarithmique douce pour mieux voir les détails
        # Évite que les pics dominent complètement
        def normalize_log(h, max_height):
            h_log = np.log1p(h)  # log(1 + x) pour éviter log(0)
            max_val = h_log.max()
            if max_val > 0:
                return (h_log / max_val * max_height).astype(int)
            return np.zeros_like(h, dtype=int)

        draw_height = hist_height - 20
        hist_r = normalize_log(hist_r, draw_height)
        hist_g = normalize_log(hist_g, draw_height)
        hist_b = normalize_log(hist_b, draw_height)
        hist_l = normalize_log(hist_l, draw_height)

        # OPTIMISATION 4: Utiliser pygame.draw.lines (un seul appel par courbe)
        bin_width = hist_width / num_bins
        base_y = hist_y + hist_height - 10

        # Préparer les listes de points pour chaque canal
        x_coords = np.linspace(hist_x, hist_x + hist_width, num_bins).astype(int)

        # Fonction pour créer la liste de points
        def make_points(hist_data):
            return [(int(x_coords[i]), int(base_y - hist_data[i])) for i in range(num_bins)]

        # Dessiner les courbes avec pygame.draw.lines (beaucoup plus rapide)
        # Ordre: Luminance en fond, puis R, G, B
        points_l = make_points(hist_l)
        points_r = make_points(hist_r)
        points_g = make_points(hist_g)
        points_b = make_points(hist_b)

        if len(points_l) > 1:
            pygame.draw.lines(windowSurfaceObj, (180, 180, 180), False, points_l, 2)
        if len(points_r) > 1:
            pygame.draw.lines(windowSurfaceObj, (255, 80, 80), False, points_r, 2)
        if len(points_g) > 1:
            pygame.draw.lines(windowSurfaceObj, (80, 255, 80), False, points_g, 2)
        if len(points_b) > 1:
            pygame.draw.lines(windowSurfaceObj, (80, 80, 255), False, points_b, 2)

        # =====================================================================
        # COURBE DE TRANSFORMATION EN VIOLET (GHS ou Arcsinh)
        # =====================================================================
        if stretch_preset > 0:
            try:
                # Calculer la courbe de transformation sur 64 points (même résolution que l'histogramme)
                x_norm = np.linspace(0, 1, num_bins)  # Entrée normalisée [0, 1]
                y_transform = np.zeros(num_bins)

                if stretch_preset == 1:
                    # GHS - Generalized Hyperbolic Stretch
                    D = ghs_D / 10.0
                    b = ghs_b / 10.0
                    SP = ghs_SP / 100.0
                    LP = ghs_LP / 100.0
                    HP = ghs_HP / 100.0

                    epsilon = 1e-10
                    LP = max(0.0, min(LP, SP))
                    HP = max(SP, min(HP, 1.0))

                    if abs(D) >= epsilon:
                        # Fonction de transformation de base T(x)
                        def T_base_curve(x, D_val, b_val):
                            if abs(b_val - (-1.0)) < epsilon:
                                return np.log1p(D_val * x)
                            elif b_val < 0 and abs(b_val - (-1.0)) >= epsilon:
                                base = np.maximum(1.0 - b_val * D_val * x, epsilon)
                                exponent = (b_val + 1.0) / b_val
                                return (1.0 - np.power(base, exponent)) / (D_val * (b_val + 1.0))
                            elif abs(b_val) < epsilon:
                                return 1.0 - np.exp(-D_val * x)
                            elif abs(b_val - 1.0) < epsilon:
                                return 1.0 - 1.0 / (1.0 + D_val * x)
                            else:
                                base = np.maximum(1.0 + b_val * D_val * x, epsilon)
                                return 1.0 - np.power(base, -1.0 / b_val)

                        def T_prime_curve(x, D_val, b_val):
                            if abs(b_val - (-1.0)) < epsilon:
                                return D_val / (1.0 + D_val * x)
                            elif b_val < 0 and abs(b_val - (-1.0)) >= epsilon:
                                base = np.maximum(1.0 - b_val * D_val * x, epsilon)
                                return np.power(base, 1.0 / b_val)
                            elif abs(b_val) < epsilon:
                                return D_val * np.exp(-D_val * x)
                            elif abs(b_val - 1.0) < epsilon:
                                return D_val * np.power(1.0 + D_val * x, -2.0)
                            else:
                                base = np.maximum(1.0 + b_val * D_val * x, epsilon)
                                return D_val * np.power(base, -(1.0 + b_val) / b_val)

                        # Calculer les valeurs aux bornes
                        T2_LP = -T_base_curve(SP - LP, D, b)
                        T2_prime_LP = T_prime_curve(SP - LP, D, b)
                        T3_HP = T_base_curve(HP - SP, D, b)
                        T3_prime_HP = T_prime_curve(HP - SP, D, b)

                        T1_0 = T2_prime_LP * (0.0 - LP) + T2_LP
                        T4_1 = T3_prime_HP * (1.0 - HP) + T3_HP
                        norm_range = T4_1 - T1_0

                        if abs(norm_range) >= epsilon:
                            for i, x in enumerate(x_norm):
                                if x < LP:
                                    y = T2_prime_LP * (x - LP) + T2_LP
                                elif x < SP:
                                    y = -T_base_curve(SP - x, D, b)
                                elif x < HP:
                                    y = T_base_curve(x - SP, D, b)
                                else:
                                    y = T3_prime_HP * (x - HP) + T3_HP
                                y_transform[i] = (y - T1_0) / norm_range
                        else:
                            y_transform = x_norm.copy()
                    else:
                        y_transform = x_norm.copy()

                elif stretch_preset == 2:
                    # Arcsinh stretch
                    factor = stretch_factor / 10.0
                    if factor > 0.01:
                        # Éviter division par zéro si arcsinh(factor) est très petit
                        arcsinh_factor = np.arcsinh(factor)
                        if arcsinh_factor > 1e-10:
                            y_transform = np.arcsinh(x_norm * factor) / arcsinh_factor
                        else:
                            y_transform = x_norm.copy()
                    else:
                        y_transform = x_norm.copy()

                # Clip et création des points pour la courbe
                y_transform = np.clip(y_transform, 0, 1)

                # Convertir en coordonnées écran
                # x_coords est déjà défini pour l'histogramme
                curve_points = [(int(x_coords[i]), int(base_y - y_transform[i] * draw_height))
                               for i in range(num_bins)]

                # Dessiner la courbe de transformation en VIOLET (épaisseur 3 pour visibilité)
                if len(curve_points) > 1:
                    pygame.draw.lines(windowSurfaceObj, (180, 80, 255), False, curve_points, 3)

            except Exception as curve_error:
                # Afficher l'erreur pour debug
                print(f"[HISTOGRAM] Erreur courbe: {curve_error}")

    except Exception as e:
        # En cas d'erreur silencieuse
        pass

def handle_stretch_slider_click(mousex, mousey, slider_rects):
    """
    Gère le clic sur un slider stretch et met à jour la valeur correspondante.
    Inclut aussi les sliders de gain et ev pour YUV/RGB.

    Args:
        mousex, mousey: Position du clic
        slider_rects: Dict des rectangles des sliders

    Returns:
        True si un slider a été modifié, False sinon
    """
    global ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP
    global stretch_factor, stretch_p_low, stretch_p_high
    global gain, ev, still_limits, custom_sspeed

    for name, rect in slider_rects.items():
        if rect.collidepoint(mousex, mousey):
            # Calculer la position relative dans le slider
            rel_x = mousex - rect.x - 5
            slider_width = rect.width - 10
            ratio = max(0, min(1, rel_x / slider_width))

            # Mettre à jour la valeur selon le slider
            if name == 'ghs_D':
                # D: -1.0 à 10.0, stocké x10
                ghs_D = int(-10 + ratio * 110)
            elif name == 'ghs_b':
                # b: -30.0 à 10.0, stocké x10
                ghs_b = int(-300 + ratio * 400)
            elif name == 'ghs_SP':
                # SP: 0.0 à 1.0, stocké x100
                ghs_SP = int(ratio * 100)
            elif name == 'ghs_LP':
                # LP: 0.0 à 1.0, stocké x100
                ghs_LP = int(ratio * 100)
            elif name == 'ghs_HP':
                # HP: 0.0 à 1.0, stocké x100
                ghs_HP = int(ratio * 100)
            elif name == 'stretch_factor':
                # Factor: 0.0 à 8.0, stocké x10
                stretch_factor = int(ratio * 80)
                print(f"[ARCSINH] stretch_factor = {stretch_factor} (factor={stretch_factor/10.0})")
            elif name == 'stretch_p_low':
                # Clip low: 0.0 à 2.0, stocké x10
                stretch_p_low = int(ratio * 20)
                print(f"[ARCSINH] stretch_p_low = {stretch_p_low}")
            elif name == 'stretch_p_high':
                # Clip high: 99.5 à 100.0, stocké x100
                stretch_p_high = int(9950 + ratio * 50)
                print(f"[ARCSINH] stretch_p_high = {stretch_p_high}")
            # Sliders Gain et Exposition pour YUV/RGB
            elif name == 'preview_gain':
                # Gain: 0 à 300 (0 = AUTO)
                gain_max = 300  # Plage étendue pour astrophoto
                gain = int(ratio * gain_max)
                # Appliquer immédiatement les changements à la caméra
                apply_gain_exposure_to_camera()
            elif name == 'preview_exposure':
                # Exposition: échelle logarithmique de 1ms à 20s
                import math
                exp_min_us = 1000        # 1ms
                exp_max_us = 20000000    # 20s
                log_min = math.log10(exp_min_us)
                log_max = math.log10(exp_max_us)
                # Convertir le ratio (0-1) en valeur logarithmique
                log_value = log_min + ratio * (log_max - log_min)
                custom_sspeed = int(10 ** log_value)
                # Appliquer immédiatement les changements à la caméra
                apply_gain_exposure_to_camera()

            return True

    return False

def is_click_on_stretch_hand_icon(mousex, mousey, screen_width):
    """
    Vérifie si le clic est sur l'icône de main en mode stretch.

    Args:
        mousex, mousey: Position du clic
        screen_width: Largeur de l'écran

    Returns:
        True si clic sur l'icône, False sinon
    """
    icon_size = 50
    margin = 15
    icon_x = screen_width - icon_size - margin
    icon_y = margin

    return (icon_x <= mousex <= icon_x + icon_size and
            icon_y <= mousey <= icon_y + icon_size)

# ============================================================================
# FIN STRETCH FULLSCREEN CONTROLS
# ============================================================================

def draw_livestack_button(screen_width, screen_height, is_raw_mode=False):
    """
    Dessine le bouton Live Stack à côté de l'icône ISP/ADJ en mode stretch fullscreen.
    Position: à gauche de ISP (mode RAW) ou à gauche de ADJ (mode RGB/YUV)

    Args:
        screen_width: Largeur de l'écran
        screen_height: Hauteur de l'écran
        is_raw_mode: True si en mode RAW (bouton à gauche de ISP)

    Returns:
        Rect du bouton pour détecter les clics
    """
    global windowSurfaceObj, _font_cache

    # Taille et position du bouton
    icon_size = 50
    margin = 15
    adj_icon_x = screen_width - icon_size - margin  # Position ISP/ADJ
    icon_x = adj_icon_x - icon_size - 10  # STACK à gauche de ISP/ADJ
    icon_y = margin

    # Couleur du bouton (vert pour indiquer action de lancement)
    bg_color = (60, 90, 60)  # Vert foncé
    border_color = (100, 150, 100)
    text_color = (150, 220, 150)

    # Dessiner le fond arrondi
    icon_rect = pygame.Rect(icon_x, icon_y, icon_size, icon_size)
    pygame.draw.rect(windowSurfaceObj, bg_color, icon_rect, border_radius=8)
    pygame.draw.rect(windowSurfaceObj, border_color, icon_rect, 2, border_radius=8)

    # Dessiner le texte "STACK"
    cache_key = 20
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    stack_text = fontObj.render("STACK", True, text_color)
    stack_rect = stack_text.get_rect(center=icon_rect.center)
    windowSurfaceObj.blit(stack_text, stack_rect)

    return icon_rect


def is_click_on_livestack_button(mousex, mousey, screen_width):
    """
    Vérifie si le clic est sur le bouton Live Stack.

    Args:
        mousex, mousey: Position du clic
        screen_width: Largeur de l'écran

    Returns:
        True si clic sur le bouton, False sinon
    """
    icon_size = 50
    margin = 15
    adj_icon_x = screen_width - icon_size - margin  # Position ISP/ADJ
    icon_x = adj_icon_x - icon_size - 10  # STACK à gauche de ISP/ADJ
    icon_y = margin

    return (icon_x <= mousex <= icon_x + icon_size and
            icon_y <= mousey <= icon_y + icon_size)


# ============================================================================
# RAW FULLSCREEN CONTROLS - Mode ajustement ISP/Stretch en temps réel
# ============================================================================

def draw_raw_hand_icon(screen_width, screen_height, active=False):
    """
    Dessine l'icône "RAW" à gauche de l'icône ADJ en mode stretch fullscreen.
    Cette icône permet d'activer/désactiver le panneau de réglage RAW/ISP.

    Args:
        screen_width: Largeur de l'écran
        screen_height: Hauteur de l'écran
        active: True si le panneau RAW est actif

    Returns:
        Rect de l'icône pour détecter les clics
    """
    global windowSurfaceObj, _font_cache

    # Taille et position de l'icône (à gauche de ADJ)
    icon_size = 50
    margin = 15
    adj_icon_x = screen_width - icon_size - margin  # Position ADJ
    icon_x = adj_icon_x - icon_size - 10  # RAW à gauche de ADJ
    icon_y = margin

    # Couleur de fond selon l'état
    if active:
        bg_color = (120, 80, 80)  # Rouge si actif
        border_color = (180, 120, 120)
    else:
        bg_color = (60, 60, 70)  # Gris si inactif
        border_color = (100, 100, 110)

    # Dessiner le fond arrondi
    icon_rect = pygame.Rect(icon_x, icon_y, icon_size, icon_size)
    pygame.draw.rect(windowSurfaceObj, bg_color, icon_rect, border_radius=8)
    pygame.draw.rect(windowSurfaceObj, border_color, icon_rect, 2, border_radius=8)

    # Dessiner le texte "RAW"
    cache_key = 28
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    raw_text = fontObj.render("RAW", True, (220, 100, 100) if active else (180, 180, 180))
    raw_rect = raw_text.get_rect(center=icon_rect.center)
    windowSurfaceObj.blit(raw_text, raw_rect)

    return icon_rect


def draw_raw_tab_buttons(start_x, start_y, active_tab):
    """
    Dessine les onglets ISP/STRETCH en haut du panneau RAW.

    Args:
        start_x, start_y: Position de départ
        active_tab: 0=ISP, 1=Stretch

    Returns:
        Tuple (isp_rect, stretch_rect) pour détection des clics
    """
    global windowSurfaceObj, _font_cache

    tab_width = 100
    tab_height = 30
    tab_spacing = 5

    # Onglet ISP
    isp_rect = pygame.Rect(start_x, start_y, tab_width, tab_height)
    if active_tab == 0:
        pygame.draw.rect(windowSurfaceObj, (80, 100, 120), isp_rect, border_radius=5)
        text_color = (220, 220, 255)
    else:
        pygame.draw.rect(windowSurfaceObj, (50, 55, 65), isp_rect, border_radius=5)
        text_color = (150, 150, 160)
    pygame.draw.rect(windowSurfaceObj, (100, 110, 130), isp_rect, 1, border_radius=5)

    cache_key = 22
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    isp_text = fontObj.render("ISP", True, text_color)
    isp_text_rect = isp_text.get_rect(center=isp_rect.center)
    windowSurfaceObj.blit(isp_text, isp_text_rect)

    # Onglet STRETCH
    stretch_rect = pygame.Rect(start_x + tab_width + tab_spacing, start_y, tab_width, tab_height)
    if active_tab == 1:
        pygame.draw.rect(windowSurfaceObj, (80, 100, 120), stretch_rect, border_radius=5)
        text_color = (220, 220, 255)
    else:
        pygame.draw.rect(windowSurfaceObj, (50, 55, 65), stretch_rect, border_radius=5)
        text_color = (150, 150, 160)
    pygame.draw.rect(windowSurfaceObj, (100, 110, 130), stretch_rect, 1, border_radius=5)

    stretch_text = fontObj.render("STRETCH", True, text_color)
    stretch_text_rect = stretch_text.get_rect(center=stretch_rect.center)
    windowSurfaceObj.blit(stretch_text, stretch_text_rect)

    return (isp_rect, stretch_rect)


def draw_raw_isp_slider(x, y, width, height, label, value, vmin, vmax, color=(100, 140, 180)):
    """
    Dessine un slider horizontal pour ajuster un paramètre ISP.
    """
    global windowSurfaceObj, _font_cache

    # Fond semi-transparent
    slider_rect = pygame.Rect(x, y, width, height)
    s = pygame.Surface((width, height), pygame.SRCALPHA)
    s.fill((30, 30, 40, 200))
    windowSurfaceObj.blit(s, (x, y))

    # Label
    cache_key = 16
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    # Afficher label et valeur formatée
    if isinstance(value, int):
        label_text = fontObj.render(f"{label}: {value}", True, (220, 220, 220))
    else:
        label_text = fontObj.render(f"{label}: {value:.2f}", True, (220, 220, 220))
    windowSurfaceObj.blit(label_text, (x + 5, y + 2))

    # Barre de progression
    bar_y = y + 18
    bar_height = height - 22
    pygame.draw.rect(windowSurfaceObj, (50, 50, 60), (x + 5, bar_y, width - 10, bar_height))

    # Position du curseur
    if vmax > vmin:
        ratio = (value - vmin) / (vmax - vmin)
    else:
        ratio = 0
    cursor_x = x + 5 + int(ratio * (width - 15))

    # Barre remplie
    pygame.draw.rect(windowSurfaceObj, color, (x + 5, bar_y, cursor_x - x - 5, bar_height))

    # Curseur
    pygame.draw.rect(windowSurfaceObj, (200, 200, 220), (cursor_x, bar_y - 2, 5, bar_height + 4))

    # Bordure
    pygame.draw.rect(windowSurfaceObj, (80, 80, 90), slider_rect, 1)

    return slider_rect


def draw_raw_action_buttons(start_x, start_y):
    """
    Dessine les boutons RESET et SAVE pour le panneau RAW.

    Returns:
        Tuple (reset_rect, save_rect) pour détection des clics
    """
    global windowSurfaceObj, _font_cache

    btn_width = 80
    btn_height = 28
    btn_spacing = 15

    cache_key = 20
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    # Bouton RESET
    reset_rect = pygame.Rect(start_x, start_y, btn_width, btn_height)
    pygame.draw.rect(windowSurfaceObj, (80, 60, 60), reset_rect, border_radius=5)
    pygame.draw.rect(windowSurfaceObj, (120, 90, 90), reset_rect, 2, border_radius=5)
    reset_text = fontObj.render("RESET", True, (220, 180, 180))
    reset_text_rect = reset_text.get_rect(center=reset_rect.center)
    windowSurfaceObj.blit(reset_text, reset_text_rect)

    # Bouton SAVE
    save_rect = pygame.Rect(start_x + btn_width + btn_spacing, start_y, btn_width, btn_height)
    pygame.draw.rect(windowSurfaceObj, (60, 80, 60), save_rect, border_radius=5)
    pygame.draw.rect(windowSurfaceObj, (90, 120, 90), save_rect, 2, border_radius=5)
    save_text = fontObj.render("SAVE", True, (180, 220, 180))
    save_text_rect = save_text.get_rect(center=save_rect.center)
    windowSurfaceObj.blit(save_text, save_text_rect)

    return (reset_rect, save_rect)


def draw_raw_controls(screen_width, screen_height, image_array=None):
    """
    Dessine le panneau complet de réglage RAW avec onglets ISP/Stretch.
    Inclut également les sliders de gain et exposition pour RAW.

    Args:
        screen_width: Largeur de l'écran
        screen_height: Hauteur de l'écran
        image_array: Image numpy pour l'histogramme (optionnel)

    Returns:
        dict avec les rects des contrôles pour détection des clics
    """
    global raw_adjust_tab, windowSurfaceObj, _font_cache
    global isp_wb_red, isp_wb_green, isp_wb_blue, isp_gamma, isp_black_level
    global isp_brightness, isp_contrast, isp_saturation, isp_sharpening
    global ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP
    global stretch_preset, stretch_factor, stretch_p_low, stretch_p_high
    global gain, ev, still_limits, custom_sspeed, sspeed

    control_rects = {}

    # Dimensions du panneau
    panel_width = 280
    panel_margin = 20
    start_x = panel_margin
    start_y = 70

    # Titre
    cache_key = 26
    if cache_key not in _font_cache:
        _font_cache[cache_key] = pygame.font.Font(None, cache_key)
    fontObj = _font_cache[cache_key]

    title_text = fontObj.render("RAW Adjustment Panel", True, (200, 150, 150))
    windowSurfaceObj.blit(title_text, (start_x, 20))

    # =========================================================================
    # SECTION EXPOSURE (Gain + EV) - À droite de l'écran pour RAW
    # =========================================================================
    exp_slider_width = 250
    exp_slider_height = 45
    exp_margin = 10
    exp_start_x = screen_width - exp_slider_width - 40
    exp_start_y = 60

    # Titre Exposure
    cache_key_exp = 20
    if cache_key_exp not in _font_cache:
        _font_cache[cache_key_exp] = pygame.font.Font(None, cache_key_exp)
    expFont = _font_cache[cache_key_exp]

    exp_title = expFont.render("Exposure (RAW)", True, (200, 120, 120))
    windowSurfaceObj.blit(exp_title, (exp_start_x, 20))

    # Slider Gain (0 = AUTO, 1-300 = valeur fixe)
    gain_max = 300  # Plage étendue pour astrophoto
    gain_label = "AUTO" if gain == 0 else f"{gain:.0f}"
    control_rects['raw_gain'] = draw_raw_isp_slider(
        exp_start_x, exp_start_y, exp_slider_width, exp_slider_height,
        f"Gain ({gain_label})", float(gain), 0.0, float(gain_max), (200, 160, 100)
    )

    # Slider Temps d'exposition (0.001s à 20s) - échelle logarithmique
    # custom_sspeed est en microsecondes (1000 = 1ms, 1000000 = 1s)
    # Plage: 1000µs (1ms) à 20000000µs (20s)
    exp_min_us = 1000        # 1ms = 0.001s
    exp_max_us = 20000000    # 20s
    current_exp_us = custom_sspeed if custom_sspeed > 0 else sspeed
    current_exp_us = max(exp_min_us, min(exp_max_us, current_exp_us))

    # Afficher en format lisible
    if current_exp_us >= 1000000:
        exp_label = f"{current_exp_us / 1000000:.1f}s"
    elif current_exp_us >= 1000:
        exp_label = f"{current_exp_us / 1000:.0f}ms"
    else:
        exp_label = f"{current_exp_us}µs"

    # Position logarithmique pour le slider (0 à 1)
    import math
    log_min = math.log10(exp_min_us)
    log_max = math.log10(exp_max_us)
    log_current = math.log10(current_exp_us)
    exp_ratio = (log_current - log_min) / (log_max - log_min)

    control_rects['raw_exposure'] = draw_raw_isp_slider(
        exp_start_x, exp_start_y + exp_slider_height + exp_margin, exp_slider_width, exp_slider_height,
        f"Expo ({exp_label})", exp_ratio, 0.0, 1.0, (100, 180, 180)
    )

    # Onglets
    isp_tab_rect, stretch_tab_rect = draw_raw_tab_buttons(start_x, 50, raw_adjust_tab)
    control_rects['tab_isp'] = isp_tab_rect
    control_rects['tab_stretch'] = stretch_tab_rect

    slider_width = 250
    slider_height = 38
    margin = 6

    if raw_adjust_tab == 0:
        # Onglet ISP - 9 sliders
        cache_key_section = 18
        if cache_key_section not in _font_cache:
            _font_cache[cache_key_section] = pygame.font.Font(None, cache_key_section)
        sectionFont = _font_cache[cache_key_section]

        # Section White Balance
        section_text = sectionFont.render("White Balance:", True, (150, 180, 200))
        windowSurfaceObj.blit(section_text, (start_x, start_y))
        start_y += 18

        control_rects['isp_wb_red'] = draw_raw_isp_slider(
            start_x, start_y, slider_width, slider_height,
            "R Gain", isp_wb_red / 100.0, 0.5, 2.0, (180, 100, 100)
        )
        control_rects['isp_wb_green'] = draw_raw_isp_slider(
            start_x, start_y + slider_height + margin, slider_width, slider_height,
            "G Gain", isp_wb_green / 100.0, 0.5, 2.0, (100, 180, 100)
        )
        control_rects['isp_wb_blue'] = draw_raw_isp_slider(
            start_x, start_y + 2*(slider_height + margin), slider_width, slider_height,
            "B Gain", isp_wb_blue / 100.0, 0.5, 2.0, (100, 100, 180)
        )

        # Section Tone
        start_y += 3*(slider_height + margin) + 10
        section_text = sectionFont.render("Tone:", True, (150, 180, 200))
        windowSurfaceObj.blit(section_text, (start_x, start_y))
        start_y += 18

        control_rects['isp_gamma'] = draw_raw_isp_slider(
            start_x, start_y, slider_width, slider_height,
            "Gamma", isp_gamma / 100.0, 0.5, 3.0, (180, 160, 100)
        )
        control_rects['isp_black_level'] = draw_raw_isp_slider(
            start_x, start_y + slider_height + margin, slider_width, slider_height,
            "Black Level", isp_black_level, 0, 500, (120, 120, 140)
        )
        control_rects['isp_brightness'] = draw_raw_isp_slider(
            start_x, start_y + 2*(slider_height + margin), slider_width, slider_height,
            "Brightness", isp_brightness / 100.0, -0.5, 0.5, (160, 160, 180)
        )
        control_rects['isp_contrast'] = draw_raw_isp_slider(
            start_x, start_y + 3*(slider_height + margin), slider_width, slider_height,
            "Contrast", isp_contrast / 100.0, 0.5, 2.0, (140, 180, 160)
        )

        # Section Color/Detail - À DROITE sous les sliders Gain/Expo
        # Position sous les sliders Gain et Exposure (à droite de l'écran)
        color_start_y = exp_start_y + 2*(exp_slider_height + exp_margin) + 20
        section_text = sectionFont.render("Color/Detail:", True, (150, 180, 200))
        windowSurfaceObj.blit(section_text, (exp_start_x, color_start_y))
        color_start_y += 18

        control_rects['isp_saturation'] = draw_raw_isp_slider(
            exp_start_x, color_start_y, exp_slider_width, slider_height,
            "Saturation", isp_saturation / 100.0, 0.0, 2.0, (180, 140, 180)
        )
        control_rects['isp_sharpening'] = draw_raw_isp_slider(
            exp_start_x, color_start_y + slider_height + margin, exp_slider_width, slider_height,
            "Sharpening", isp_sharpening / 100.0, 0.0, 2.0, (140, 160, 180)
        )

        # Boutons RESET et SAVE - À DROITE sous Color/Detail
        color_start_y += 2*(slider_height + margin) + 15
        reset_rect, save_rect = draw_raw_action_buttons(exp_start_x, color_start_y)
        control_rects['reset'] = reset_rect
        control_rects['save'] = save_rect

    else:
        # Onglet STRETCH - Affiche GHS ou Arcsinh selon stretch_preset
        cache_key_title = 22
        if cache_key_title not in _font_cache:
            _font_cache[cache_key_title] = pygame.font.Font(None, cache_key_title)
        titleFont = _font_cache[cache_key_title]

        if stretch_preset == 2:
            # Mode Arcsinh
            title = titleFont.render("Arcsinh Stretch Parameters", True, (200, 150, 100))
            windowSurfaceObj.blit(title, (start_x, start_y))
            start_y += 25

            control_rects['stretch_factor'] = draw_raw_isp_slider(
                start_x, start_y, slider_width, slider_height,
                "Factor", stretch_factor / 10.0, 0.0, 8.0, (200, 160, 100)
            )
            control_rects['stretch_p_low'] = draw_raw_isp_slider(
                start_x, start_y + slider_height + margin, slider_width, slider_height,
                "Clip Low %", stretch_p_low / 10.0, 0.0, 2.0, (180, 140, 100)
            )
            control_rects['stretch_p_high'] = draw_raw_isp_slider(
                start_x, start_y + 2*(slider_height + margin), slider_width, slider_height,
                "Clip High %", stretch_p_high / 100.0, 99.5, 100.0, (100, 180, 140)
            )

            # Boutons RESET et SAVE pour stretch aussi
            start_y += 3*(slider_height + margin) + 15
            reset_rect, save_rect = draw_raw_action_buttons(start_x, start_y)
            control_rects['reset'] = reset_rect
            control_rects['save'] = save_rect
        else:
            # Mode GHS (par défaut)
            title = titleFont.render("GHS Stretch Parameters", True, (100, 200, 150))
            windowSurfaceObj.blit(title, (start_x, start_y))
            start_y += 25

            control_rects['ghs_D'] = draw_raw_isp_slider(
                start_x, start_y, slider_width, slider_height,
                "D (Stretch)", ghs_D / 10.0, -1.0, 10.0, (100, 180, 140)
            )
            control_rects['ghs_b'] = draw_raw_isp_slider(
                start_x, start_y + slider_height + margin, slider_width, slider_height,
                "b (Local Int)", ghs_b / 10.0, -30.0, 10.0, (100, 160, 180)
            )
            control_rects['ghs_SP'] = draw_raw_isp_slider(
                start_x, start_y + 2*(slider_height + margin), slider_width, slider_height,
                "SP (Symmetry)", ghs_SP / 100.0, 0.0, 1.0, (180, 140, 100)
            )
            control_rects['ghs_LP'] = draw_raw_isp_slider(
                start_x, start_y + 3*(slider_height + margin), slider_width, slider_height,
                "LP (Shadows)", ghs_LP / 100.0, 0.0, 1.0, (140, 100, 180)
            )
            control_rects['ghs_HP'] = draw_raw_isp_slider(
                start_x, start_y + 4*(slider_height + margin), slider_width, slider_height,
                "HP (Highlights)", ghs_HP / 100.0, 0.0, 1.0, (180, 100, 140)
            )

            # Boutons RESET et SAVE pour stretch aussi
            start_y += 5*(slider_height + margin) + 15
            reset_rect, save_rect = draw_raw_action_buttons(start_x, start_y)
            control_rects['reset'] = reset_rect
            control_rects['save'] = save_rect

    # Histogramme en bas
    if image_array is not None:
        draw_stretch_histogram(screen_width, screen_height, image_array)

    return control_rects


def is_click_on_raw_icon(mousex, mousey, screen_width):
    """
    Vérifie si le clic est sur l'icône RAW.

    Args:
        mousex, mousey: Position du clic
        screen_width: Largeur de l'écran

    Returns:
        True si clic sur l'icône, False sinon
    """
    icon_size = 50
    margin = 15
    adj_icon_x = screen_width - icon_size - margin
    icon_x = adj_icon_x - icon_size - 10
    icon_y = margin

    return (icon_x <= mousex <= icon_x + icon_size and
            icon_y <= mousey <= icon_y + icon_size)


def handle_raw_slider_click(mousex, mousey, control_rects):
    """
    Gère le clic sur un slider RAW et met à jour la valeur correspondante.
    Applique immédiatement les changements à la session active.
    Inclut aussi les sliders de gain et ev pour RAW.

    Args:
        mousex, mousey: Position du clic
        control_rects: Dict des rectangles des contrôles

    Returns:
        True si un contrôle a été modifié, False sinon
    """
    global raw_adjust_tab, stretch_preset
    global isp_wb_red, isp_wb_green, isp_wb_blue, isp_gamma, isp_black_level
    global isp_brightness, isp_contrast, isp_saturation, isp_sharpening
    global ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP
    global stretch_factor, stretch_p_low, stretch_p_high
    global gain, ev, still_limits, custom_sspeed

    # Vérifier clic sur onglets
    if 'tab_isp' in control_rects and control_rects['tab_isp'].collidepoint(mousex, mousey):
        raw_adjust_tab = 0
        return True
    if 'tab_stretch' in control_rects and control_rects['tab_stretch'].collidepoint(mousex, mousey):
        raw_adjust_tab = 1
        # Activer automatiquement le preset GHS quand on ouvre l'onglet STRETCH
        if stretch_preset == 0:
            stretch_preset = 1
            print("[RAW PANEL] GHS stretch activé automatiquement")
        return True

    # Vérifier clic sur boutons RESET et SAVE
    if 'reset' in control_rects and control_rects['reset'].collidepoint(mousex, mousey):
        reset_isp_to_defaults()
        return True
    if 'save' in control_rects and control_rects['save'].collidepoint(mousex, mousey):
        save_isp_config_to_file()
        return True

    # Vérifier clic sur sliders
    for name, rect in control_rects.items():
        if name.startswith('tab_') or name in ('reset', 'save'):
            continue

        if rect.collidepoint(mousex, mousey):
            # Calculer la position relative dans le slider
            rel_x = mousex - rect.x - 5
            slider_width = rect.width - 10
            ratio = max(0, min(1, rel_x / slider_width))

            # Mettre à jour la valeur selon le slider
            if name == 'isp_wb_red':
                isp_wb_red = int(50 + ratio * 150)  # 0.5-2.0 → 50-200
            elif name == 'isp_wb_green':
                isp_wb_green = int(50 + ratio * 150)
            elif name == 'isp_wb_blue':
                isp_wb_blue = int(50 + ratio * 150)
            elif name == 'isp_gamma':
                isp_gamma = int(50 + ratio * 250)  # 0.5-3.0 → 50-300
            elif name == 'isp_black_level':
                isp_black_level = int(ratio * 500)  # 0-500 direct
            elif name == 'isp_brightness':
                isp_brightness = int(-50 + ratio * 100)  # -0.5-0.5 → -50-50
            elif name == 'isp_contrast':
                isp_contrast = int(50 + ratio * 150)  # 0.5-2.0 → 50-200
            elif name == 'isp_saturation':
                isp_saturation = int(ratio * 200)  # 0.0-2.0 → 0-200
            elif name == 'isp_sharpening':
                isp_sharpening = int(ratio * 200)  # 0.0-2.0 → 0-200
            # Sliders GHS (onglet STRETCH)
            elif name == 'ghs_D':
                ghs_D = int(-10 + ratio * 110)  # -1.0-10.0 → -10-100
            elif name == 'ghs_b':
                ghs_b = int(-300 + ratio * 400)  # -30.0-10.0 → -300-100
            elif name == 'ghs_SP':
                ghs_SP = int(ratio * 100)  # 0.0-1.0 → 0-100
            elif name == 'ghs_LP':
                ghs_LP = int(ratio * 100)
            elif name == 'ghs_HP':
                ghs_HP = int(ratio * 100)
            # Sliders Arcsinh (onglet STRETCH mode Arcsinh)
            elif name == 'stretch_factor':
                stretch_factor = int(ratio * 80)  # 0.0-8.0 → 0-80
                print(f"[RAW ARCSINH] stretch_factor = {stretch_factor} (factor={stretch_factor/10.0})")
            elif name == 'stretch_p_low':
                stretch_p_low = int(ratio * 20)  # 0.0-2.0 → 0-20
                print(f"[RAW ARCSINH] stretch_p_low = {stretch_p_low}")
            elif name == 'stretch_p_high':
                stretch_p_high = int(9950 + ratio * 50)  # 99.5-100.0 → 9950-10000
                print(f"[RAW ARCSINH] stretch_p_high = {stretch_p_high}")
            # Sliders Gain et Exposition pour RAW
            elif name == 'raw_gain':
                # Gain: 0 à 300 (0 = AUTO)
                gain_max = 300  # Plage étendue pour astrophoto
                gain = int(ratio * gain_max)
                # Appliquer immédiatement les changements à la caméra
                apply_gain_exposure_to_camera()
            elif name == 'raw_exposure':
                # Exposition: échelle logarithmique de 1ms à 20s
                import math
                exp_min_us = 1000        # 1ms
                exp_max_us = 20000000    # 20s
                log_min = math.log10(exp_min_us)
                log_max = math.log10(exp_max_us)
                # Convertir le ratio (0-1) en valeur logarithmique
                log_value = log_min + ratio * (log_max - log_min)
                custom_sspeed = int(10 ** log_value)
                # Appliquer immédiatement les changements à la caméra
                apply_gain_exposure_to_camera()

            # Appliquer les changements ISP à la session active
            if name.startswith('isp_'):
                apply_isp_to_session()

            # Activer automatiquement GHS quand on modifie un slider GHS
            if name.startswith('ghs_') and stretch_preset == 0:
                stretch_preset = 1
                print("[RAW PANEL] GHS stretch activé automatiquement (slider modifié)")

            return True

    return False


def apply_isp_to_session():
    """
    Applique les paramètres ISP GUI à la session de stacking active.
    Modifie directement session.isp.config pour mise à jour temps réel.
    """
    global livestack, luckystack
    global isp_wb_red, isp_wb_green, isp_wb_blue, isp_gamma, isp_black_level
    global isp_brightness, isp_contrast, isp_saturation, isp_sharpening

    session = None

    # Trouver la session active
    if livestack is not None and livestack.session is not None:
        session = livestack.session
    elif luckystack is not None and luckystack.session is not None:
        session = luckystack.session

    if session is None or session.isp is None:
        return

    # Appliquer les valeurs à la config ISP
    # IMPORTANT: Utiliser les MÊMES formules que apply_isp_to_preview() pour cohérence
    config = session.isp.config
    config.wb_red_gain = isp_wb_red / 100.0       # 50-200 → 0.5-2.0
    config.wb_green_gain = isp_wb_green / 100.0   # 50-200 → 0.5-2.0
    config.wb_blue_gain = isp_wb_blue / 100.0     # 50-200 → 0.5-2.0
    config.gamma = isp_gamma / 100.0              # 50-300 → 0.5-3.0 (100=1.0 linéaire)
    config.black_level = isp_black_level          # 0-500 direct
    config.brightness_offset = isp_brightness / 100.0  # -50-50 → -0.5-0.5
    config.contrast = isp_contrast / 100.0        # 50-200 → 0.5-2.0
    config.saturation = isp_saturation / 100.0    # 0-200 → 0.0-2.0
    config.sharpening = isp_sharpening / 100.0    # 0-200 → 0.0-2.0


def apply_gain_exposure_to_camera():
    """
    Applique les changements de gain et temps d'exposition à la caméra Picamera2.
    Cette fonction est appelée depuis les sliders de réglage en mode stretch preview.

    Utilise:
        - gain: valeur de gain analogique (0 = AUTO)
        - custom_sspeed: temps d'exposition en microsecondes
    """
    global picam2, gain, custom_sspeed, sspeed, mode

    if picam2 is None:
        return

    try:
        controls = {}

        # Appliquer le gain si différent de 0 (0 = AUTO)
        if gain > 0:
            controls["AnalogueGain"] = float(gain)

        # Appliquer le temps d'exposition en microsecondes
        # custom_sspeed contient la valeur définie par le slider
        exp_time = custom_sspeed if custom_sspeed > 0 else sspeed
        if exp_time > 0:
            controls["ExposureTime"] = int(exp_time)

        # Appliquer les contrôles à la caméra
        if controls:
            picam2.set_controls(controls)
            # Afficher en format lisible
            if exp_time >= 1000000:
                exp_label = f"{exp_time / 1000000:.1f}s"
            elif exp_time >= 1000:
                exp_label = f"{exp_time / 1000:.0f}ms"
            else:
                exp_label = f"{exp_time}µs"
            print(f"[STRETCH] Appliqué: Gain={gain if gain > 0 else 'AUTO'}, Expo={exp_label}")

    except Exception as e:
        print(f"[STRETCH] Erreur application gain/exposition: {e}")


def reset_isp_to_defaults():
    """
    Réinitialise tous les paramètres ISP à leurs valeurs par défaut.
    """
    global isp_wb_red, isp_wb_green, isp_wb_blue, isp_gamma, isp_black_level
    global isp_brightness, isp_contrast, isp_saturation, isp_sharpening
    global ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP, raw_adjust_tab

    if raw_adjust_tab == 0:
        # Reset ISP
        isp_wb_red = 100
        isp_wb_green = 100
        isp_wb_blue = 100
        isp_gamma = 100
        isp_black_level = 64
        isp_brightness = 0
        isp_contrast = 100
        isp_saturation = 100
        isp_sharpening = 0
        apply_isp_to_session()
    else:
        # Reset GHS (valeurs optimisées par défaut)
        ghs_D = 31
        ghs_b = 1
        ghs_SP = 19
        ghs_LP = 0
        ghs_HP = 0


def save_isp_config_to_file():
    """
    Sauvegarde la configuration ISP actuelle dans session_isp_config.json.
    """
    global isp_wb_red, isp_wb_green, isp_wb_blue, isp_gamma, isp_black_level
    global isp_brightness, isp_contrast, isp_saturation, isp_sharpening
    global ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP

    import json

    config = {
        "isp": {
            "wb_red_gain": isp_wb_red / 100.0,
            "wb_green_gain": isp_wb_green / 100.0,
            "wb_blue_gain": isp_wb_blue / 100.0,
            "gamma": isp_gamma / 100.0 * 2.2,
            "black_level": isp_black_level,
            "brightness_offset": isp_brightness / 100.0,
            "contrast": isp_contrast / 100.0,
            "saturation": isp_saturation / 100.0,
            "sharpening": isp_sharpening / 100.0
        },
        "ghs": {
            "D": ghs_D / 10.0,
            "b": ghs_b / 10.0,
            "SP": ghs_SP / 100.0,
            "LP": ghs_LP / 100.0,
            "HP": ghs_HP / 100.0
        }
    }

    try:
        config_path = os.path.join(os.path.dirname(__file__), "session_isp_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[RAW] Configuration ISP sauvegardée: {config_path}")
    except Exception as e:
        print(f"[RAW] Erreur sauvegarde config: {e}")


def load_isp_from_session():
    """
    Charge les valeurs ISP depuis la session active vers les variables GUI.
    Appelé quand on active le panneau RAW pour synchroniser l'affichage.
    """
    global livestack, luckystack
    global isp_wb_red, isp_wb_green, isp_wb_blue, isp_gamma, isp_black_level
    global isp_brightness, isp_contrast, isp_saturation, isp_sharpening

    session = None

    # Trouver la session active
    if livestack is not None and livestack.session is not None:
        session = livestack.session
    elif luckystack is not None and luckystack.session is not None:
        session = luckystack.session

    if session is None or session.isp is None:
        return

    # Charger les valeurs depuis la config ISP
    config = session.isp.config
    isp_wb_red = int(config.wb_red_gain * 100)
    isp_wb_green = int(config.wb_green_gain * 100)
    isp_wb_blue = int(config.wb_blue_gain * 100)
    isp_gamma = int((config.gamma / 2.2) * 100)  # Convertir gamma en ratio
    isp_black_level = config.black_level
    isp_brightness = int(config.brightness_offset * 100)
    isp_contrast = int(config.contrast * 100)
    isp_saturation = int(config.saturation * 100)
    isp_sharpening = int(config.sharpening * 100)


def apply_isp_to_preview(array):
    """
    Applique les paramètres ISP GUI à une image de preview en temps réel.
    Utilise les MÊMES formules que libastrostack/isp.py pour que le preview
    corresponde exactement aux images stackées.

    Args:
        array: numpy array de l'image (H, W, 3) en RGB, uint8 ou float32

    Returns:
        numpy array traité de même dimension
    """
    global isp_wb_red, isp_wb_green, isp_wb_blue, isp_gamma, isp_black_level
    global isp_brightness, isp_contrast, isp_saturation, isp_sharpening

    # Convertir les paramètres GUI en valeurs réelles (IDENTIQUE au mapping ISP)
    # IMPORTANT: Ces formules doivent être identiques à celles du mapping ISP config
    wb_r = isp_wb_red / 100.0         # 50-200 → 0.5-2.0
    wb_g = isp_wb_green / 100.0       # 50-200 → 0.5-2.0
    wb_b = isp_wb_blue / 100.0        # 50-200 → 0.5-2.0
    gamma = isp_gamma / 100.0         # 50-300 → 0.5-3.0 (100=1.0 linéaire)
    black_level = isp_black_level     # 0-500 direct
    brightness = isp_brightness / 100.0  # -50-50 → -0.5-0.5
    contrast = isp_contrast / 100.0   # 50-200 → 0.5-2.0
    saturation = isp_saturation / 100.0  # 0-200 → 0.0-2.0
    sharpening = isp_sharpening / 100.0  # 0-200 → 0.0-2.0

    # Vérifier si tous les paramètres sont à leur valeur par défaut
    # Dans ce cas, ne pas traiter pour optimiser les performances
    if (wb_r == 1.0 and wb_g == 1.0 and wb_b == 1.0 and
        abs(gamma - 1.0) < 0.01 and black_level == 64 and
        brightness == 0.0 and contrast == 1.0 and
        saturation == 1.0 and sharpening == 0.0):
        return array

    # Convertir en float32 [0-1] (IDENTIQUE à ISP._to_float)
    input_dtype = array.dtype
    if input_dtype == np.uint8:
        img = array.astype(np.float32) / 255.0
    elif input_dtype == np.uint16:
        img = array.astype(np.float32) / 65535.0
    else:
        img = array.astype(np.float32)
        if img.max() > 1.0:
            img = img / img.max()

    # 1. Black level subtraction (IDENTIQUE à ISP._apply_black_level)
    # Normalisation 12-bit comme dans libastrostack
    if black_level > 0:
        black_norm = black_level / 4095.0
        img = np.clip(img - black_norm, 0, 1)
        # Renormaliser
        if img.max() > 0:
            img = img / img.max()

    # 2. White balance - ATTENTION: pygame fait un swap R↔B pour l'affichage ([:,:,[2,1,0]])
    # Donc on doit appliquer wb_r sur canal 2 et wb_b sur canal 0 pour que l'effet
    # corresponde visuellement au label du slider
    if wb_r != 1.0 or wb_g != 1.0 or wb_b != 1.0:
        # INVERSÉ pour compenser le swap pygame [:,:,[2,1,0]]
        img[:, :, 2] = img[:, :, 2] * wb_r  # Canal 2 natif → Rouge à l'écran
        img[:, :, 1] = img[:, :, 1] * wb_g  # Canal 1 = Vert (inchangé)
        img[:, :, 0] = img[:, :, 0] * wb_b  # Canal 0 natif → Bleu à l'écran
        img = np.clip(img, 0, 1)

    # 3. Gamma correction (IDENTIQUE à ISP._apply_gamma)
    if gamma != 1.0:
        img = np.power(np.clip(img, 1e-10, 1.0), 1.0 / gamma)

    # 4. Brightness offset (IDENTIQUE à ISP._apply_brightness_offset)
    if brightness != 0.0:
        img = np.clip(img + brightness, 0.0, 1.0)

    # 5. Contrast (IDENTIQUE à ISP._apply_contrast)
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5
        img = np.clip(img, 0, 1)

    # 6. Saturation (IDENTIQUE à ISP._apply_saturation - utilise HSV)
    if saturation != 1.0 and len(img.shape) == 3 and img.shape[2] == 3:
        # Convertir en uint8 pour cv2
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[:, :, 1] *= saturation
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        img = img_rgb.astype(np.float32) / 255.0

    # 7. Sharpening (IDENTIQUE à ISP._apply_sharpening - unsharp mask)
    if sharpening > 0.01 and len(img.shape) == 3:
        # Unsharp mask via cv2 (plus rapide que scipy)
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (0, 0), 1.0)
        img_sharpened = cv2.addWeighted(img_uint8, 1.0 + sharpening, blurred, -sharpening, 0)
        img = np.clip(img_sharpened.astype(np.float32) / 255.0, 0, 1)

    # Reconvertir dans le type d'origine
    if input_dtype == np.uint8:
        return (img * 255).astype(np.uint8)
    elif input_dtype == np.uint16:
        return (img * 65535).astype(np.uint16)
    else:
        return (img * 255).astype(np.float32)


def save_with_external_processing(stacker_obj, filename=None, raw_format_name=None):
    """
    Sauvegarde le résultat stacké avec ISP+stretch externe (pour mode RAW).

    Cette fonction récupère le résultat brut du stacker, applique apply_isp_to_preview()
    et astro_stretch(), puis sauvegarde en FITS et PNG.

    Args:
        stacker_obj: livestack ou luckystack object
        filename: Nom fichier (optionnel)
        raw_format_name: Format RAW pour le nom de fichier

    Returns:
        Path du fichier sauvegardé ou None
    """
    global stretch_preset

    if stacker_obj is None or stacker_obj.session is None:
        print("[SAVE] Aucune session active")
        return None

    # Récupérer le résultat brut (sans ISP/stretch de libastrostack)
    try:
        if hasattr(stacker_obj, 'get_final_result'):
            raw_result = stacker_obj.get_final_result()
        elif hasattr(stacker_obj.session, 'stacker') and stacker_obj.session.stacker:
            raw_result = stacker_obj.session.stacker.get_stacked_image()
        else:
            raw_result = None

        if raw_result is None:
            print("[SAVE] Aucun résultat à sauvegarder")
            return None

    except Exception as e:
        print(f"[SAVE] Erreur récupération résultat: {e}")
        return None

    # Appliquer ISP externe
    processed = apply_isp_to_preview(raw_result.copy())

    # Appliquer stretch externe (si activé)
    if stretch_preset != 0:
        processed = astro_stretch(processed)

    # Générer le nom de fichier
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_format_str = raw_format_name or stacker_obj.camera_params.get('raw_format', 'RAW')
        # Récupérer la méthode de stacking (compatible avec LegacyStackingConfig et AdvancedStackingConfig)
        method = 'mean'
        if stacker_obj.session:
            if hasattr(stacker_obj.session.config, 'stacking') and hasattr(stacker_obj.session.config.stacking, 'method'):
                method = stacker_obj.session.config.stacking.method.value
        filename = f"stack_{raw_format_str}_ext_{method}_{timestamp}"

    output_dir = stacker_obj.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder FITS (linéaire, 32-bit)
    fits_path = output_dir / f"{filename}.fit"
    try:
        from astropy.io import fits

        # Pour FITS: utiliser le résultat brut linéaire (pas de stretch)
        fits_data = apply_isp_to_preview(raw_result.copy())

        # Convertir en format FITS (channels, height, width)
        if len(fits_data.shape) == 3:
            fits_data = np.transpose(fits_data, (2, 0, 1))

        hdu = fits.PrimaryHDU(fits_data.astype(np.float32))
        hdu.header['NAXIS'] = 3
        hdu.header['COMMENT'] = 'Created by RPiCamera2 with external ISP processing'
        hdu.header['STACKCNT'] = stacker_obj.session.config.num_stacked if stacker_obj.session else 0
        hdul = fits.HDUList([hdu])
        hdul.writeto(str(fits_path), overwrite=True)
        print(f"[SAVE] FITS: {fits_path}")
    except Exception as e:
        print(f"[SAVE] Erreur FITS: {e}")

    # Sauvegarder PNG (avec stretch, 16-bit)
    png_path = output_dir / f"{filename}.png"
    try:
        import cv2

        # Convertir en uint16 pour PNG 16-bit
        if processed.dtype == np.float32:
            png_data = np.clip(processed / 255.0 * 65535, 0, 65535).astype(np.uint16)
        elif processed.dtype == np.uint8:
            png_data = (processed.astype(np.uint16) * 257)  # 0-255 -> 0-65535
        else:
            png_data = np.clip(processed, 0, 65535).astype(np.uint16)

        # Convertir RGB -> BGR pour OpenCV
        if len(png_data.shape) == 3 and png_data.shape[2] == 3:
            png_data = cv2.cvtColor(png_data, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(png_path), png_data)
        print(f"[SAVE] PNG: {png_path} (16-bit, traitement externe)")
    except Exception as e:
        print(f"[SAVE] Erreur PNG: {e}")

    return fits_path


# ============================================================================
# FIN RAW FULLSCREEN CONTROLS
# ============================================================================

def calculate_fwhm(image_surface, center_x, center_y, area_size):
    """
    Calcule le FWHM pour mesurer la netteté
    Version corrigée qui ne verrouille pas la surface
    """
    try:
        # IMPORTANT : Utiliser array3d au lieu de pixels3d (ne verrouille pas)
        image_array = pygame.surfarray.array3d(image_surface)
        
        # Extraire la région d'intérêt
        y1 = max(0, center_y - area_size)
        y2 = min(image_array.shape[1], center_y + area_size)  # Note: shape[1] pour y
        x1 = max(0, center_x - area_size)
        x2 = min(image_array.shape[0], center_x + area_size)  # Note: shape[0] pour x
        
        # pygame.surfarray retourne (width, height, channels), donc on transpose
        roi = image_array[x1:x2, y1:y2, :]
        roi = np.transpose(roi, (1, 0, 2))  # Convertir en (height, width, channels)
        
        # Convertir en niveaux de gris
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Trouver le point le plus brillant dans la ROI
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        
        # Profils horizontal et vertical passant par le point le plus brillant
        profile_h = gray[max_loc[1], :]
        profile_v = gray[:, max_loc[0]]
        
        # Calculer FWHM pour chaque profil
        fwhm_h = calculate_fwhm_1d(profile_h)
        fwhm_v = calculate_fwhm_1d(profile_v)
        
        # Retourner la moyenne
        if fwhm_h is not None and fwhm_v is not None:
            return (fwhm_h + fwhm_v) / 2.0
        elif fwhm_h is not None:
            return fwhm_h
        elif fwhm_v is not None:
            return fwhm_v
        else:
            return None
            
    except Exception as e:
        print(f"Erreur FWHM: {e}")
        return None

def calculate_fwhm_1d(profile):
    """Calcule le FWHM d'un profil 1D avec détection de pic améliorée"""
    try:
        max_val = np.max(profile)
        min_val = np.min(profile)
        
        # Vérifier qu'il y a un contraste suffisant (au moins 10% de différence)
        if max_val <= min_val or (max_val - min_val) / max_val < 0.1:
            return None
            
        half_max = min_val + (max_val - min_val) / 2.0
        above_half = profile >= half_max
        
        if not np.any(above_half):
            return None
        
        indices = np.where(above_half)[0]
        
        if len(indices) < 2:
            return None
        
        # Calculer FWHM (largeur à mi-hauteur)
        fwhm = indices[-1] - indices[0] + 1  # +1 pour inclure les deux bords
            
        return float(fwhm)
    except:
        return None

def get_fwhm_color(fwhm):
    """Retourne une couleur RGB selon la qualité du FWHM"""
    if fwhm is None:
        return (128, 128, 128)
    elif fwhm < 3:
        return (0, 255, 0)
    elif fwhm < 6:
        return (255, 255, 0)
    elif fwhm < 10:
        return (255, 165, 0)
    else:
        return (255, 0, 0)

def get_fwhm_quality_text(fwhm):
    """Retourne un texte de qualité selon le FWHM"""
    if fwhm is None:
        return "N/A"
    elif fwhm < 3:
        return "Excellente"
    elif fwhm < 6:
        return "Bonne"
    elif fwhm < 10:
        return "Moyenne"
    else:
        return "Mauvaise"

def init_fwhm_graph():
    """Initialise le graphique matplotlib pour le FWHM"""
    global fwhm_fig, fwhm_ax

    if fwhm_fig is None:
        fwhm_fig, fwhm_ax = plt.subplots(figsize=(6, 3), dpi=100)
        fwhm_fig.patch.set_facecolor('#1a1a1a')
        fwhm_ax.set_facecolor('#0a0a0a')

    return fwhm_fig, fwhm_ax

def init_hfr_graph():
    """Initialise le graphique matplotlib pour le HFR"""
    global hfr_fig, hfr_ax

    if hfr_fig is None:
        # Taille et style élégants pour meilleure lisibilité
        hfr_fig, hfr_ax = plt.subplots(figsize=(6, 3), dpi=100)
        hfr_fig.patch.set_facecolor('#1a1a1a')
        hfr_ax.set_facecolor('#0a0a0a')

    return hfr_fig, hfr_ax

def update_fwhm_graph(fwhm_val):
    """Met à jour le graphique FWHM et retourne une surface pygame"""
    global fwhm_history, fwhm_times, fwhm_start_time, fwhm_fig, fwhm_ax
    
    if fwhm_val is None:
        return None
    
    if fwhm_start_time == 0:
        fwhm_start_time = time.time()
    
    current_time = time.time() - fwhm_start_time
    fwhm_history.append(fwhm_val)
    fwhm_times.append(current_time)
    
    if len(fwhm_history) < 2:
        return None
    
    fig, ax = init_fwhm_graph()
    ax.clear()
    
    # Zones de qualité
    ax.axhspan(0, 3, alpha=0.2, color='green', linewidth=0)
    ax.axhspan(3, 6, alpha=0.2, color='yellow', linewidth=0)
    ax.axhspan(6, 10, alpha=0.2, color='orange', linewidth=0)
    ax.axhspan(10, max(list(fwhm_history) + [15]), alpha=0.2, color='red', linewidth=0)
    
    # Courbe FWHM avec couleurs
    times_list = list(fwhm_times)
    fwhm_list = list(fwhm_history)
    
    for i in range(len(fwhm_list) - 1):
        color = get_fwhm_color(fwhm_list[i])
        color_norm = tuple(c/255.0 for c in color)
        ax.plot(times_list[i:i+2], fwhm_list[i:i+2], 
               color=color_norm, linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel('FWHM (px)', color='white', fontsize=11, fontweight='bold')
    ax.set_title('Évolution FWHM', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)
    
    max_fwhm = max(fwhm_list)
    ax.set_ylim(0, max(max_fwhm * 1.2, 15))

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def update_hfr_graph(hfr_val):
    """Met à jour le graphique HFR et retourne une surface pygame"""
    global hfr_history, hfr_times, hfr_start_time, hfr_fig, hfr_ax
    
    if hfr_val is None:
        return None
    
    if hfr_start_time == 0:
        hfr_start_time = time.time()
    
    current_time = time.time() - hfr_start_time
    hfr_history.append(hfr_val)
    hfr_times.append(current_time)
    
    if len(hfr_history) < 2:
        return None
    
    fig, ax = init_hfr_graph()
    ax.clear()
    
    # Zones de qualité HFR (valeurs différentes de FWHM)
    ax.axhspan(0, 2, alpha=0.2, color='green', linewidth=0)
    ax.axhspan(2, 3.5, alpha=0.2, color='yellow', linewidth=0)
    ax.axhspan(3.5, 5, alpha=0.2, color='orange', linewidth=0)
    ax.axhspan(5, max(list(hfr_history) + [8]), alpha=0.2, color='red', linewidth=0)
    
    # Courbe HFR avec couleurs
    times_list = list(hfr_times)
    hfr_list = list(hfr_history)
    
    for i in range(len(hfr_list) - 1):
        # Déterminer la couleur selon HFR
        if hfr_list[i] < 2:
            color = (0, 255, 0)  # vert
        elif hfr_list[i] < 3.5:
            color = (255, 255, 0)  # jaune
        elif hfr_list[i] < 5:
            color = (255, 165, 0)  # orange
        else:
            color = (255, 0, 0)  # rouge
        
        color_norm = tuple(c/255.0 for c in color)
        ax.plot(times_list[i:i+2], hfr_list[i:i+2],
               color=color_norm, linewidth=2.5, marker='o', markersize=4)
    
    ax.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel('HFR (px)', color='white', fontsize=11, fontweight='bold')
    ax.set_title('Évolution HFR', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)

    max_hfr = max(hfr_list)
    ax.set_ylim(0, max(max_hfr * 1.2, 8))

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def reset_fwhm_history():
    """Réinitialise l'historique FWHM"""
    global fwhm_history, fwhm_times, fwhm_start_time
    fwhm_history.clear()
    fwhm_times.clear()
    fwhm_start_time = 0

def reset_hfr_history():
    """Réinitialise l'historique HFR"""
    global hfr_history, hfr_times, hfr_start_time
    hfr_history.clear()
    hfr_times.clear()
    hfr_start_time = 0

def init_focus_graph():
    """Initialise le graphique Focus (Laplacian variance)"""
    global focus_fig, focus_ax
    if focus_fig is None:
        focus_fig, focus_ax = plt.subplots(figsize=(6, 3), dpi=100)
        focus_fig.patch.set_facecolor('#1a1a1a')
        focus_ax.set_facecolor('#0a0a0a')
    return focus_fig, focus_ax

def update_focus_graph(focus_val, method_name='Laplacian'):
    """
    Met à jour le graphique Focus et retourne une surface pygame

    Args:
        focus_val: Valeur du focus
        method_name: Nom de la méthode de focus (Laplacian, Gradient, Sobel, Tenengrad)

    Returns:
        Surface pygame du graphique ou None
    """
    global focus_history, focus_times, focus_start_time, focus_fig, focus_ax
    global _focus_frame_counter, _graphs_update_interval

    if focus_val is None or focus_val == 0:
        return None

    if focus_start_time == 0:
        focus_start_time = time.time()

    current_time = time.time() - focus_start_time
    focus_history.append(focus_val)
    focus_times.append(current_time)

    if len(focus_history) < 2:
        return None

    # Optimisation Phase 2 Demande 7 : Limiter la fréquence de mise à jour
    _focus_frame_counter += 1
    if _focus_frame_counter < _graphs_update_interval:
        return None
    _focus_frame_counter = 0

    # Optimisation Phase 2 Demande 7 : Réutiliser fig/ax au lieu de recréer
    if focus_fig is None or focus_ax is None:
        fig, ax = init_focus_graph()
    else:
        fig, ax = focus_fig, focus_ax
    ax.clear()

    # Zones de qualité Focus avec gradients subtils
    ax.axhspan(0, 50, alpha=0.15, color='red', linewidth=0)
    ax.axhspan(50, 200, alpha=0.15, color='orange', linewidth=0)
    ax.axhspan(200, 500, alpha=0.15, color='yellow', linewidth=0)
    max_focus = max(list(focus_history) + [800])
    ax.axhspan(500, max_focus * 1.2, alpha=0.15, color='green', linewidth=0)

    # Courbe Focus avec dégradé de couleurs
    times_list = list(focus_times)
    focus_list = list(focus_history)

    for i in range(len(focus_list) - 1):
        # Déterminer la couleur selon la qualité du focus
        if focus_list[i] > 500:
            color = (0, 255, 0)  # vert
        elif focus_list[i] > 200:
            color = (255, 255, 0)  # jaune
        elif focus_list[i] > 50:
            color = (255, 165, 0)  # orange
        else:
            color = (255, 0, 0)  # rouge

        color_norm = tuple(c/255.0 for c in color)
        ax.plot(times_list[i:i+2], focus_list[i:i+2],
               color=color_norm, linewidth=3, marker='o', markersize=5,
               markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel('Focus', color='white', fontsize=11, fontweight='bold')
    # CORRECTION : Utiliser le nom de la méthode passé en paramètre
    ax.set_title(f'Évolution Focus ({method_name})', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)

    ax.set_ylim(0, max(max_focus * 1.2, 800))

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def update_combined_hfr_fwhm_graph(hfr_val, fwhm_val):
    """Met à jour un graphique combiné HFR+FWHM et retourne une surface pygame"""
    global hfr_history, hfr_times, hfr_start_time
    global fwhm_history, fwhm_times, fwhm_start_time
    global hfr_fig, hfr_ax, fwhm_fig, fwhm_ax
    global _hfr_fwhm_frame_counter, _graphs_update_interval

    # Synchroniser les temps de départ
    if hfr_start_time == 0 and fwhm_start_time == 0:
        hfr_start_time = fwhm_start_time = time.time()
    elif hfr_start_time == 0:
        hfr_start_time = fwhm_start_time
    elif fwhm_start_time == 0:
        fwhm_start_time = hfr_start_time

    current_time = time.time() - hfr_start_time

    # Ajouter les valeurs aux historiques
    if hfr_val is not None:
        hfr_history.append(hfr_val)
        hfr_times.append(current_time)

    if fwhm_val is not None:
        fwhm_history.append(fwhm_val)
        fwhm_times.append(current_time)

    if len(hfr_history) < 2 and len(fwhm_history) < 2:
        return None

    # Optimisation Phase 2 Demande 7 : Limiter la fréquence de mise à jour
    _hfr_fwhm_frame_counter += 1
    if _hfr_fwhm_frame_counter < _graphs_update_interval:
        return None
    _hfr_fwhm_frame_counter = 0

    # Optimisation Phase 2 Demande 7 : Réutiliser fig/ax au lieu de recréer
    if hfr_fig is None or hfr_ax is None:
        fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
        hfr_fig, hfr_ax = fig, ax1
        fig.patch.set_facecolor('#1a1a1a')
        ax1.set_facecolor('#0a0a0a')
    else:
        fig, ax1 = hfr_fig, hfr_ax
        ax1.clear()
        ax1.set_facecolor('#0a0a0a')

    # Axe pour HFR (gauche)
    if len(hfr_history) >= 2:
        hfr_times_list = list(hfr_times)
        hfr_list = list(hfr_history)

        # Tracer HFR avec couleur dynamique
        for i in range(len(hfr_list) - 1):
            if hfr_list[i] < 2:
                color = '#00ff00'  # vert
            elif hfr_list[i] < 3.5:
                color = '#ffff00'  # jaune
            else:
                color = '#ff6600'  # orange

            ax1.plot(hfr_times_list[i:i+2], hfr_list[i:i+2],
                    color=color, linewidth=2.5, marker='o', markersize=4)

        ax1.set_ylabel('HFR (px)', color='#00ff88', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00ff88', colors='#00ff88', labelsize=10)
        max_hfr = max(hfr_list)
        ax1.set_ylim(0, max(max_hfr * 1.2, 8))

    # Axe pour FWHM (droite)
    ax2 = ax1.twinx()
    if len(fwhm_history) >= 2:
        fwhm_times_list = list(fwhm_times)
        fwhm_list = list(fwhm_history)

        # Tracer FWHM avec couleur dynamique
        for i in range(len(fwhm_list) - 1):
            if fwhm_list[i] < 5:
                color = '#ff00ff'  # magenta
            elif fwhm_list[i] < 10:
                color = '#ff88ff'  # rose
            else:
                color = '#ff0088'  # rouge-rose

            ax2.plot(fwhm_times_list[i:i+2], fwhm_list[i:i+2],
                    color=color, linewidth=2.5, marker='s', markersize=4, linestyle='--')

        ax2.set_ylabel('FWHM (px)', color='#ff00ff', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#ff00ff', colors='#ff00ff', labelsize=10)
        max_fwhm = max(fwhm_list)
        ax2.set_ylim(0, max(max_fwhm * 1.2, 15))

    ax1.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax1.set_title('HFR + FWHM', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', colors='white', labelsize=10)
    ax1.grid(True, alpha=0.3, color='gray', linestyle=':', linewidth=0.8)

    for spine in ax1.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)
    for spine in ax2.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def update_star_metric_graph(metric_val, metric_name):
    """
    Met à jour un graphique pour une métrique stellaire (HFR ou FWHM)

    Args:
        metric_val: Valeur de la métrique (HFR ou FWHM)
        metric_name: 'HFR' ou 'FWHM'

    Returns:
        Surface pygame du graphique ou None
    """
    global hfr_history, hfr_times, hfr_start_time
    global fwhm_history, fwhm_times, fwhm_start_time
    global _hfr_fwhm_frame_counter, _graphs_update_interval

    # Sélectionner l'historique approprié
    if metric_name == 'HFR':
        history = hfr_history
        times = hfr_times
        if hfr_start_time == 0:
            hfr_start_time = time.time()
        start_time = hfr_start_time
        color_primary = '#00ff88'  # Vert
        threshold_excellent = 2
        threshold_good = 3.5
        ylabel = 'HFR (px)'
        y_max_default = 8
    else:  # FWHM
        history = fwhm_history
        times = fwhm_times
        if fwhm_start_time == 0:
            fwhm_start_time = time.time()
        start_time = fwhm_start_time
        color_primary = '#ff00ff'  # Magenta
        threshold_excellent = 5
        threshold_good = 10
        ylabel = 'FWHM (px)'
        y_max_default = 15

    if metric_val is None:
        return None

    current_time = time.time() - start_time
    history.append(metric_val)
    times.append(current_time)

    if len(history) < 2:
        return None

    # Limiter la fréquence de mise à jour
    _hfr_fwhm_frame_counter += 1
    if _hfr_fwhm_frame_counter < _graphs_update_interval:
        return None
    _hfr_fwhm_frame_counter = 0

    # Créer ou réutiliser la figure
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#0a0a0a')

    # Tracer les données avec couleurs dynamiques
    times_list = list(times)
    values_list = list(history)

    for i in range(len(values_list) - 1):
        if values_list[i] < threshold_excellent:
            color = '#00ff00'  # vert
        elif values_list[i] < threshold_good:
            color = '#ffff00'  # jaune
        else:
            color = '#ff6600'  # orange

        ax.plot(times_list[i:i+2], values_list[i:i+2],
                color=color, linewidth=2.5, marker='o', markersize=4)

    ax.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, color=color_primary, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric_name} des étoiles', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)

    max_val = max(values_list)
    ax.set_ylim(0, max(max_val * 1.2, y_max_default))

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        plt.close(fig)  # Libérer la mémoire
        return graph_surface
    except:
        plt.close(fig)
        return None

def reset_focus_history():
    """Réinitialise l'historique Focus"""
    global focus_history, focus_times, focus_start_time
    focus_history.clear()
    focus_times.clear()
    focus_start_time = 0

def ghs_stretch(array, D, b, SP, LP, HP):
    """
    Generalized Hyperbolic Stretch (GHS) - Algorithme conforme Siril/PixInsight
    VERSION AMÉLIORÉE: Préserve le bit depth (float32 ou uint8)

    Basé sur les travaux de Dave Payne et Mike Cranfield (ghsastro.co.uk)
    Implémentation conforme à la documentation officielle GHS.

    Args:
        array: numpy array de l'image (H, W, 3) ou (H, W) en uint8 OU float32 [0-255]
        D: Stretch factor (0.0 à 5.0) - force de l'étirement
        b: Local intensity (-5.0 à 15.0) - concentration du contraste autour de SP
        SP: Symmetry point (0.0 à 1.0) - point focal du contraste maximum
        LP: Protect shadows (0.0 à SP) - protection des basses lumières (linéaire)
        HP: Protect highlights (SP à 1.0) - protection des hautes lumières (linéaire)

    Returns:
        numpy array étiré de même dimension (préserve dtype: float32 ou uint8)

    Notes:
        - D = 0 : pas de transformation (identité)
        - b = 0 : transformation exponentielle
        - b = 1 : transformation harmonique (similaire Histogram Transform)
        - b > 1 : transformation hyperbolique (recommandé pour stretch initial)
        - b < 0 : transformation intégrale/logarithmique
        - LP protège les shadows en appliquant une transformation linéaire sous LP
        - HP protège les highlights en appliquant une transformation linéaire au-dessus de HP
        - NOUVEAU: Si entrée float32 → sortie float32 (préserve dynamique RAW12/16)

    Exemple d'utilisation pour galaxie:
        ghs_stretch(img, D=2.0, b=6.0, SP=0.15, LP=0.0, HP=0.85)

    Exemple pour stretch initial (linéaire -> non-linéaire):
        ghs_stretch(img, D=3.5, b=12.0, SP=0.08, LP=0.0, HP=1.0)
    """

    # Détection du type d'entrée pour préserver le bit depth
    input_dtype = array.dtype
    input_is_float = np.issubdtype(input_dtype, np.floating)

    # Normaliser l'image entre 0 et 1
    if input_is_float:
        # Float32 [0-255] provenant de RAW12/16 débayérisé
        img_float = array.astype(np.float64) / 255.0
    else:
        # uint8 [0-255] classique
        img_float = array.astype(np.float64) / 255.0
    
    # Constante pour éviter divisions par zéro
    epsilon = 1e-10
    img_float = np.clip(img_float, epsilon, 1.0 - epsilon)
    
    # Si D = 0, pas de transformation (identité)
    if abs(D) < epsilon:
        return array
    
    # Assurer les contraintes : 0 <= LP <= SP <= HP <= 1
    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))
    
    # =========================================================================
    # FONCTIONS DE TRANSFORMATION DE BASE T(x) selon la valeur de b
    # Source: Documentation GHSAstro - ghsastro.co.uk
    # =========================================================================
    
    def T_base(x, D, b):
        """
        Transformation de base T(x) selon le type déterminé par b
        
        | b value    | Type        | Formula                                    |
        |------------|-------------|---------------------------------------------|
        | b = -1     | Logarithmic | ln(1 + D*x)                                |
        | b < 0      | Integral    | (1 - (1-b*D*x)^((b+1)/b)) / (D*(b+1))     |
        | b = 0      | Exponential | 1 - exp(-D*x)                              |
        | b = 1      | Harmonic    | 1 - (1 + D*x)^(-1)                         |
        | b > 0      | Hyperbolic  | 1 - (1 + b*D*x)^(-1/b)                     |
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        
        if abs(b - (-1.0)) < epsilon:
            # Logarithmic: T(x) = ln(1 + D*x)
            result = np.log1p(D * x)
            
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            # Integral: T(x) = (1 - (1 - b*D*x)^((b+1)/b)) / (D*(b+1))
            base = np.maximum(1.0 - b * D * x, epsilon)
            exponent = (b + 1.0) / b
            result = (1.0 - np.power(base, exponent)) / (D * (b + 1.0))
            
        elif abs(b) < epsilon:
            # Exponential: T(x) = 1 - exp(-D*x)
            result = 1.0 - np.exp(-D * x)
            
        elif abs(b - 1.0) < epsilon:
            # Harmonic: T(x) = 1 - (1 + D*x)^(-1)
            result = 1.0 - 1.0 / (1.0 + D * x)
            
        else:  # b > 0, b != 1
            # Hyperbolic: T(x) = 1 - (1 + b*D*x)^(-1/b)
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = 1.0 - np.power(base, -1.0 / b)
        
        return result
    
    def T_prime(x, D, b):
        """
        Dérivée première T'(x) - nécessaire pour les segments linéaires LP et HP
        
        | b value    | Formula T'(x)                              |
        |------------|---------------------------------------------|
        | b = -1     | D / (1 + D*x)                              |
        | b < 0      | (1 - b*D*x)^(1/b)                          |
        | b = 0      | D * exp(-D*x)                              |
        | b = 1      | D * (1 + D*x)^(-2)                         |
        | b > 0      | D * (1 + b*D*x)^(-(1+b)/b)                 |
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        
        if abs(b - (-1.0)) < epsilon:
            # T'(x) = D / (1 + D*x)
            result = D / (1.0 + D * x)
            
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            # T'(x) = (1 - b*D*x)^(1/b)
            base = np.maximum(1.0 - b * D * x, epsilon)
            result = np.power(base, 1.0 / b)
            
        elif abs(b) < epsilon:
            # T'(x) = D * exp(-D*x)
            result = D * np.exp(-D * x)
            
        elif abs(b - 1.0) < epsilon:
            # T'(x) = D * (1 + D*x)^(-2)
            result = D * np.power(1.0 + D * x, -2.0)
            
        else:  # b > 0, b != 1
            # T'(x) = D * (1 + b*D*x)^(-(1+b)/b)
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = D * np.power(base, -(1.0 + b) / b)
        
        return result
    
    # =========================================================================
    # CONSTRUCTION DE LA TRANSFORMATION COMPLÈTE
    # =========================================================================
    
    # T3(x) = T(x - SP) pour x >= SP (transformation centrée sur SP)
    def T3(x):
        return T_base(x - SP, D, b)
    
    def T3_prime(x):
        return T_prime(x - SP, D, b)
    
    # T2(x) = -T(SP - x) pour LP <= x < SP (symétrie autour de SP)
    def T2(x):
        return -T_base(SP - x, D, b)
    
    def T2_prime(x):
        return T_prime(SP - x, D, b)
    
    # Valeurs aux bornes pour les segments linéaires
    T2_LP = float(T2(LP))
    T2_prime_LP = float(T2_prime(LP))
    T3_HP = float(T3(HP))
    T3_prime_HP = float(T3_prime(HP))
    
    # T1(x) = T2'(LP) * (x - LP) + T2(LP) pour x < LP (linéaire - protection shadows)
    def T1(x):
        return T2_prime_LP * (x - LP) + T2_LP
    
    # T4(x) = T3'(HP) * (x - HP) + T3(HP) pour x >= HP (linéaire - protection highlights)
    def T4(x):
        return T3_prime_HP * (x - HP) + T3_HP
    
    # Valeurs pour la normalisation (transformation doit aller de 0 à 1)
    T1_0 = float(T1(0.0))
    T4_1 = float(T4(1.0))
    norm_range = T4_1 - T1_0
    
    if abs(norm_range) < epsilon:
        return array  # Pas de transformation possible
    
    # =========================================================================
    # APPLICATION DE LA TRANSFORMATION PAR RÉGION
    # =========================================================================
    
    img_stretched = np.zeros_like(img_float)
    
    # Masques pour les 4 régions
    mask1 = img_float < LP                          # Région 1: 0 <= x < LP (linéaire)
    mask2 = (img_float >= LP) & (img_float < SP)    # Région 2: LP <= x < SP (symétrie)
    mask3 = (img_float >= SP) & (img_float < HP)    # Région 3: SP <= x < HP (principale)
    mask4 = img_float >= HP                         # Région 4: HP <= x <= 1 (linéaire)
    
    # Appliquer les transformations par région
    if np.any(mask1):
        img_stretched[mask1] = T1(img_float[mask1])
    if np.any(mask2):
        img_stretched[mask2] = T2(img_float[mask2])
    if np.any(mask3):
        img_stretched[mask3] = T3(img_float[mask3])
    if np.any(mask4):
        img_stretched[mask4] = T4(img_float[mask4])
    
    # Normaliser entre 0 et 1
    img_stretched = (img_stretched - T1_0) / norm_range
    
    # Clip et reconvertir dans le type d'origine (préserve bit depth)
    img_stretched = np.clip(img_stretched, 0.0, 1.0)

    if input_is_float:
        # Retourner float32 [0-255] pour préserver la dynamique RAW12/16
        img_stretched = (img_stretched * 255.0).astype(np.float32)
    else:
        # Retourner uint8 [0-255] pour compatibilité pygame avec YUV420
        img_stretched = (img_stretched * 255.0).astype(np.uint8)

    return img_stretched



def astro_stretch(array):
    """
    Applique un étirement astro selon le preset sélectionné
    - OFF: pas de stretch
    - GHS: Generalized Hyperbolic Stretch
    - Arcsinh: transformation arcsinh classique

    Args:
        array: numpy array de l'image (H, W, 3) en RGB

    Returns:
        numpy array étiré de même dimension
    """
    global stretch_preset, stretch_p_low, stretch_p_high, stretch_factor
    global ghs_D, ghs_b, ghs_SP, ghs_LP, ghs_HP

    if stretch_preset == 0:
        # OFF - pas de stretch
        return array

    elif stretch_preset == 1:
        # GHS stretch - Phase 2 (algorithme conforme Siril/PixInsight)
        D = ghs_D / 10.0       # 0-50 -> 0.0-5.0
        b = ghs_b / 10.0       # -50 à 150 -> -5.0 à 15.0
        SP = ghs_SP / 100.0    # 0-100 -> 0.0-1.0
        LP = ghs_LP / 100.0    # 0-100 -> 0.0-1.0
        HP = ghs_HP / 100.0    # 0-100 -> 0.0-1.0
        return ghs_stretch(array, D, b, SP, LP, HP)

    elif stretch_preset == 2:
        # Arcsinh stretch - VERSION AMÉLIORÉE: Préserve le bit depth (float32 ou uint8)

        # Détection du type d'entrée pour préserver le bit depth
        input_dtype = array.dtype
        input_is_float = np.issubdtype(input_dtype, np.floating)

        # Convertir en float pour les calculs
        img_float = array.astype(np.float32)

        # Calculer les percentiles pour éviter l'effet des pixels chauds
        # stretch_p_low divisé par 10 (stocké x10), stretch_p_high divisé par 100 (stocké x100)
        p_low = np.percentile(img_float, stretch_p_low / 10.0)
        p_high = np.percentile(img_float, stretch_p_high / 100.0)

        # Éviter la division par zéro
        if p_high - p_low < 1:
            return array

        # Normaliser entre 0 et 1
        img_normalized = (img_float - p_low) / (p_high - p_low)
        img_normalized = np.clip(img_normalized, 0, 1)

        # Appliquer une transformation asinh pour accentuer les détails faibles
        # asinh est plus doux que sqrt et préserve mieux les détails
        # Utilise le facteur configurable (divisé par 10 car stocké x10)
        factor = stretch_factor / 10.0

        # Protection contre division par zéro si factor est trop petit
        if factor > 0.01:
            arcsinh_factor = np.arcsinh(factor)
            if arcsinh_factor > 1e-10:
                img_stretched = np.arcsinh(img_normalized * factor) / arcsinh_factor
            else:
                img_stretched = img_normalized  # Pas de stretch si factor trop petit
        else:
            img_stretched = img_normalized  # Pas de stretch si factor <= 0.01

        # Reconvertir dans le type d'origine (préserve bit depth comme GHS)
        if input_is_float:
            # Retourner float32 [0-255] pour préserver la dynamique RAW12/16
            img_stretched = (img_stretched * 255.0).astype(np.float32)
        else:
            # Retourner uint8 [0-255] pour compatibilité pygame avec YUV420
            img_stretched = (img_stretched * 255.0).astype(np.uint8)

        return img_stretched

    else:
        return array

def preview():
    global use_ard,lver,Pi,scientif,scientific,fxx,fxy,fxz,v3_focus,v3_hdr,v3_f_mode,v3_f_modes,prev_fps,focus_fps,focus_mode,restart,datastr
    global count,p, brightness,contrast,modes,mode,red,blue,gain,sspeed,ev,preview_width,preview_height,zoom,igw,igh,zx,zy,awbs,awb,saturations
    global saturation,meters,meter,flickers,flicker,sharpnesss,sharpness,rotate,v3_hdrs,mjpeg_extractor
    global picam2, capture_thread, use_picamera2, Pi_Cam, camera, v3_af, v5_af, vflip, hflip, denoise, denoises, quality, use_native_sensor_mode, zfs
    global livestack_active, luckystack_active, raw_format, raw_stream_size, capture_size

    # Variables statiques pour mémoriser la configuration précédente
    if not hasattr(preview, 'prev_config'):
        preview.prev_config = {}

    # ===== MODE PICAMERA2 =====
    if use_picamera2:
        # Déterminer la taille de capture pour détecter si changement
        # NOUVEAU: IMX585 - Utiliser modes hardware crop
        if Pi_Cam == 10:
            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
            if sensor_mode:
                capture_size = sensor_mode
                raw_stream_size = sensor_mode  # Important: raw stream = sensor mode
            else:
                # Fallback (ne devrait jamais arriver)
                capture_size = (1928, 1090)
                raw_stream_size = capture_size
        # Mode natif avec zoom : calculer la taille ROI native
        elif use_native_sensor_mode == 1 and zoom > 0:
            # Calculer la taille du ROI en pixels natifs du capteur
            # zfs[zoom] donne la fraction de la résolution native
            roi_width = int(igw * zfs[zoom])
            roi_height = int(igh * zfs[zoom])
            # Arrondir à des nombres pairs
            if roi_width % 2 != 0:
                roi_width -= 1
            if roi_height % 2 != 0:
                roi_height -= 1
            capture_size = (roi_width, roi_height)
            # Pour le flux RAW en mode natif : toujours utiliser la résolution native complète
            # même avec zoom. Le ScalerCrop définit le ROI au niveau capteur (→ cadence augmentée)
            # Après capture, on extraira la région ROI du tableau RAW
            raw_stream_size = (igw, igh)
        # Mode natif sans zoom : utiliser la résolution native complète
        elif use_native_sensor_mode == 1:
            # Utiliser la résolution native complète du capteur
            capture_size = (igw, igh)
            raw_stream_size = capture_size
        # Si zoom actif en mode binning, utiliser la résolution correspondante au niveau de zoom
        elif zoom > 0:
            # Résolutions correspondant aux niveaux de zoom (ordre décroissant)
            zoom_capture_sizes = {
                1: (2880, 2160),  # 2x zoom (résolution la plus haute)
                2: (1920, 1080),  # 3x zoom
                3: (1280, 720),   # 4x zoom
                4: (800, 600),    # 5x zoom
                5: (800, 600)     # 6x zoom (même résolution que zoom 4)
            }
            capture_size = zoom_capture_sizes.get(zoom, (vwidth, vheight))
            raw_stream_size = capture_size
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and focus_mode == 1:
            capture_size = (3280, 2464)
            raw_stream_size = capture_size
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
            capture_size = (1920, 1440)
            raw_stream_size = capture_size
        elif Pi_Cam == 3:
            capture_size = (2304, 1296)
            raw_stream_size = capture_size
        else:
            capture_size = (preview_width, preview_height)
            raw_stream_size = capture_size

        # Détecter si recréation nécessaire (changements majeurs de config)
        # Note: raw_format et v3_hdr nécessitent recréation car ils changent le stream RAW (SRGGB12/16) et HdrMode
        # Note: stacking_active change nécessite recréation pour basculer entre stream 'raw' et 'main'
        current_stacking_active = livestack_active or luckystack_active
        prev_stacking_active = preview.prev_config.get('stacking_active', False)

        need_recreation = (
            picam2 is None or
            preview.prev_config.get('camera') != camera or
            preview.prev_config.get('capture_size') != capture_size or
            preview.prev_config.get('vflip') != vflip or
            preview.prev_config.get('hflip') != hflip or
            preview.prev_config.get('mode_type') != (0 if mode == 0 or sspeed > 80000 else 1) or
            preview.prev_config.get('use_native_sensor_mode') != use_native_sensor_mode or
            preview.prev_config.get('awb') != awb or  # Changement AWB nécessite recréation (ColourGains persistent)
            preview.prev_config.get('raw_format') != raw_format or  # Changement RAW12/16 nécessite recréation
            preview.prev_config.get('v3_hdr') != v3_hdr or  # Changement mode HDR nécessite recréation
            (raw_format >= 2 and prev_stacking_active != current_stacking_active)  # Changement mode stacking avec RAW
        )

        # Debug: afficher si le changement de mode stacking force la recréation
        if raw_format >= 2 and prev_stacking_active != current_stacking_active:
            print(f"  [STACKING MODE CHANGE] {prev_stacking_active} -> {current_stacking_active}, forcing full recreation")

        # Calculer speed2 et autres paramètres (avant le if/else)
        speed2 = sspeed
        max_exposure_seconds = max_shutters[Pi_Cam]
        max_frame_duration = int(max_exposure_seconds * 1_000_000)
        min_frame_duration = 11415 if Pi_Cam == 10 else 100

        # ========== CHEMIN RAPIDE : Juste changer les contrôles ==========
        if not need_recreation and picam2 is not None:
            if show_cmds == 1:
                print(f"  [FAST PATH] Just updating controls (no recreation)...")

            # Préparer les contrôles à changer
            fast_controls = {}

            # Exposition et gain (mode manuel)
            if mode == 0:
                fast_controls["FrameDurationLimits"] = (min_frame_duration, max(max_frame_duration, speed2))
                fast_controls["ExposureTime"] = speed2
                fast_controls["AnalogueGain"] = float(gain)

            # Brightness & Contrast
            fast_controls["Brightness"] = brightness / 100
            fast_controls["Contrast"] = contrast / 100

            # AWB
            if awb == 0:
                fast_controls["ColourGains"] = (red/10, blue/10)
                if show_cmds == 1:
                    print(f"  DEBUG AWB: Mode manuel (awb=0), ColourGains=({red/10}, {blue/10})")
            else:
                awb_modes = {
                    1: controls.AwbModeEnum.Auto, 2: controls.AwbModeEnum.Incandescent,
                    3: controls.AwbModeEnum.Tungsten, 4: controls.AwbModeEnum.Fluorescent,
                    5: controls.AwbModeEnum.Indoor, 6: controls.AwbModeEnum.Daylight, 7: controls.AwbModeEnum.Cloudy,
                }
                if awb in awb_modes:
                    fast_controls["AwbMode"] = awb_modes[awb]
                    if show_cmds == 1:
                        print(f"  DEBUG AWB: Mode auto (awb={awb}={awbs[awb]}), AwbMode={awb_modes[awb]}")
                else:
                    if show_cmds == 1:
                        print(f"  DEBUG AWB: ERREUR - awb={awb} non trouvé dans awb_modes!")

            # Saturation & Sharpness
            fast_controls["Saturation"] = saturation / 10
            fast_controls["Sharpness"] = sharpness / 10

            # HDR Mode (pour IMX585 Clear HDR)
            # Multi-Exp incompatible mode manuel, autres OK
            if v3_hdr > 0 and Pi_Cam == 10:
                if v3_hdr == 4:
                    # Clear HDR 16-bit : désactiver HdrMode (traitement capteur)
                    fast_controls["HdrMode"] = controls.HdrModeEnum.Off
                elif v3_hdr == 2 and mode == 0:
                    # Multi-Exp en mode manuel : incompatible
                    fast_controls["HdrMode"] = controls.HdrModeEnum.Off
                else:
                    # Activer HDR (Single-Exp, Multi-Exp, Night)
                    hdr_modes = {
                        1: controls.HdrModeEnum.SingleExposure,
                        2: controls.HdrModeEnum.MultiExposure,
                        3: controls.HdrModeEnum.Night
                    }
                    if v3_hdr in hdr_modes:
                        fast_controls["HdrMode"] = hdr_modes[v3_hdr]
            elif v3_hdr == 0:
                fast_controls["HdrMode"] = controls.HdrModeEnum.Off

            # Appliquer tous les contrôles en une seule fois (rapide!)
            try:
                if show_cmds == 1:
                    print(f"  DEBUG: fast_controls = {fast_controls}")
                picam2.set_controls(fast_controls)

                # Les contrôles sont appliqués immédiatement par set_controls()
                # Pas besoin de redémarrer le thread (sauf si changement de type de capture raw/main)

                # Vérifier que les contrôles ont été appliqués
                if show_cmds == 1:
                    metadata = picam2.capture_metadata()
                    actual_awb = metadata.get("AwbMode", "N/A")
                    actual_gains = metadata.get("ColourGains", "N/A")
                    print(f"  ✓ Controls updated instantly - ExposureTime={speed2}µs, Gain={gain}")
                    print(f"  → Camera reports: AwbMode={actual_awb}, ColourGains={actual_gains}")

                # *** IMPORTANT: Mettre à jour capture_thread pour le mode stacking ***
                # Même en fast path, on doit s'assurer que le capture_thread est configuré correctement
                if capture_thread is not None:
                    if (livestack_active or luckystack_active) and raw_format >= 2:
                        capture_thread.set_capture_params({'type': 'raw'})
                        if show_cmds == 1:
                            print(f"  → Capture thread configuré en mode RAW (stacking actif)")
                    else:
                        capture_thread.set_capture_params({'type': 'main'})
                        if show_cmds == 1:
                            print(f"  → Capture thread configuré en mode MAIN")

                # Pas besoin de mémoriser la config car elle n'a pas changé
                restart = 0
                return  # Early return - évite toute la recréation !
            except RuntimeError as e:
                # Si les contrôles ne sont pas disponibles (ex: après rpicam-still),
                # forcer une recréation complète
                if show_cmds == 1:
                    print(f"  ⚠ Fast path failed ({e}), forcing full recreation...")
                need_recreation = True  # Forcer la recréation
                # Continue vers le chemin complet ci-dessous

        # ========== CHEMIN COMPLET : Recréation nécessaire ==========
        if show_cmds == 1:
            print(f"  [FULL RECREATION] Config changed - recreating Picamera2...")

            # Arrêter l'ancienne instance
            if picam2 is not None:
                try:
                    picam2.stop()
                    picam2.close()
                    time.sleep(0.5)
                except:
                    pass

            # Créer nouvelle instance
            picam2 = Picamera2(camera)

        # Déterminer la taille de capture selon caméra (pour les deux chemins)
        # IMX585 : Utiliser TOUJOURS les modes sensor hardware (crop ou full frame)
        if Pi_Cam == 10:
            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
            if sensor_mode:
                capture_size = sensor_mode  # Modes hardware: crop ou full frame
                raw_stream_size = sensor_mode  # Important: raw stream = sensor mode
            else:
                capture_size = (1928, 1090)  # Fallback binning
                raw_stream_size = capture_size
        # Mode natif avec zoom : calculer la taille ROI native (AUTRES caméras)
        elif use_native_sensor_mode == 1 and zoom > 0:
            # Calculer la taille du ROI en pixels natifs du capteur
            # zfs[zoom] donne la fraction de la résolution native
            roi_width = int(igw * zfs[zoom])
            roi_height = int(igh * zfs[zoom])
            # Arrondir à des nombres pairs
            if roi_width % 2 != 0:
                roi_width -= 1
            if roi_height % 2 != 0:
                roi_height -= 1
            capture_size = (roi_width, roi_height)
        # Mode natif sans zoom : utiliser la résolution native complète
        elif use_native_sensor_mode == 1:
            # Utiliser la résolution native complète du capteur
            capture_size = (igw, igh)
        # Si zoom actif en mode binning, utiliser la résolution correspondante au niveau de zoom
        elif zoom > 0:
            zoom_capture_sizes = {
                1: (2880, 2160),  # 2x zoom (résolution la plus haute)
                2: (1920, 1080),  # 3x zoom
                3: (1280, 720),   # 4x zoom
                4: (800, 600),    # 5x zoom
                5: (800, 600)     # 6x zoom (même résolution que zoom 4)
            }
            capture_size = zoom_capture_sizes.get(zoom, (vwidth, vheight))
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and focus_mode == 1:
            capture_size = (3280, 2464)
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
            capture_size = (1920, 1440)
        elif Pi_Cam == 3:  # Pi v3
            capture_size = (2304, 1296)
        elif Pi_Cam == 10: # imx585
            capture_size = (1928, 1090)
        else:
            capture_size = (preview_width, preview_height)

        # Configuration de base
        # IMPORTANT: Forcer le mode 12-bit au lieu de 16-bit pour que les contrôles fonctionnent
        # Le mode 16-bit de l'IMX585 ne supporte pas les contrôles automatiques
        from picamera2 import Preview

        # Calculer speed2 (comme dans le code rpicam-vid)
        speed2 = sspeed
        # Limite retirée pour permettre les longues expositions
        # speed2 = min(speed2, 2000000)

        # FrameDurationLimits : Calculer basé sur les capacités du capteur
        # Pour IMX585: min=11415µs, max=163070574µs (~163s)
        max_exposure_seconds = max_shutters[Pi_Cam]
        max_frame_duration = int(max_exposure_seconds * 1_000_000)  # Convertir en µs
        # Utiliser le minimum du capteur IMX585, pas une valeur arbitraire
        min_frame_duration = 11415 if Pi_Cam == 10 else 100  # 11.415ms pour IMX585

        # Afficher les paramètres de configuration (traceur)
        if show_cmds == 1:
            sensor_mode_name = "NATIVE" if use_native_sensor_mode == 1 else "BINNING"
            stacking_status = "STACKING ACTIF" if (livestack_active or luckystack_active) else "PREVIEW"
            print(f"  [CONFIG] Mode: {stacking_status}, Sensor: {sensor_mode_name}, Size: {capture_size}")
            print(f"  [CONFIG] RAW Format: {raw_formats[raw_format]} (index {raw_format})")

        # Créer la configuration adaptée au mode
        # En mode manuel ou avec exposition > 80ms, utiliser still_configuration
        # (preview_configuration limite l'exposition à ~83ms pour l'IMX585)
        # IMPORTANT: Format RAW dépend du mode sélectionné:
        # - raw_format=0 (YUV420): Pas de stream raw, ISP → YUV420
        # - raw_format=1 (XRGB8888): Pas de stream raw, ISP → XRGB8888
        # - raw_format=2 (RAW12): Stream raw SRGGB12, bypass ISP
        # - raw_format=3 (RAW16): Stream raw SRGGB16 + activation automatique Clear HDR

        # Déterminer le format MAIN et RAW à utiliser
        main_format = "RGB888"  # Par défaut
        raw_stream_format = None  # Pas de stream raw par défaut
        use_raw_stream = False

        # Sélectionner format selon raw_format
        if raw_format == 1:
            # XRGB8888 : format ISP direct, pas de stream RAW
            main_format = "XRGB8888"
            use_raw_stream = False
        elif raw_format == 2:
            # RAW12 Bayer
            main_format = "RGB888"
            raw_stream_format = "SRGGB12"
            use_raw_stream = True
        elif raw_format == 3:
            # RAW16 Clear HDR
            main_format = "RGB888"
            raw_stream_format = "SRGGB16"
            use_raw_stream = True

        # AUTO-LINKING : Clear HDR 16-bit <-> RAW16
        # LOGIQUE UNIDIRECTIONNELLE (corrigée) :
        # - Clear HDR (v3_hdr=4) REQUIERT RAW16 → force raw_format=3
        # - RAW16 (raw_format=3) PERMET tous les modes HDR (0,1,2,3,4) → ne force rien
        # Ceci permet d'utiliser RAW16 avec OFF/Single/Multi/Night sans verrouillage

        if v3_hdr == 4:
            # Clear HDR 16-bit activé → FORCER RAW16 (obligatoire)
            if raw_format != 3:
                print(f"  [AUTO-CONFIG] Clear HDR 16-bit activé → passage en RAW16 (raw_format: {raw_format} → 3)")
                raw_format = 3
                main_format = "RGB888"
                raw_stream_format = "SRGGB16"
                use_raw_stream = True

        # Créer la configuration selon le type de stream nécessaire
        # IMPORTANT: Pour IMX585, spécifier TOUJOURS raw["size"] pour forcer le mode sensor
        # Cela remplace ScalerCrop par sélection native du mode hardware
        if mode == 0 or sspeed > 80000:
            if Pi_Cam == 10:  # IMX585
                sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                if use_raw_stream:
                    # IMPORTANT: Ajouter "unpacked": True pour éviter le format compressé PISP_COMP1
                    # qui cause une résolution réduite lors du debayering
                    config = picam2.create_still_configuration(
                        main={"size": capture_size, "format": main_format},
                        raw={"size": sensor_mode, "format": raw_stream_format, "unpacked": True}
                    )
                else:
                    # Même sans raw stream, spécifier raw size force le mode
                    # IMPORTANT: Ajouter "unpacked": True pour éviter PISP_COMP1 compressé
                    config = picam2.create_still_configuration(
                        main={"size": capture_size, "format": main_format},
                        raw={"size": sensor_mode, "unpacked": True}
                    )
            elif use_raw_stream:
                # IMPORTANT: Ajouter "unpacked": True pour éviter le format compressé
                config = picam2.create_still_configuration(
                    main={"size": capture_size, "format": main_format},
                    raw={"size": raw_stream_size, "format": raw_stream_format, "unpacked": True}
                )
            else:
                config = picam2.create_still_configuration(
                    main={"size": capture_size, "format": main_format}
                )
            if show_cmds == 1:
                print(f"  [CONFIG] Type: STILL, Main format: {main_format}")
                if use_raw_stream or Pi_Cam == 10:
                    print(f"  [CONFIG] Stream RAW: {raw_stream_format if use_raw_stream else 'sensor mode forced'}")
                print(f"  [CONFIG] Requested capture_size: {capture_size}")
        else:
            if Pi_Cam == 10:  # IMX585
                sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                if use_raw_stream:
                    # IMPORTANT: Ajouter "unpacked": True pour éviter le format compressé PISP_COMP1
                    config = picam2.create_preview_configuration(
                        main={"size": capture_size, "format": main_format},
                        raw={"size": sensor_mode, "format": raw_stream_format, "unpacked": True}
                    )
                else:
                    # Même sans raw stream, spécifier raw size force le mode
                    # IMPORTANT: Ajouter "unpacked": True pour éviter PISP_COMP1 compressé
                    config = picam2.create_preview_configuration(
                        main={"size": capture_size, "format": main_format},
                        raw={"size": sensor_mode, "unpacked": True}
                    )
            elif use_raw_stream:
                # IMPORTANT: Ajouter "unpacked": True pour éviter le format compressé
                config = picam2.create_preview_configuration(
                    main={"size": capture_size, "format": main_format},
                    raw={"size": raw_stream_size, "format": raw_stream_format, "unpacked": True}
                )
            else:
                config = picam2.create_preview_configuration(
                    main={"size": capture_size, "format": main_format}
                )
            if show_cmds == 1:
                print(f"  [CONFIG] Type: PREVIEW, Main format: {main_format}")
                if use_raw_stream or Pi_Cam == 10:
                    print(f"  [CONFIG] Stream RAW: {raw_stream_format if use_raw_stream else 'sensor mode forced'}")
                print(f"  [CONFIG] Requested capture_size: {capture_size}")

        # Préparer TOUS les contrôles pour application après start()
        controls_dict = {}

        # Brightness & Contrast
        controls_dict["Brightness"] = brightness / 100
        controls_dict["Contrast"] = contrast / 100

        # Exposition et mode
        if mode == 0:
            # Mode manuel
            # IMPORTANT: Appliquer FrameDurationLimits dynamiquement pour éviter que la caméra
            # garde un FrameDuration élevé après une exposition longue
            controls_dict["FrameDurationLimits"] = (min_frame_duration, max(max_frame_duration, speed2))
            controls_dict["ExposureTime"] = speed2
            controls_dict["AnalogueGain"] = float(gain)
        else:
            # Mode auto - ne rien définir, laisser l'AE gérer
            pass

        # Framerate
        # NE PAS définir FrameRate en mode manuel pour permettre les longues expositions
        if mode != 0:
            if zoom > 0:
                fps = focus_fps
            else:
                fps = prev_fps
            controls_dict["FrameRate"] = fps

        # AWB (Balance des blancs)
        # Ne PAS utiliser AwbEnable qui n'est pas supporté sur toutes les caméras
        if awb == 0:
            # AWB manuel - définir ColourGains désactive automatiquement AWB
            controls_dict["ColourGains"] = (red/10, blue/10)
            if show_cmds == 1:
                print(f"  DEBUG AWB [FULL]: Mode manuel (awb=0), ColourGains=({red/10}, {blue/10})")
        else:
            # AWB auto - utiliser AwbMode
            awb_modes = {
                1: controls.AwbModeEnum.Auto,
                2: controls.AwbModeEnum.Incandescent,
                3: controls.AwbModeEnum.Tungsten,
                4: controls.AwbModeEnum.Fluorescent,
                5: controls.AwbModeEnum.Indoor,
                6: controls.AwbModeEnum.Daylight,
                7: controls.AwbModeEnum.Cloudy,
            }
            if awb in awb_modes:
                controls_dict["AwbMode"] = awb_modes[awb]
                if show_cmds == 1:
                    print(f"  DEBUG AWB [FULL]: Mode auto (awb={awb}={awbs[awb]}), AwbMode={awb_modes[awb]}")
            else:
                if show_cmds == 1:
                    print(f"  DEBUG AWB [FULL]: ERREUR - awb={awb} non trouvé dans awb_modes!")

        # Saturation & Sharpness
        controls_dict["Saturation"] = saturation / 10
        controls_dict["Sharpness"] = sharpness / 10

        # HDR Mode (pour IMX585 Clear HDR)
        # 0=Off, 1=SingleExposure, 2=MultiExposure, 3=Night, 4=Clear HDR 16-bit (capteur IMX585)
        # IMPORTANT: MultiExposure et Night nécessitent AE/AGC automatique
        # SingleExposure peut fonctionner en mode manuel (exposition fixe)
        if v3_hdr > 0 and Pi_Cam == 10:  # Seulement pour IMX585
            if v3_hdr == 4:
                # Clear HDR 16-bit : le capteur IMX585 gère le HDR en interne
                # On désactive le HdrMode de libcamera pour éviter un double traitement
                controls_dict["HdrMode"] = controls.HdrModeEnum.Off
                if show_cmds == 1:
                    print(f"  [HDR] Clear HDR 16-bit: HdrMode=Off (traitement capteur IMX585)")
            elif (v3_hdr == 2 or v3_hdr == 3) and mode == 0:
                # Multi-Exp et Night en mode manuel : incompatibles (nécessitent AE/AGC)
                controls_dict["HdrMode"] = controls.HdrModeEnum.Off
                if show_cmds == 1:
                    mode_name = "Multi-Exp" if v3_hdr == 2 else "Night"
                    print(f"  [HDR] {mode_name} désactivé en mode manuel (nécessite AE/AGC)")
                    print(f"       → Utilisez mode Auto, ou Single-Exp/Clear HDR en manuel")
            else:
                # Autres modes HDR : compatible manuel et auto
                hdr_modes = {
                    1: controls.HdrModeEnum.SingleExposure,
                    2: controls.HdrModeEnum.MultiExposure,
                    3: controls.HdrModeEnum.Night
                }
                if v3_hdr in hdr_modes:
                    controls_dict["HdrMode"] = hdr_modes[v3_hdr]
                    if show_cmds == 1:
                        print(f"  [HDR] Mode activé: {v3_hdrs[v3_hdr]} (HdrMode={hdr_modes[v3_hdr]})")
                        if mode != 0 and (v3_hdr == 2 or v3_hdr == 3):
                            print(f"       ⚠ Mode Auto: AE/AGC contrôle l'exposition et le gain")
        elif v3_hdr == 0:
            # Explicitement désactiver HDR
            controls_dict["HdrMode"] = controls.HdrModeEnum.Off
            if show_cmds == 1 and Pi_Cam == 10:
                print(f"  [HDR] Mode désactivé (HdrMode=Off)")

        # Focus (autofocus ou manuel) - sera appliqué séparément après start()
        focus_controls = {}
        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
            if v3_f_mode == 1:  # Manuel
                focus_controls["AfMode"] = controls.AfModeEnum.Manual
                if Pi_Cam == 3:
                    focus_controls["LensPosition"] = v3_focus / 100
                elif Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                    focus_controls["LensPosition"] = focus / 100
            else:  # Auto
                focus_controls["AfMode"] = controls.AfModeEnum.Auto

        # Transform (flip)
        if vflip == 1 or hflip == 1:
            config["transform"] = Transform(vflip=(vflip == 1), hflip=(hflip == 1))

        # DEBUG: Afficher la config réelle avant de la passer à picamera2
        if show_cmds == 1:
            print(f"  [DEBUG] Config avant picam2.configure():")
            print(f"    main: {config.get('main', 'N/A')}")
            print(f"    raw: {config.get('raw', 'N/A')}")
            if 'raw' in config and config['raw'] is not None:
                print(f"    ⚠ WARNING: Config contient un stream raw alors qu'on ne le voulait pas!")

        picam2.configure(config)

        # DEBUG: Vérifier les tailles réelles après configuration
        if show_cmds == 1:
            actual_config = picam2.camera_configuration()
            print(f"  [DEBUG] Config APRÈS picam2.configure():")
            for stream_name in ['main', 'raw']:
                if stream_name in actual_config:
                    stream_cfg = actual_config[stream_name]
                    print(f"    {stream_name}: size={stream_cfg.get('size', 'N/A')}, format={stream_cfg.get('format', 'N/A')}")

        # CRITIQUE: En mode manuel, appliquer TOUS les contrôles AVANT start()
        # Cela évite que libcamera garde un état résiduel d'une configuration précédente
        if mode == 0:
            initial_frame_duration = (min_frame_duration, max(max_frame_duration, speed2))
            pre_start_controls = {
                "FrameDurationLimits": initial_frame_duration,
                "ExposureTime": speed2,
                "AnalogueGain": float(gain)
            }
            picam2.set_controls(pre_start_controls)
            if show_cmds == 1:
                print(f"  Pre-start controls: FDL={initial_frame_duration[0]}-{initial_frame_duration[1]}µs, Exp={speed2}µs, Gain={gain}")
        else:
            initial_frame_duration = (min_frame_duration, max(max_frame_duration, speed2))
            picam2.set_controls({"FrameDurationLimits": initial_frame_duration})
            if show_cmds == 1:
                print(f"  Pre-start FrameDurationLimits: {initial_frame_duration[0]}µs to {initial_frame_duration[1]}µs")

        picam2.start()

        # Attendre que la caméra démarre (réduit pour accélérer)
        time.sleep(0.1)

        # APPLIQUER TOUS LES CONTRÔLES APRÈS LE START
        # Pour l'IMX585, c'est nécessaire pour que les contrôles soient correctement appliqués
        try:
            if show_cmds == 1:
                print(f"Applying controls: ExposureTime={speed2}µs, Gain={controls_dict.get('AnalogueGain')}, Mode={mode}")
                fd_limits = controls_dict.get('FrameDurationLimits', (min_frame_duration, max_frame_duration))
                print(f"  FrameDurationLimits: {fd_limits[0]}µs to {fd_limits[1]}µs ({fd_limits[1]/1000000:.1f}s max)")

            # En mode manuel, appliquer ExposureTime et AnalogueGain ensemble
            # Cela désactive automatiquement l'auto-exposition
            if mode == 0:
                exposure_controls = {
                    "FrameDurationLimits": controls_dict["FrameDurationLimits"],
                    "ExposureTime": controls_dict["ExposureTime"],
                    "AnalogueGain": controls_dict["AnalogueGain"]
                }

                picam2.set_controls(exposure_controls)

                # Suppression des étapes de stabilisation/réapplication pour expositions longues
                # (trop lent, pas nécessaire)
                time.sleep(0.1)

                # Appliquer les autres contrôles
                other_controls = {k: v for k, v in controls_dict.items()
                                if k not in ["FrameDurationLimits", "ExposureTime", "AnalogueGain"]}
                if other_controls:
                    picam2.set_controls(other_controls)
                    time.sleep(0.05)
            else:
                # Mode auto : appliquer tous les contrôles ensemble
                picam2.set_controls(controls_dict)
                time.sleep(0.1)

            # Appliquer les contrôles de focus séparément si nécessaire
            if focus_controls:
                picam2.set_controls(focus_controls)
                time.sleep(0.05)

            # ===== ZOOM HARDWARE POUR IMX585 / SCALERCROP POUR AUTRES CAMÉRAS =====
            if Pi_Cam == 10 and zoom > 0:
                # IMX585: Le mode sensor a déjà été sélectionné via raw["size"]
                # Pas besoin de ScalerCrop
                if show_cmds == 1:
                    sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                    mode_name = imx585_crop_modes[zoom][2] if zoom in imx585_crop_modes and imx585_crop_modes[zoom] else "Full"
                    print(f"  Zoom hardware IMX585: {mode_name} @ {sensor_mode}")
            elif zoom > 0 and zoom <= 5:  # Autres caméras - Zoom 3 désactivé
                # Autres caméras: conserver ScalerCrop
                try:
                    # Obtenir la taille native du capteur pour ScalerCrop
                    # ScalerCrop utilise les coordonnées absolues du capteur natif
                    if Pi_Cam == 3:  # Pi v3
                        sensor_width = 4608
                        sensor_height = 2592
                    elif Pi_Cam == 4:  # Pi HQ
                        sensor_width = 4056
                        sensor_height = 3040
                    else:
                        sensor_width = 3280
                        sensor_height = 2464

                    # Calculer la région crop (même logique que le ROI pour rpicam-vid)
                    # Arrondir à des nombres pairs
                    crop_width = int(sensor_width * zfs[zoom])
                    crop_height = int(sensor_height * zfs[zoom])
                    if crop_width % 2 != 0:
                        crop_width -= 1
                    if crop_height % 2 != 0:
                        crop_height -= 1

                    # Centrer le crop
                    crop_x = (sensor_width - crop_width) // 2
                    crop_y = (sensor_height - crop_height) // 2

                    # Appliquer ScalerCrop
                    scaler_crop = (crop_x, crop_y, crop_width, crop_height)
                    picam2.set_controls({"ScalerCrop": scaler_crop})
                    time.sleep(0.05)

                    if show_cmds == 1:
                        zoom_factor = 1.0 / zfs[zoom]
                        print(f"  Zoom preview: {zoom_factor:.1f}x applied via ScalerCrop")
                        print(f"    ScalerCrop: {scaler_crop} (x,y,w,h)")
                except Exception as e:
                    if show_cmds == 1:
                        print(f"  Warning: ScalerCrop application failed: {e}")

        except Exception as e:
            if show_cmds == 1:
                print(f"Warning: Control application failed: {e}")

        if show_cmds == 1:
            print("✓ Picamera2 started successfully")
            print(f"  Capture size: {capture_size}")
            print(f"  Mode: {mode} (0=manuel, other=auto)")
            print(f"  ExposureTime demandé: {speed2}µs ({speed2/1000}ms)")
            print(f"  Gain demandé: {gain}")

            # Afficher les limites des contrôles disponibles
            camera_controls = picam2.camera_controls
            if "ExposureTime" in camera_controls:
                exp_limits = camera_controls["ExposureTime"]
                print(f"  Camera ExposureTime limits: {exp_limits}")
            if "FrameDurationLimits" in camera_controls:
                fd_limits = camera_controls["FrameDurationLimits"]
                print(f"  Camera FrameDurationLimits: {fd_limits}")

            # Lire les contrôles effectifs après application
            # Délai réduit pour accélérer
            time.sleep(0.1)
            actual_controls = picam2.capture_metadata()
            if actual_controls:
                actual_exp = actual_controls.get('ExposureTime')
                actual_gain = actual_controls.get('AnalogueGain')
                actual_fd = actual_controls.get('FrameDuration')
                print(f"  ExposureTime réel: {actual_exp}µs ({actual_exp/1000:.2f}ms)")
                print(f"  Gain réel: {actual_gain:.2f}")
                if actual_fd:
                    print(f"  FrameDuration réel: {actual_fd}µs ({actual_fd/1000:.2f}ms)")
                # Vérifier si les valeurs sont proches (seulement en mode manuel)
                if mode == 0 and abs(actual_exp - speed2) > speed2 * 0.1:  # Plus de 10% de différence
                    print(f"  ⚠ WARNING: ExposureTime réel différent de la demande!")
                    print(f"    Écart: {speed2 - actual_exp}µs ({100*(speed2-actual_exp)/speed2:.1f}%)")

        # Créer et démarrer le thread de capture asynchrone
        if capture_thread is not None:
            # Arrêter l'ancien thread s'il existe
            capture_thread.stop()

        capture_thread = AsyncCaptureThread(picam2)
        # Définir les paramètres de capture selon le format RAW
        if (livestack_active or luckystack_active) and raw_format >= 2:
            capture_thread.set_capture_params({'type': 'raw'})
        else:
            capture_thread.set_capture_params({'type': 'main'})
        capture_thread.start()

        # Mémoriser la configuration actuelle pour la prochaine fois
        preview.prev_config = {
            'camera': camera,
            'capture_size': capture_size,
            'vflip': vflip,
            'hflip': hflip,
            'mode_type': 0 if mode == 0 or sspeed > 80000 else 1,
            'use_native_sensor_mode': use_native_sensor_mode,
            'awb': awb,
            'raw_format': raw_format,
            'v3_hdr': v3_hdr,
            'stacking_active': livestack_active or luckystack_active
        }

        restart = 0
        # Suppression du sleep final pour accélérer
        return

    # ===== MODE RPICAM-VID (code original) =====
    # Nettoyer les anciens fichiers et processus
    files = glob.glob('/run/shm/*.jpg')
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists('/run/shm/stream.mjpeg'):
        os.remove('/run/shm/stream.mjpeg')
    
    # Arrêter l'ancien extracteur s'il existe
    if mjpeg_extractor is not None:
        mjpeg_extractor.stop()
        mjpeg_extractor = None
    speed2 = sspeed
    # Limite retirée pour permettre les longues expositions
    # speed2 = min(speed2,2000000)
    if lver != "bookwo" and lver != "trixie":
        datastr = "libcamera-vid"
    else:
        datastr = "rpicam-vid"

    # RETOUR À LA MÉTHODE ÉPROUVÉE : --segment 1 (programme original ligne 704)
    # --segment 1 = 1ms, force un nouveau fichier à chaque frame
    datastr += " --camera " + str(camera) + " -n --codec mjpeg -t 0 --segment 1"

    # SORTIE DIRECTE VERS FICHIERS JPEG (méthode originale simple et stable)
    # Si zoom actif, utiliser la résolution correspondante au niveau de zoom (ordre décroissant)
    if zoom > 0:
        zoom_cmd_resolutions = {
            1: " --width 2880 --height 2160 -o /run/shm/test%04d.jpg ",
            2: " --width 1920 --height 1080 -o /run/shm/test%04d.jpg ",
            3: " --width 1280 --height 720 -o /run/shm/test%04d.jpg ",
            4: " --width 800 --height 600 -o /run/shm/test%04d.jpg ",
            5: " --width 800 --height 600 -o /run/shm/test%04d.jpg "
        }
        datastr += zoom_cmd_resolutions.get(zoom, f" --width {vwidth} --height {vheight} -o /run/shm/test%04d.jpg ")
    elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and focus_mode == 1:
        datastr += " --width 3280 --height 2464 -o /run/shm/test%04d.jpg "
    elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
        datastr += " --width 1920 --height 1440 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 3:  # Pi v3
        datastr += " --width 2304 --height 1296 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 7:  # Pi GS
        datastr += " --width 1456 --height 1088 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 9:  # imx290
        datastr += " --width 1920 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 10: # imx585
        datastr += " --width 1928 --height 1090 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 11: # imx293
        datastr += " --width 1920 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 12: # imx294
        datastr += " --width 2048 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 13: # imx283
        datastr += " --width 1920 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 14: # imx500
        datastr += " --width 2028 --height 1520 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 15: # ov9281
        datastr += " --width 1280 --height  800 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 1:  # v1 / ov5647
        datastr += " --width 1296 --height 972 -o /run/shm/test%04d.jpg "
    else:
        if preview_width == 640 and preview_height == 480:
            datastr += " --width 720 --height 540 -o /run/shm/test%04d.jpg "
        else:
            # Pour un affichage 880x580, forcer le mode de capteur le plus bas : 1280x720
            # --mode force le capteur à utiliser ce mode natif (au lieu de 3856x2180 + downscale)
            # Format: largeur:hauteur:profondeur_bits (10 ou 12 bits selon capteur)
            datastr += " --mode 1280:720:10 --width 1280 --height 720 -o /run/shm/test%04d.jpg "
    if ev != 0:
        datastr += " --ev " + str(ev)
    datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
    if mode == 0:
        datastr += " --shutter " + str(speed2)
    else:
        datastr += " --exposure " + str(modes[mode])
    # Framerate (méthode identique au programme original ligne 741-748)
    if zoom > 0 and mode != 0:
        datastr += " --framerate " + str(focus_fps)
    elif mode != 0:
        datastr += " --framerate " + str(prev_fps)
    elif mode == 0:
        # Calculer FPS basé sur l'exposition (protection division par zéro)
        if speed2 > 0:
            speed3 = max(min(1000000/speed2, 25), 0.01)
        else:
            speed3 = 25
        datastr += " --framerate " + str(speed3)
    if sspeed > 5000000 and mode == 0:
        datastr += " --gain 1 --awbgains " + str(red/10) + "," + str(blue/10)
    else:
        datastr += " --gain " + str(gain)
        if awb == 0:
            datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
        else:
            datastr += " --awb " + awbs[awb]
    datastr += " --metering "   + meters[meter]
    datastr += " --saturation " + str(saturation/10)
    datastr += " --sharpness "  + str(sharpness/10)
    datastr += " --denoise "    + denoises[denoise]
    datastr += " --quality " + str(quality)
    if vflip == 1:
        datastr += " --vflip"
    if hflip == 1:
        datastr += " --hflip"
    if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
        if v3_f_mode == 1:
            if Pi_Cam == 3:
                datastr += " --lens-position " + str(v3_focus/100)
            if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                datastr += " --lens-position " + str(focus/100)
    if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0 and fxx != 0 and v3_f_mode != 1:
        datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
    if (Pi_Cam == 3 and v3_af == 1) and v3_f_speed != 0:
        datastr += " --autofocus-speed " + v3_f_speeds[v3_f_speed]
    if (Pi_Cam == 3 and v3_af == 1) and v3_f_range != 0:
        datastr += " --autofocus-range " + v3_f_ranges[v3_f_range]
    if Pi_Cam == 3 or Pi == 5:
        datastr += " --hdr " + v3_hdrs_cli[v3_hdr]
    if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
    if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"
    if Pi_Cam == 4 and scientific == 1:
        if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
        if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
    # Zoom fixe 1x à 6x
    if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop modes
        sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
        if sensor_mode and zoom != 0:  # zoom > 0 déjà vérifié
            # Spécifier le mode exact au lieu de --roi
            datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
            # Note: zoom 3 maintenant activé pour IMX585
    elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel, zoom 3 désactivé
        zws = int(igw * zfs[zoom])
        zhs = int(igh * zfs[zoom])
        zxo = ((igw-zws)/2)/igw
        zyo = ((igh-zhs)/2)/igh
        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
    # Supprimé: ancien zoom manuel (zoom == 5)
    if False and zoom == 5:
        zxo = ((igw/2)-(preview_width/2))/igw
        if alt_dis == 2:
            zyo = ((igh/2)-((preview_height * .75)/2))/igh
        else:
            zyo = ((igh/2)-(preview_height/2))/igh
        if igw/igh > 1.5:
            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str((preview_height * .75)/igh)
        else:
            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
    p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
    if show_cmds == 1:
        print(datastr)

    # Plus besoin d'extracteur MJPEG : libcamera-vid écrit directement les fichiers JPEG
    # C'est la méthode simple et éprouvée du programme original

    restart = 0
    time.sleep(0.2)

def Menu():
    global vwidths2,vheights2,Pi_Cam,scientif,mode,v3_hdr,scientific,tinterval,zoom,vwidth,vheight,preview_width,preview_height,ft,fv,focus,fxz,v3_hdr,v3_hdrs,bw,bh,ft,fv,cam1,v3_f_mode,v3_af,button_row,xx,xy,use_native_sensor_mode
    global allsky_mode,allsky_mean_target,allsky_mean_threshold,allsky_video_fps,allsky_max_gain,allsky_apply_stretch,allsky_cleanup_jpegs,allsky_modes
    pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(preview_width,0,bw,preview_height))
    if menu > 0: 
        # set button sizes
        bw = int(preview_width/5.66)
        bh = int(preview_height/10)
        ft = int(preview_width/46)
        fv = int(preview_width/46)
        for d in range(1,9):
            button(0,0,0,4)
            if menu == 1:
                button(0,d,0,4)
            elif menu == 2:
                button(0,d,0,4)
            elif menu == 3 or menu == 4:
                button(0,d,6,4)
            elif menu == 5:
                button(0,d,7,4)
            elif menu == 6:
                button(0,d,8,4)
            elif menu == 7:
                button(0,d,0,4)
            elif menu == 8:
                button(0,d,0,4)
            elif menu == 9:
                button(0,d,0,4)
            elif menu == 10:
                button(0,d,0,4)
        text(0,0,1,0,1,"MAIN MENU ",ft,7)
      
    # ========== MENU 0 - MAIN MENU ==========
    if menu == 0:
        # set button sizes
        bw = int(preview_width/5.66)
        bh = int(preview_height/9)  # 9 buttons instead of 8
        ft = int(preview_width/46)
        fv = int(preview_width/46)
        # Effacer la zone des boutons pour éviter les sliders résiduels
        pygame.draw.rect(windowSurfaceObj,blackColor,Rect(preview_width,0,bw,preview_height))
        button(0,0,4,4)  # STILL
        button(0,1,2,4)  # VIDEO
        button(0,2,3,4)  # TIMELAPSE
        button(0,3,9,4)  # LIVE STACK
        button(0,4,10,4)  # LUCKY STACK button - bleu marine avec texte rouge
        button(0,5,5,4)  # STRETCH
        button(0,6,0,4)  # CAMERA Settings
        button(0,7,0,4)  # OTHER Settings
        button(0,8,0,4)  # EXIT
        text(0,0,8,0,1,"         STILL ",ft,1)
        text(0,1,8,0,1,"         VIDEO",ft,2)
        text(0,2,8,0,1,"       TIMELAPSE",ft-1,4)
        text(0,3,1,0,1,"     LIVE STACK",ft,1)
        text(0,4,3,0,1,"    LUCKY STACK",ft,1)  # Texte rouge (fColor=3)
        text(0,5,1,0,1,"       STRETCH",ft,1)
        text(0,6,1,0,1,"      CAMERA",ft,7)
        text(0,6,1,1,1,"    Settings",ft,7)
        text(0,7,1,0,1,"       OTHER",ft,7)
        text(0,7,1,1,1,"    Settings",ft,7)
        text(0,8,2,0,1,"     EXIT",fv+10,7)
      
    # ========== MENU 1 - CAMERA SETTINGS ==========
    elif menu == 1:
      text(0,1,1,0,1,"STILL",ft,7)
      text(0,1,1,1,1,"Settings",ft,7)
      text(0,2,1,0,1,"VIDEO",ft,7)
      text(0,2,1,1,1,"Settings",ft,7)
      text(0,3,1,0,1,"TIMELAPSE",ft,7)
      text(0,3,1,1,1,"Settings",ft,7)
      text(0,6,1,0,1,"LIVE STACK",ft,7)
      text(0,6,1,1,1,"Settings",ft,7)
      text(0,7,1,0,1,"LUCKY STACK",ft,7)
      text(0,7,1,1,1,"Settings",ft,7)
      text(0,8,1,0,1,"STRETCH",ft,7)
      text(0,8,1,1,1,"Settings",ft,7)
      if zoom == 0:
          button(0,4,0,9)
          text(0,4,5,0,1,"Zoom",ft,7)
          text(0,4,3,1,1,"",fv,7)
          # determine if camera native format
          vw = 0
          x = 0
          while x < len(vwidths2) and vw == 0:
              if vwidth == vwidths2[x]:
                  if vheight == vheights2[x]:
                      vw = 1
              x += 1
      elif zoom < 10:
          button(0,4,1,9)
          text(0,4,2,0,1,"ZOOMED",ft,0)
          # Afficher la résolution ROI au lieu du numéro de zoom
          text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
          draw_Vbar(0,4,greyColor,'zoom',zoom)
      if Pi_Cam == 3 and v3_af == 1:
          if fxz != 1:
              text(0,5,3,1,1,"Spot",fv,7)
          else:
              text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
          if v3_f_mode == 1 :
              button(0,5,1,9)
              draw_Vbar(0,5,dgryColor,'v3_focus',v3_focus-pmin)
              fd = 1/(v3_focus/100)
              text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,7)
          elif v3_f_mode == 0 or v3_f_mode == 2:
              button(0,5,0,9)
              text(0,5,5,0,1,"FOCUS",ft,7)
          text(0,7,2,0,1,"Focus Speed",ft,7)
          text(0,7,3,1,1,v3_f_speeds[v3_f_speed],fv,7)
          text(0,7,2,0,1,"Focus Range",ft,7)
          text(0,7,3,1,1,v3_f_ranges[v3_f_range],fv,7)
          
      else:
          button(0,5,0,9)
          text(0,5,5,0,1,"FOCUS",ft,7)
          text(0,5,3,1,1,"    ",fv,7)
          
      draw_Vbar(0,4,greyColor,'zoom',zoom)
      if Pi_Cam == 3:
          draw_bar(0,6,greyColor,'v3_f_speed',v3_f_speed)
          draw_Vbar(0,7,greyColor,'v3_f_range',v3_f_range)
        
                 
    # ========== MENU 2 - OTHER SETTINGS ==========
    elif menu == 2:
        if cam1 != "1":
            text(0,1,2,0,1,"Switch Camera",ft,7)
            text(0,1,3,1,1,str(camera),fv,7)
        # Ligne 2 - Accès menu METRICS
        button(0,2,0,2)
        text(0,2,5,0,1,"METRICS",ft,7)
        text(0,2,3,1,1,"Settings >",fv,7)

        # Ligne 3 - Histogram
        text(0,3,3,0,1,"Histogram",ft,7)
        text(0,3,3,1,1,histograms[histogram],fv,7)
        draw_bar(0,3,greyColor,'histogram',histogram)

        # Ligne 4 - Hist Area
        text(0,4,2,0,1,"Hist Area",ft,7)
        text(0,4,3,1,1,str(histarea),fv,7)
        draw_Vbar(0,4,greyColor,'histarea',histarea)

        # Ligne 5 - Vert Flip
        text(0,5,5,0,1,"Vert Flip",ft,7)
        text(0,5,3,1,1,str(vflip),fv,7)

        # Ligne 6 - Horiz Flip
        text(0,6,5,0,1,"Horiz Flip",ft,7)
        text(0,6,3,1,1,str(hflip),fv,7)

        # Ligne 7 - RAW Format (remplace STILL -t time)
        text(0,7,5,0,1,"RAW Format",ft,7)
        text(0,7,3,1,1,raw_formats[raw_format],fv,7)
        draw_Vbar(0,7,greyColor,'raw_format',raw_format)

        # Ligne 8 - Sensor Mode
        text(0,8,2,0,1,"Sensor Mode",ft,7)
        if use_native_sensor_mode == 0:
            text(0,8,3,1,1,"Binning",fv,7)
        else:
            text(0,8,3,1,1,"Native",fv,7)
        draw_Vbar(0,8,greyColor,'use_native_sensor_mode',use_native_sensor_mode)

        # Ligne 9 - SAVE CONFIG
        button(0,9,0,4)
        text(0,9,2,0,1,"SAVE CONFIG",fv,7)
      
    # ========== MENU 3 - STILL SETTINGS (Page 1) ==========
    elif menu == 3:
      text(0,1,5,0,1,"Mode",ft,10)
      text(0,1,3,1,1,modes[mode],fv,10)
      if mode == 0:
          text(0,2,5,0,1,"Shutter S",ft,10)
          if shutters[speed] < 0:
              text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
          else:
              text(0,2,3,1,1,str(shutters[speed]),fv,10)
      else:
          text(0,2,5,0,1,"eV",ft,10)
          text(0,2,3,1,1,str(ev),fv,10)
      text(0,3,5,0,1,"Gain    A/D",ft,10)
      if gain > 0:
          text(0,3,5,0,1,"Gain    A/D",ft,10)
          if gain <= mag:
              text(0,3,3,1,1,str(gain) + " :  " + str(gain) + "/1",fv,10)
          else:
              text(0,3,3,1,1,str(gain) + " :  " + str(int(mag)) + "/" + str(((gain/mag)*10)/10)[0:3],fv,10)
      else:
          text(0,3,5,0,1,"Gain",ft,10)
          text(0,3,3,1,1,"Auto",fv,10)
      text(0,4,5,0,1,"Brightness",ft,10)
      text(0,4,3,1,1,str(brightness/100)[0:4],fv,10)
      text(0,5,5,0,1,"Contrast",ft,10)
      text(0,5,3,1,1,str(contrast/100)[0:4],fv,10)
      text(0,6,5,0,1,"AWB",ft,10)
      text(0,6,3,1,1,awbs[awb],fv,10)
      text(0,7,5,0,1,"Blue",ft,10)
      text(0,7,3,1,1,str(blue/10)[0:3],fv,10)
      text(0,8,5,0,1,"Red",ft,10)
      text(0,8,3,1,1,str(red/10)[0:3],fv,10)
      button(0,9,0,9) 
      text(0,9,1,0,1,"Page 2 ",ft,7)
      draw_bar(0,2,lgrnColor,'mode',mode)
      if mode == 0:
            draw_bar(0,2,lgrnColor,'speed',speed)  # Affiche la barre du shutter en mode manuel
      else:
            draw_bar(0,2,lgrnColor,'ev',ev)
      draw_bar(0,3,lgrnColor,'gain',gain)
      draw_bar(0,4,lgrnColor,'brightness',brightness)
      draw_bar(0,5,lgrnColor,'contrast',contrast)
      draw_bar(0,6,lgrnColor,'awb',awb)
      draw_bar(0,7,lgrnColor,'blue',blue)
      draw_bar(0,8,lgrnColor,'red',red)
                
    # ========== MENU 4 - STILL SETTINGS (Page 2) ==========
    elif menu == 4:

        # Ligne 1 - Page 1 (retour)
        button(0,1,0,9)
        text(0,1,1,0,1,"Page 1 ",ft,7)

        # Ligne 2 - Metering
        text(0,2,5,0,1,"Metering",ft,10)
        text(0,2,3,1,1,meters[meter],fv,10)
        draw_bar(0,2,lgrnColor,'meter',meter)

        # Ligne 3 - Quality
        text(0,3,5,0,1,"Quality",ft,10)
        text(0,3,3,1,1,str(quality)[0:3],fv,10)
        draw_bar(0,3,lgrnColor,'quality',quality)

        # Ligne 4 - Saturation
        text(0,4,5,0,1,"Saturation",ft,10)
        text(0,4,3,1,1,str(saturation/10),fv,10)
        draw_bar(0,4,lgrnColor,'saturation',saturation)

        # Ligne 5 - Denoise
        text(0,5,5,0,1,"Denoise",ft,10)
        text(0,5,3,1,1,denoises[denoise],fv,10)
        draw_bar(0,5,lgrnColor,'denoise',denoise)

        # Ligne 6 - Sharpness
        text(0,6,5,0,1,"Sharpness",ft,10)
        text(0,6,3,1,1,str(sharpness/10),fv,10)
        draw_bar(0,6,lgrnColor,'sharpness',sharpness)

        # Ligne 7 - HDR / IR Filter / Scientific (selon caméra)
        if (Pi_Cam == 3 or Pi == 5):
            button(0,7,6,4)
            text(0,7,5,0,1,"HDR",ft,10)
            text(0,7,3,1,1,v3_hdrs[v3_hdr],fv,10)
            draw_bar(0,7,lgrnColor,'v3_hdr',v3_hdr)
        elif Pi_Cam == 9:
            button(0,7,6,4)
            text(0,7,5,0,1,"IR Filter",ft,10)
            if IRF == 0:
                text(0,7,3,1,1,"Off",fv,10)
            else:
                text(0,7,3,1,1,"ON ",fv,10)
        elif Pi_Cam == 4 and scientif == 1:
            button(0,7,6,4)
            text(0,7,5,0,1,"Scientific",ft,10)
            if scientific == 0:
                text(0,7,3,1,1,"Off",fv,10)
            else:
                text(0,7,3,1,1,"ON ",fv,10)

        # Ligne 8 - File Format
        text(0,8,5,0,1,"File Format",ft,10)
        text(0,8,3,1,1,extns[extn],fv,10)
        draw_bar(0,8,lgrnColor,'extn',extn)

        # Ligne 9 - SAVE CONFIG
        button(0,9,6,4)
        text(0,9,2,0,1,"SAVE CONFIG",fv,10)
      
    # ========== MENU 5 - VIDEO SETTINGS ==========
    elif menu == 5:
        text(0,1,5,0,1,"V_Length",ft,11)
        td = timedelta(seconds=vlen)
        text(0,1,3,1,1,str(td),fv,11)
        text(0,2,5,0,1,"V_FPS",ft,11)
        text(0,2,3,1,1,str(fps),fv,11)
        text(0,3,5,0,1,"V_Format",ft,11)
        text(0,4,5,0,1,"V_Codec",ft,11)
        text(0,4,3,1,1,codecs[codec],fv,11)
        text(0,5,5,0,1,"h264 Profile",ft,11)
        text(0,5,3,1,1,str(h264profiles[profile]),fv,11)
        text(0,7,5,0,1,"V_Preview",ft,11)
        text(0,7,3,1,1,"ON ",fv,11)
        draw_Vbar(0,3,lpurColor,'vformat',vformat)
        button(0,8,7,4)
        text(0,8,2,0,1,"SAVE CONFIG",fv,11)
        # determine if camera native format
        vw = 0
        x = 0
        while x < len(vwidths2) and vw == 0:
            if vwidth == vwidths2[x]:
                if vheight == vheights2[x]:
                    vw = 1
            x += 1
        if vw == 0:
            text(0,3,3,1,1,str(vwidth) + "x" + str(vheight),fv,11)
        if vw == 1:
            text(0,3,1,1,1,str(vwidth) + "x" + str(vheight),fv,11)
        draw_Vbar(0,1,lpurColor,'vlen',vlen)
        draw_Vbar(0,2,lpurColor,'fps',fps)
        draw_Vbar(0,3,lpurColor,'vformat',vformat)
        draw_Vbar(0,4,lpurColor,'codec',codec)
        draw_Vbar(0,5,lpurColor,'profile',profile)
      
    # ========== MENU 6 - TIMELAPSE SETTINGS ==========
    elif menu == 6:
        # TIMELAPSE Settings - Multi-pages
        current_page = menu_page.get(6, 1)  # Page par défaut = 1

        if current_page == 1:
            # ========== PAGE 1 - Standard Timelapse Parameters ==========
            # Ligne 1 - Duration
            td = timedelta(seconds=tduration)
            text(0,1,5,0,1,"Duration",ft,12)
            text(0,1,3,1,1,str(td),fv,12)
            draw_Vbar(0,1,lyelColor,'tduration',tduration)

            # Ligne 2 - Interval
            td = timedelta(seconds=tinterval)
            text(0,2,5,0,1,"Interval",ft,12)
            text(0,2,3,1,1,str(td),fv,12)
            draw_Vbar(0,2,lyelColor,'tinterval',tinterval)

            # Ligne 3 - No. of Shots
            text(0,3,5,0,1,"No. of Shots",ft,12)
            if tinterval > 0:
                text(0,3,3,1,1,str(tshots),fv,12)
            else:
                text(0,3,3,1,1," ",fv,12)
            draw_Vbar(0,3,lyelColor,'tshots',tshots)

            # Ligne 8 - SAVE CONFIG
            button(0,8,8,4)
            text(0,8,2,0,1,"SAVE CONFIG",fv,12)

            # Ligne 9 - Navigation to Page 2
            button(0,9,0,9)
            text(0,9,3,0,1,"ALLSKY",ft,12)
            text(0,9,3,1,1,"Page 2 ->",fv,12)

        elif current_page == 2:
            # ========== PAGE 2 - Allsky Parameters ==========
            # Ligne 1 - Navigation retour
            button(0,1,0,1)
            text(0,1,3,0,1,"<- Page 1",ft,10)
            text(0,1,3,1,1,"Standard",fv,10)

            # Ligne 2 - Allsky Mode
            text(0,2,5,0,1,"Allsky Mode",ft,10)
            text(0,2,3,1,1,allsky_modes[allsky_mode],fv,10)
            draw_Vbar(0,2,greyColor,'allsky_mode',allsky_mode)

            # Ligne 3 - Mean Target (only if Auto-Gain)
            text(0,3,5,0,1,"Mean Target",ft,10)
            if allsky_mode == 2:
                text(0,3,3,1,1,str(allsky_mean_target/100.0)[0:4],fv,10)
                draw_Vbar(0,3,greyColor,'allsky_mean_target',allsky_mean_target)
            else:
                text(0,3,3,1,1,"N/A",fv,10)

            # Ligne 4 - Mean Threshold (only if Auto-Gain)
            text(0,4,5,0,1,"Mean Thresh",ft,10)
            if allsky_mode == 2:
                text(0,4,3,1,1,str(allsky_mean_threshold/100.0)[0:4],fv,10)
                draw_Vbar(0,4,greyColor,'allsky_mean_threshold',allsky_mean_threshold)
            else:
                text(0,4,3,1,1,"N/A",fv,10)

            # Ligne 5 - Video FPS (only if Allsky ON or Auto-Gain)
            text(0,5,5,0,1,"Video FPS",ft,10)
            if allsky_mode > 0:
                text(0,5,3,1,1,str(allsky_video_fps),fv,10)
                draw_Vbar(0,5,greyColor,'allsky_video_fps',allsky_video_fps)
            else:
                text(0,5,3,1,1,"N/A",fv,10)

            # Ligne 6 - Max Gain (only if Auto-Gain)
            text(0,6,5,0,1,"Max Gain",ft,10)
            if allsky_mode == 2:
                text(0,6,3,1,1,str(allsky_max_gain),fv,10)
                draw_Vbar(0,6,greyColor,'allsky_max_gain',allsky_max_gain)
            else:
                text(0,6,3,1,1,"N/A",fv,10)

            # Ligne 7 - Apply Stretch (only if Allsky ON or Auto-Gain)
            text(0,7,5,0,1,"Apply Stretch",ft,10)
            if allsky_mode > 0:
                stretch_text = "ON" if allsky_apply_stretch == 1 else "OFF"
                text(0,7,3,1,1,stretch_text,fv,10)
                draw_Vbar(0,7,greyColor,'allsky_apply_stretch',allsky_apply_stretch)
            else:
                text(0,7,3,1,1,"N/A",fv,10)

            # Ligne 8 - Cleanup JPEGs (only if Allsky ON or Auto-Gain)
            text(0,8,5,0,1,"Cleanup JPEGs",ft,10)
            if allsky_mode > 0:
                cleanup_text = "YES" if allsky_cleanup_jpegs == 1 else "NO"
                text(0,8,3,1,1,cleanup_text,fv,10)
                draw_Vbar(0,8,greyColor,'allsky_cleanup_jpegs',allsky_cleanup_jpegs)
            else:
                text(0,8,3,1,1,"N/A",fv,10)

            # Ligne 9 - SAVE CONFIG (on Page 2)
            button(0,9,0,4)
            text(0,9,2,0,1,"SAVE CONFIG",fv,10)

    # ========== MENU 7 - STRETCH SETTINGS ==========
    elif menu == 7:
        # STRETCH Settings - Multi-pages (Phase 2)
        current_page = menu_page.get(7, 1)  # Page par défaut = 1

        if current_page == 1:
            # ========== PAGE 1 - Paramètres Arcsinh + Preset ==========
            # Ligne 1 - Stretch Low %
            text(0,1,5,0,1,"Stretch Low %",ft,7)
            text(0,1,3,1,1,str(stretch_p_low/10)[0:4],fv,7)
            draw_Vbar(0,1,greyColor,'stretch_p_low',stretch_p_low)

            # Ligne 2 - Stretch High %
            text(0,2,5,0,1,"Stretch High %",ft,7)
            text(0,2,3,1,1,str(stretch_p_high/100)[0:6],fv,7)
            draw_Vbar(0,2,greyColor,'stretch_p_high',stretch_p_high)

            # Ligne 3 - Stretch Factor
            text(0,3,5,0,1,"Stretch Factor",ft,7)
            text(0,3,3,1,1,str(stretch_factor/10)[0:4],fv,7)
            draw_Vbar(0,3,greyColor,'stretch_factor',stretch_factor)

            # Ligne 4 - Preset
            text(0,4,5,0,1,"Preset",ft,7)
            text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
            draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)

            # Ligne 8 - SAVE CONFIG
            button(0,8,0,4)
            text(0,8,2,0,1,"SAVE CONFIG",fv,7)

            # Ligne 9 - Navigation
            if stretch_preset == 1:  # GHS sélectionné - montrer accès page 2
                button(0,9,0,9)
                text(0,9,3,0,1,"GHS Params",ft,7)
                text(0,9,3,1,1,"Page 2 ->",ft,7)
            else:
                button(0,9,0,9)
                text(0,9,1,0,1,"CAMERA",ft,7)
                text(0,9,1,1,1,"Settings",ft,7)

        elif current_page == 2:
            # ========== PAGE 2 - Paramètres GHS complets (D, b, SP, LP, HP) ==========
            # Ligne 1 - Navigation retour
            button(0,1,0,1)
            text(0,1,3,0,1,"<- Page 1",ft,7)
            text(0,1,3,1,1,"Arcsinh",ft,7)

            # Ligne 2 - GHS D (Stretch Factor)
            text(0,2,5,0,1,"GHS D (Force)",ft,7)
            text(0,2,3,1,1,str(ghs_D/10.0)[0:5],fv,7)
            draw_Vbar(0,2,greyColor,'ghs_D',ghs_D)

            # Ligne 3 - GHS b (Local Intensity)
            text(0,3,5,0,1,"GHS b (Focus)",ft,7)
            text(0,3,3,1,1,str(ghs_b/10.0)[0:5],fv,7)
            draw_Vbar(0,3,greyColor,'ghs_b',ghs_b)

            # Ligne 4 - GHS SP (Symmetry Point)
            text(0,4,5,0,1,"GHS SP (Sym)",ft,7)
            text(0,4,3,1,1,str(ghs_SP/100.0)[0:5],fv,7)
            draw_Vbar(0,4,greyColor,'ghs_SP',ghs_SP)

            # Ligne 5 - GHS LP (Protect Shadows)
            text(0,5,5,0,1,"GHS LP (Shad)",ft,7)
            text(0,5,3,1,1,str(ghs_LP/100.0)[0:5],fv,7)
            draw_Vbar(0,5,greyColor,'ghs_LP',ghs_LP)

            # Ligne 6 - GHS HP (Protect Highlights)
            text(0,6,5,0,1,"GHS HP (High)",ft,7)
            text(0,6,3,1,1,str(ghs_HP/100.0)[0:5],fv,7)
            draw_Vbar(0,6,greyColor,'ghs_HP',ghs_HP)

            # Ligne 7 - GHS Preset
            text(0,7,5,0,1,"GHS Preset",ft,7)
            text(0,7,3,1,1,ghs_presets[ghs_preset],fv,7)
            draw_Vbar(0,7,greyColor,'ghs_preset',ghs_preset)

            # Ligne 8 - SAVE CONFIG
            button(0,8,0,4)
            text(0,8,2,0,1,"SAVE CONFIG",fv,7)

            # Ligne 9 - Retour menu principal
            button(0,9,0,9)
            text(0,9,1,0,1,"CAMERA",ft,7)
            text(0,9,1,1,1,"Settings",ft,7)

    # ========== MENU 8 - LIVE STACK SETTINGS ==========
    elif menu == 8:
        # LIVE STACK Settings - Multi-pages
        current_page = menu_page.get(8, 1)  # Page par défaut = 1

        if current_page == 1:
            # PAGE 1 - Paramètres existants + Stack Method
            text(0,1,5,0,1,"Preview Refresh",ft,7)
            text(0,1,3,1,1,str(ls_preview_refresh),fv,7)
            draw_Vbar(0,1,greyColor,'ls_preview_refresh',ls_preview_refresh)

            text(0,2,5,0,1,"Alignment Mode",ft,7)
            text(0,2,3,1,1,ls_alignment_modes[ls_alignment_mode],fv,7)
            draw_Vbar(0,2,greyColor,'ls_alignment_mode',ls_alignment_mode)

            text(0,3,5,0,1,"Max FWHM",ft,7)
            if ls_max_fwhm == 0:
                text(0,3,3,1,1,"OFF",fv,7)
            else:
                text(0,3,3,1,1,str(ls_max_fwhm/10)[0:4],fv,7)
            draw_Vbar(0,3,greyColor,'ls_max_fwhm',ls_max_fwhm)

            text(0,4,5,0,1,"Min Sharpness",ft,7)
            if ls_min_sharpness == 0:
                text(0,4,3,1,1,"OFF",fv,7)
            else:
                text(0,4,3,1,1,str(ls_min_sharpness/1000)[0:5],fv,7)
            draw_Vbar(0,4,greyColor,'ls_min_sharpness',ls_min_sharpness)

            text(0,5,5,0,1,"Max Drift",ft,7)
            if ls_max_drift == 0:
                text(0,5,3,1,1,"OFF",fv,7)
            else:
                text(0,5,3,1,1,str(ls_max_drift),fv,7)
            draw_Vbar(0,5,greyColor,'ls_max_drift',ls_max_drift)

            text(0,6,5,0,1,"Min Stars",ft,7)
            if ls_min_stars == 0:
                text(0,6,3,1,1,"OFF",fv,7)
            else:
                text(0,6,3,1,1,str(ls_min_stars),fv,7)
            draw_Vbar(0,6,greyColor,'ls_min_stars',ls_min_stars)

            text(0,7,5,0,1,"Quality Control",ft,7)
            if ls_enable_qc == 0:
                text(0,7,3,1,1,"OFF",fv,7)
            else:
                text(0,7,3,1,1,"ON",fv,7)
            draw_Vbar(0,7,greyColor,'ls_enable_qc',ls_enable_qc)

            # Ligne 8 - Navigation Page 2
            button(0,8,0,9)
            text(0,8,1,0,1,"Page 2",ft,7)
            text(0,8,1,1,1,">>>",ft,7)

        else:  # Page 2
            # PAGE 2 - Stacker + Planetary
            text(0,1,5,0,1,"Stack Method",ft,7)
            text(0,1,3,1,1,stack_methods[ls_stack_method],fv,7)
            draw_Vbar(0,1,greyColor,'ls_stack_method',ls_stack_method)

            text(0,2,5,0,1,"Stack Kappa",ft,7)
            text(0,2,3,1,1,str(ls_stack_kappa/10.0)[0:4],fv,7)
            draw_Vbar(0,2,greyColor,'ls_stack_kappa',ls_stack_kappa)

            text(0,3,5,0,1,"Stack Iterations",ft,7)
            text(0,3,3,1,1,str(ls_stack_iterations),fv,7)
            draw_Vbar(0,3,greyColor,'ls_stack_iterations',ls_stack_iterations)

            text(0,4,5,0,1,"Planetary Enable",ft,7)
            if ls_planetary_enable == 0:
                text(0,4,3,1,1,"OFF",fv,7)
            else:
                text(0,4,3,1,1,"ON",fv,7)
            draw_Vbar(0,4,greyColor,'ls_planetary_enable',ls_planetary_enable)

            text(0,5,5,0,1,"Planetary Mode",ft,7)
            text(0,5,3,1,1,planetary_modes[ls_planetary_mode],fv,7)
            draw_Vbar(0,5,greyColor,'ls_planetary_mode',ls_planetary_mode)

            text(0,6,5,0,1,"Planet Disk Min",ft,7)
            text(0,6,3,1,1,str(ls_planetary_disk_min),fv,7)
            draw_Vbar(0,6,greyColor,'ls_planetary_disk_min',ls_planetary_disk_min)

            text(0,7,5,0,1,"Planet Disk Max",ft,7)
            text(0,7,3,1,1,str(ls_planetary_disk_max),fv,7)
            draw_Vbar(0,7,greyColor,'ls_planetary_disk_max',ls_planetary_disk_max)

            # Ligne 8 - SAVE CONFIG
            button(0,8,6,4)
            text(0,8,2,0,1,"SAVE CONFIG",fv,7)

            # Ligne 9 - Retour Page 1
            button(0,9,0,9)
            text(0,9,1,0,1,"<<< Page 1",ft,7)
            text(0,9,1,1,1,"",ft,7)

    # ========== MENU 9 - LUCKY STACK SETTINGS ==========
    elif menu == 9:
        # LUCKY STACK Settings - Complet
        # (Lucky Enable géré par le bouton LUCKY STACK lui-même)

        # Ligne 1 - Lucky Buffer
        text(0,1,5,0,1,"Buffer Size",ft,7)
        text(0,1,3,1,1,str(ls_lucky_buffer),fv,7)
        draw_Vbar(0,1,greyColor,'ls_lucky_buffer',ls_lucky_buffer)

        # Ligne 2 - Lucky Keep %
        text(0,2,5,0,1,"Keep Best %",ft,7)
        text(0,2,3,1,1,str(ls_lucky_keep)+"%",fv,7)
        draw_Vbar(0,2,greyColor,'ls_lucky_keep',ls_lucky_keep)

        # Ligne 3 - Lucky Score Method
        text(0,3,5,0,1,"Score Method",ft,7)
        text(0,3,3,1,1,lucky_score_methods[ls_lucky_score],fv,7)
        draw_Vbar(0,3,greyColor,'ls_lucky_score',ls_lucky_score)

        # Ligne 4 - Lucky Stack Method
        text(0,4,5,0,1,"Stack Method",ft,7)
        text(0,4,3,1,1,lucky_stack_methods[ls_lucky_stack],fv,7)
        draw_Vbar(0,4,greyColor,'ls_lucky_stack',ls_lucky_stack)

        # Ligne 5 - Lucky Align
        text(0,5,5,0,1,"Align Images",ft,7)
        if ls_lucky_align == 0:
            text(0,5,3,1,1,"OFF",fv,7)
        else:
            text(0,5,3,1,1,"ON",fv,7)
        draw_Vbar(0,5,greyColor,'ls_lucky_align',ls_lucky_align)

        # Ligne 6 - Lucky ROI %
        text(0,6,5,0,1,"ROI %",ft,7)
        text(0,6,3,1,1,str(ls_lucky_roi)+"%",fv,7)
        draw_Vbar(0,6,greyColor,'ls_lucky_roi',ls_lucky_roi)

        # Ligne 7 - (vide pour l'instant)

        # Ligne 8 - SAVE CONFIG
        button(0,8,0,4)
        text(0,8,2,0,1,"SAVE CONFIG",fv,7)

        # Ligne 9 - Retour
        button(0,9,0,9)
        text(0,9,1,0,1,"CAMERA",ft,7)
        text(0,9,1,1,1,"Settings",ft,7)

    # ========== MENU 10 - METRICS SETTINGS ==========
    elif menu == 10:
        # ========== MENU 10 - METRICS Settings ==========

        # Ligne 1 - Focus Method
        text(0,1,5,0,1,"Focus Method",ft,10)
        text(0,1,3,1,1,focus_methods[focus_method],fv,10)
        draw_bar(0,1,lgrnColor,'focus_method',focus_method)

        # Ligne 2 - Star Metric
        text(0,2,5,0,1,"Star Metric",ft,10)
        text(0,2,3,1,1,star_metrics[star_metric],fv,10)
        draw_bar(0,2,lgrnColor,'star_metric',star_metric)

        # Ligne 3 - SNR Display
        text(0,3,5,0,1,"SNR Display",ft,10)
        if snr_display == 0:
            text(0,3,3,1,1,"OFF",fv,10)
        else:
            text(0,3,3,1,1,"ON",fv,10)
        draw_bar(0,3,lgrnColor,'snr_display',snr_display)

        # Ligne 4 - Calc Interval
        text(0,4,5,0,1,"Calc Interval",ft,10)
        text(0,4,3,1,1,str(metrics_interval),fv,10)
        draw_bar(0,4,lgrnColor,'metrics_interval',metrics_interval)

        # Ligne 8 - SAVE CONFIG
        button(0,8,6,4)
        text(0,8,2,0,1,"SAVE CONFIG",fv,10)

        # Ligne 9 - Retour OTHER Settings
        button(0,9,0,9)
        text(0,9,1,0,1,"OTHER",ft,10)
        text(0,9,1,1,1,"Settings",ft,10)

text(0,0,6,2,1,"Please Wait, checking camera",int(fv* 1.7),1)
text(0,0,6,2,1,"Found " + str(cameras[Pi_Cam]),int(fv*1.7),1)

Menu()

time.sleep(1)
pygame.display.update()

# determine max speed for camera
max_speed = 0
while max_shutter > shutters[max_speed]:
    max_speed +=1
if speed > max_speed:
    speed = max_speed
    custom_sspeed = 0  # Réinitialiser car speed a changé
    shutter = shutters[speed]
    if shutter < 0:
        shutter = abs(1/shutter)
    sspeed = int(shutter * 1000000)
    # Les lignes suivantes sont commentées car elles dessinent prématurément
    # un slider avant l'affichage correct du menu, créant un artefact visuel
    # if mode == 0:
    #     if shutters[speed] < 0:
    #         text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
    #     else:
    #         text(0,2,3,1,1,str(shutters[speed]),fv,10)
    # else:
    #     if shutters[speed] < 0:
    #         text(0,2,0,1,1,"1/" + str(abs(shutters[speed])),fv,10)
    #     else:
    #         text(0,2,0,1,1,str(shutters[speed]),fv,10)
# if mode == 0:
#     draw_bar(0,2,lgrnColor,'speed',speed)
pygame.display.update()
time.sleep(.25)

xx = int(preview_width/2)
xy = int(preview_height/2)

fxx = 0
fxy = 0
fxz = 1
fyz = 1
old_histarea = histarea

# start preview
text(0,0,6,2,1,"Please Wait for preview...",int(fv*1.7),1)
preview()

# Créer une image noire par défaut pour éviter NameError au premier passage
if igw/igh > 1.5:
    image = pygame.Surface((preview_width, int(preview_height * 0.75)))
else:
    image = pygame.Surface((preview_width, preview_height))
image.fill((0, 0, 0))  # Remplir en noir

# Variables pour la détection du double-clic sur les sliders
last_click_time = 0
last_click_row = -1
last_click_x = -1
last_click_y = -1
DOUBLE_CLICK_DELAY = 2.0  # 2 secondes pour détecter un double-clic (tenir compte des time.sleep)

# main loop
while True:
    time.sleep(0.01)
    # focus UP button
    if (Pi_Cam == 3 and v3_af == 1) or Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
      if buttonFUP.is_pressed:
        if v3_f_mode != 1:
            focus_mode = 1
            v3_f_mode = 1 # manual focus
            foc_man = 1
            if menu == 0:
                button(0,7,1,9)
                text(0,7,3,1,1,str(v3_f_modes[v3_f_mode]),fv,0)
        v3_focus += 10
        for f in range(0,len(video_limits)-1,3):
          if video_limits[f] == 'v3_focus':
            v3_pmin = video_limits[f+1]
            v3_pmax = video_limits[f+2]
        v3_focus = min(v3_focus,v3_pmax)
        focus = v3_focus
        if Pi_Cam == 3 and menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus * 4)
            fd = 1/(v3_focus/100)
            text(0,7,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
        elif menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus)
            text(0,7,3,0,1,'<<< ' + str(v3_focus) + ' >>>',fv,0)

        # ===== MODE PICAMERA2 : Changement focus en temps réel =====
        if use_picamera2 and picam2 is not None:
            try:
                picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": v3_focus / 100 if Pi_Cam == 3 else focus / 100
                })
            except:
                pass
        # ===== MODE RPICAM-VID : Redémarrer le subprocess =====
        else:
            kill_preview_process()
            text(0,0,6,2,1,"Waiting for preview ...",int(fv*1.7),1)
            preview()

    # focus DOWN button
    if (Pi_Cam == 3 and v3_af == 1) or Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
      if buttonFDN.is_pressed:
        if v3_f_mode != 1:
            focus_mode = 1
            v3_f_mode = 1 # manual focus
            foc_man = 1
            if menu == 0:
                button(0,7,1,9)
                text(0,7,3,1,1,str(v3_f_modes[v3_f_mode]),fv,0)
        v3_focus -= 10
        for f in range(0,len(video_limits)-1,3):
          if video_limits[f] == 'v3_focus':
            v3_pmin = video_limits[f+1]
            v3_pmax = video_limits[f+2]
        v3_focus = max(v3_focus,v3_pmin)
        focus = v3_focus
        if Pi_Cam == 3 and menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus * 4)
            fd = 1/(v3_focus/100)
            text(0,7,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
        elif menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus)
            text(0,7,3,0,1,'<<< ' + str(v3_focus) + ' >>>',fv,0)

        # ===== MODE PICAMERA2 : Changement focus en temps réel =====
        if use_picamera2 and picam2 is not None:
            try:
                picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": v3_focus / 100 if Pi_Cam == 3 else focus / 100
                })
            except:
                pass
        # ===== MODE RPICAM-VID : Redémarrer le subprocess =====
        else:
            kill_preview_process()
            text(0,0,6,2,1,"Waiting for preview ...",int(fv*1.7),1)
            preview()    

       
    # ===== CAPTURE AVEC PICAMERA2 =====
    if use_picamera2 and picam2 is not None:
        # DEBUG: afficher une fois qu'on rentre dans cette section (si show_cmds activé)
        if show_cmds == 1 and not hasattr(pygame, '_picam2_section_debug'):
            print("DEBUG: Entering Picamera2 capture section")
            pygame._picam2_section_debug = True
        try:
            # Récupérer la dernière frame depuis le thread de capture asynchrone
            # Au lieu de bloquer avec capture_array(), on récupère la frame si disponible
            frame_from_thread, metadata_from_thread = None, None
            if capture_thread is not None:
                frame_from_thread, metadata_from_thread = capture_thread.get_latest_frame()

            # Si une frame est disponible, la traiter et l'afficher
            # Sinon, on saute juste le traitement (l'interface reste réactive)
            if frame_from_thread is not None:
                # Si (livestack/luckystack actif OU mode stretch preview) ET format RAW Bayer sélectionné → débayériser en uint16
                # Le mode stretch_mode == 1 permet de prévisualiser les images RAW avant de lancer le stacking
                if ((livestack_active or luckystack_active) or stretch_mode == 1) and raw_format >= 2:
                    # Utiliser la frame RAW capturée par le thread
                    raw_array = frame_from_thread

                    # *** Validation: Vérifier que la frame est bien du format RAW (2D) ***
                    skip_raw_processing = False
                    if len(raw_array.shape) != 2:
                        if not hasattr(pygame, '_raw_format_mismatch_warning'):
                            print(f"\n⚠️  [RAW VALIDATION] Frame ISP reçue au lieu de RAW - ignorée")
                            print(f"  Array reçu: shape={raw_array.shape}, dtype={raw_array.dtype}")
                            print(f"  → Forçage du capture_thread en mode RAW...")
                            pygame._raw_format_mismatch_warning = True

                        # Forcer le capture_thread en mode RAW pour les prochaines captures
                        if capture_thread is not None:
                            capture_thread.set_capture_params({'type': 'raw'})

                        # *** SKIP: Ne pas traiter cette frame ISP, attendre une vraie frame RAW ***
                        skip_raw_processing = True

                    # Validation: Vérifier que la résolution capturée correspond à celle attendue
                    if not skip_raw_processing:
                        expected_height = raw_stream_size[1]
                        actual_height = raw_array.shape[0]

                        if actual_height != expected_height and len(raw_array.shape) == 2:
                            if not hasattr(pygame, '_raw_resolution_mismatch_warning'):
                                print(f"\n⚠️  [RAW VALIDATION] Incompatibilité de résolution détectée!")
                                print(f"  Attendu: {raw_stream_size}")
                                print(f"  Capturé: {raw_array.shape}")
                                print(f"  → Cette résolution n'est pas compatible avec le mode RAW")
                                print(f"  → Utilisez XRGB8888 ISP ou changez de résolution")
                                pygame._raw_resolution_mismatch_warning = True

                    # Débayériser en RGB uint16 [0-65535] (préserve dynamique 12/16-bit)
                    if skip_raw_processing:
                        # Frame ISP ignorée - on attend une vraie frame RAW
                        continue  # Sauter cette itération et attendre une vraie frame RAW

                    # *** Passer les gains AWB pour corriger la balance des blancs en RAW ***
                    # Les gains sont appliqués correctement : rouge sur rouge, bleu sur bleu
                    array_uint16 = debayer_raw_array(raw_array, raw_formats[raw_format],
                                                      red_gain=(red/10),    # Curseur rouge → canal rouge
                                                      blue_gain=(blue/10),  # Curseur bleu → canal bleu
                                                      fix_bad_pixels=bool(fix_bad_pixels) and livestack_active,  # SEULEMENT LiveStack, PAS LuckyStack
                                                      sigma_threshold=fix_bad_pixels_sigma/10.0,
                                                      min_adu_threshold=fix_bad_pixels_min_adu/10.0)

                    # *** DIAGNOSTIC: Vérifier que le débayerisation a retourné 3D ***
                    if len(array_uint16.shape) != 3:
                        print(f"[ERROR] debayer_raw_array a retourné un array {len(array_uint16.shape)}D!")
                        print(f"  Input shape: {raw_array.shape}, dtype: {raw_array.dtype}")
                        print(f"  Output shape: {array_uint16.shape}, dtype: {array_uint16.dtype}")
                        # Forcer conversion en 3D pour éviter crash
                        if len(array_uint16.shape) == 2:
                            print(f"  → Conversion forcée en RGB grayscale")
                            array_uint16 = np.stack([array_uint16, array_uint16, array_uint16], axis=-1)

                    # CORRECTION: Garder la pleine dynamique 16-bit [0-65535] pour RAW12/16
                    # Ne plus compresser à [0-255] car libastrostack gère correctement les données haute résolution
                    # Convertir juste en float32 pour compatibilité avec libastrostack
                    array = array_uint16.astype(np.float32)  # Garder [0-65535]

                    # Appliquer boost de contraste SEULEMENT si ISP libastrostack est désactivé
                    # Si ISP activé, on passe les données RAW brutes à libastrostack qui fera le traitement
                    if isp_enable == 0:
                        # CORRECTION: Boost calibré pour plage 16-bit [0-65535]
                        # Appliquer boost de contraste au lieu du gamma (préserve mieux les noirs)
                        # Méthode : (value - midpoint) × factor + midpoint + brightness_offset
                        midpoint = 32768.0  # Milieu de la plage 16-bit
                        contrast_factor = 1.15  # Léger boost de contraste
                        brightness_offset = 2048  # Légère augmentation de luminosité (équivalent de +8 en 8-bit)
                        array = np.clip((array - midpoint) * contrast_factor + midpoint + brightness_offset, 0, 65535.0)
                    if show_cmds == 1 and not hasattr(pygame, '_stacking_mode_shown'):
                        metadata = metadata_from_thread if metadata_from_thread else {}
                        print(f"\n[STACKING] Mode actif, capture depuis stream RAW")
                        print(f"  Format RAW: {raw_formats[raw_format]}")
                        print(f"  Array débayérisé uint16: shape={array_uint16.shape}, range=[{array_uint16.min()}, {array_uint16.max()}]")
                        print(f"  Array après conversion float32: shape={array.shape}, dtype={array.dtype}, range=[{array.min():.2f}, {array.max():.2f}]")
                        print(f"  Dynamique préservée: {len(np.unique(array_uint16))} niveaux distincts")
                        if isp_enable == 1:
                            print(f"  Traitement ISP software: DÉSACTIVÉ (libastrostack ISP actif)")
                        else:
                            print(f"  Traitement ISP software: ACTIVÉ (Contraste ×1.15 + Luminosité +8)")
                        print(f"  Métadonnées caméra:")
                        print(f"    ExposureTime: {metadata.get('ExposureTime', 'N/A')}µs")
                        print(f"    AnalogueGain: {metadata.get('AnalogueGain', 'N/A')}")
                        print(f"    ColourGains: {metadata.get('ColourGains', 'N/A')}")
                        print(f"    Sensor Mode: {'NATIVE' if use_native_sensor_mode == 1 else 'BINNING'}")
                        pygame._stacking_mode_shown = True
                else:
                    # Utiliser la frame MAIN capturée par le thread
                    array = frame_from_thread

                    # Si XRGB8888, extraire seulement BGR (ignorer le canal X)
                    if raw_format == 1 and len(array.shape) == 3 and array.shape[2] == 4:
                        # DEBUG: Analyser les canaux AVANT extraction (si show_cmds activé)
                        if show_cmds == 1 and not hasattr(pygame, '_xrgb_debug_shown'):
                            print(f"\n[DEBUG XRGB8888] Array AVANT extraction:")
                            print(f"  Shape: {array.shape}, dtype: {array.dtype}")
                            for i in range(4):
                                channel_mean = array[:,:,i].mean()
                                channel_min = array[:,:,i].min()
                                channel_max = array[:,:,i].max()
                                print(f"  Canal {i}: mean={channel_mean:.1f}, min={channel_min}, max={channel_max}")
                            pygame._xrgb_debug_shown = True

                        # XRGB8888 format libcamera : B, G, R, X (BGR + padding)
                        # PyGame et OpenCV utilisent BGR, donc on garde l'ordre natif
                        # Ne PAS inverser les canaux !
                        array = array[:, :, 0:3].copy()  # Prendre les 3 premiers canaux BGR tel quel

                        if show_cmds == 1 and (livestack_active or luckystack_active) and not hasattr(pygame, '_stacking_xrgb_shown'):
                            metadata = metadata_from_thread if metadata_from_thread else {}
                            print(f"\n[STACKING] Mode XRGB8888 ISP, capture depuis stream MAIN")
                            print(f"  Array BGR uint8: shape={array.shape}, dtype={array.dtype}, range=[{array.min()}, {array.max()}]")
                            print(f"  Canaux (ordre BGR): B={array[:,:,0].mean():.1f}, G={array[:,:,1].mean():.1f}, R={array[:,:,2].mean():.1f}")
                            print(f"  ISP natif RPi5: Debayer + Denoise + AWB + Gamma")
                            print(f"  Métadonnées caméra:")
                            print(f"    ExposureTime: {metadata.get('ExposureTime', 'N/A')}µs")
                            print(f"    AnalogueGain: {metadata.get('AnalogueGain', 'N/A')}")
                            print(f"    ColourGains: {metadata.get('ColourGains', 'N/A')}")
                            print(f"    Sensor Mode: {'NATIVE' if use_native_sensor_mode == 1 else 'BINNING'}")
                            pygame._stacking_xrgb_shown = True
                    elif show_cmds == 1 and (livestack_active or luckystack_active) and not hasattr(pygame, '_stacking_yuv_shown'):
                        metadata = metadata_from_thread if metadata_from_thread else {}
                        print(f"\n[STACKING] Mode YUV420, capture depuis stream MAIN")
                        print(f"  Array uint8: shape={array.shape}, dtype={array.dtype}, range=[{array.min()}, {array.max()}]")
                        print(f"  Métadonnées caméra:")
                        print(f"    ExposureTime: {metadata.get('ExposureTime', 'N/A')}µs")
                        print(f"    AnalogueGain: {metadata.get('AnalogueGain', 'N/A')}")
                        print(f"    ColourGains: {metadata.get('ColourGains', 'N/A')}")
                        print(f"    Sensor Mode: {'NATIVE' if use_native_sensor_mode == 1 else 'BINNING'}")
                        pygame._stacking_yuv_shown = True

                # DEBUG: afficher une fois au démarrage (si show_cmds activé)
                if show_cmds == 1 and not hasattr(pygame, '_picam2_debug_done'):
                    print(f"DEBUG: Array shape: {array.shape}, dtype: {array.dtype}")
                    print(f"DEBUG: Array min: {array.min()}, max: {array.max()}, mean: {array.mean():.1f}")
                    # Analyser les canaux RGB séparément (si image non-noire)
                    if len(array.shape) == 3 and array.max() > 0:
                        r_mean, g_mean, b_mean = array[:,:,0].mean(), array[:,:,1].mean(), array[:,:,2].mean()
                        r_max, g_max, b_max = array[:,:,0].max(), array[:,:,1].max(), array[:,:,2].max()
                        print(f"DEBUG: RGB channels - R: mean={r_mean:.1f} max={r_max}, G: mean={g_mean:.1f} max={g_max}, B: mean={b_mean:.1f} max={b_max}")
                        # Vérifier si les canaux sont similaires (image quasi-mono)
                        channel_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
                        if channel_diff < 5:
                            print(f"[WARNING] Canaux RGB très similaires (diff={channel_diff:.1f}) - image quasi-monochrome!")
                            print(f"           Vérifiez: saturation caméra, AWB, gains couleur (red={red}, blue={blue})")
                    # Sauvegarder une frame de test pour vérifier
                    import cv2
                    cv2.imwrite('/home/admin/debug_picamera2_frame.jpg', cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                    print("DEBUG: Frame saved to /home/admin/debug_picamera2_frame.jpg")
                    pygame._picam2_debug_done = True

                # CORRECTION AUTOMATIQUE DÉSACTIVÉE: L'IMX585 fonctionne correctement en 0-255
                # Si nécessaire, activer uniquement pour le vrai bug 0-127 avec seuil < 128
                # if array.max() < 128 and array.max() > 0:
                #     if not hasattr(pygame, '_picam2_range_warning'):
                #         print(f"[WARNING] Camera output range 0-{array.max()} detected, expanding to 0-255")
                #         pygame._picam2_range_warning = True
                #     array = (array.astype(np.float32) * 2.0).clip(0, 255).astype(np.uint8)

                # ===== LIVE STACK PROCESSING =====
                livestack_display_done = False
                if livestack_active and livestack is not None:
                    # Traiter la frame avec LiveStack (module avancé)
                    livestack.process_frame(array)

                    # Récupérer le master stack pour affichage
                    try:
                        # Récupérer le résultat stacké
                        stacked_array = livestack.get_preview_for_display()

                        if stacked_array is not None:
                            # === MODE RAW: Appliquer ISP + stretch EXTERNE ===
                            # Utilise apply_isp_to_preview() + astro_stretch() pour garantir
                            # que le résultat est IDENTIQUE au preview stretch
                            if raw_format >= 2:
                                # Le stacked_array est brut (pas d'ISP/stretch de libastrostack)
                                # Appliquer le même traitement que le preview
                                stacked_array = apply_isp_to_preview(stacked_array)
                                if stretch_preset != 0:
                                    stacked_array = astro_stretch(stacked_array)
                                # Convertir en uint8 si nécessaire
                                if stacked_array.dtype != np.uint8:
                                    stacked_array = np.clip(stacked_array, 0, 255).astype(np.uint8)

                                # Afficher les infos une fois
                                if not hasattr(pygame, '_livestack_display_info_shown'):
                                    print(f"\n[LIVESTACK DISPLAY RAW] Traitement externe:")
                                    print(f"  ✓ apply_isp_to_preview() (WB, gamma, contrast, etc.)")
                                    print(f"  ✓ astro_stretch() (GHS/Arcsinh si activé)")
                                    print(f"  → IDENTIQUE au preview stretch")
                                    pygame._livestack_display_info_shown = True
                            else:
                                # MODE RGB/YUV: libastrostack a déjà appliqué le stretch
                                if not hasattr(pygame, '_livestack_display_info_shown'):
                                    print(f"\n[LIVESTACK DISPLAY RGB/YUV] Pipeline libastrostack:")
                                    print(f"  ✓ Stretch configuré (méthode: {livestack.session.config.png_stretch_method if livestack.session else 'N/A'})")
                                    print(f"  ✓ Conversion uint8 pour pygame")
                                    pygame._livestack_display_info_shown = True

                            # Convertir en surface pygame
                            if len(stacked_array.shape) == 3:
                                # Transposer (H,W,C) → (W,H,C) pour pygame
                                transposed = np.swapaxes(stacked_array, 0, 1)
                                # IMPORTANT: pygame attend BGR, échanger R/B pour tous les modes
                                # Ceci aligne le livestack avec le preview stretch (ligne 8593)
                                image = pygame.surfarray.make_surface(transposed[:,:,[2,1,0]])
                            else:
                                # MONO
                                image = pygame.surfarray.make_surface(stacked_array.T)

                            # Redimensionner en fullscreen si stretch activé
                            if stretch_mode == 1:
                                display_modes = pygame.display.list_modes()
                                if display_modes and display_modes != -1:
                                    max_width, max_height = display_modes[0]
                                else:
                                    screen_info = pygame.display.Info()
                                    max_width, max_height = screen_info.current_w, screen_info.current_h
                                image = pygame.transform.scale(image, (max_width, max_height))
                            elif image.get_width() != preview_width or image.get_height() != preview_height:
                                image = pygame.transform.scale(image, (preview_width, preview_height))

                            # Afficher
                            windowSurfaceObj.blit(image, (0, 0))
                            pygame.display.update()

                            # Afficher les statistiques Live Stack
                            stats = livestack.get_stats()
                            stats_text = f"LiveStack: {stats['accepted_frames']}/{stats['total_frames']} | Rejected: {stats['rejected_frames']}"
                            text(0, 0, 2, 2, 1, stats_text, ft, 1)  # top=2 pour position absolue à gauche

                            # Sauvegarder PNG intermédiaire (si activé)
                            accepted = stats['accepted_frames']
                            if ls_save_progress == 1 and accepted > 0:
                                if not hasattr(pygame, '_livestack_last_saved'):
                                    pygame._livestack_last_saved = 0
                                if accepted > pygame._livestack_last_saved:
                                    try:
                                        # Mode RAW: utiliser traitement externe pour cohérence preview/stack
                                        if raw_format >= 2:
                                            save_with_external_processing(livestack, filename=f"livestack_progress_{accepted:04d}", raw_format_name=raw_formats[raw_format])
                                        else:
                                            livestack.save(filename=f"livestack_progress_{accepted:04d}")
                                        pygame._livestack_last_saved = accepted
                                        if show_cmds == 1:
                                            print(f"[LIVESTACK] PNG intermédiaire sauvegardé: {accepted} frames")
                                    except Exception as e:
                                        if show_cmds == 1:
                                            print(f"[LIVESTACK] Erreur save PNG: {e}")

                            # Marquer que l'affichage est fait
                            livestack_display_done = True
                        else:
                            # Pas encore de stack, afficher le flux vidéo en attendant
                            livestack_display_done = False
                            # Afficher message d'attente avec stats détaillées
                            stats = livestack.get_stats()
                            wait_text = f"LiveStack: {stats['accepted_frames']}/{stats['total_frames']} acceptées | Rejetées: {stats['rejected_frames']}"
                            text(0, 0, 2, 2, 1, wait_text, ft, 1)  # top=2 pour position absolue à gauche

                            # Afficher aussi un warning si trop de rejets (QC trop strict)
                            if stats['total_frames'] > 20 and stats['accepted_frames'] == 0:
                                warning_text = "⚠ Toutes les frames rejetées! Désactiver QC ou réduire min_stars"
                                text(0, 1, 2, 2, 1, warning_text, ft, 1)  # top=2 pour position absolue à gauche

                    except Exception as e:
                        if show_cmds == 1:
                            print(f"[DEBUG] Erreur affichage LiveStack: {e}")
                            import traceback
                            traceback.print_exc()
                        livestack_display_done = False

                # ===== LUCKY STACK PROCESSING =====
                elif luckystack_active and luckystack is not None:
                    # DEBUG: Vérifier la résolution de l'array reçu
                    if not hasattr(pygame, '_lucky_resolution_check'):
                        print(f"\n[LUCKYSTACK DEBUG] Résolution de capture:")
                        print(f"  Array shape: {array.shape}")
                        print(f"  Array dtype: {array.dtype}")
                        print(f"  Array range: [{array.min():.1f}, {array.max():.1f}]")
                        print(f"  Zoom actuel: {zoom}")
                        print(f"  Résolution configurée (capture_size): {capture_size}")
                        print(f"  Résolution RAW stream (raw_stream_size): {raw_stream_size}")
                        print(f"  Format RAW: {raw_formats[raw_format]}")
                        pygame._lucky_resolution_check = True

                    # Traiter la frame avec Lucky Stack (nécessaire pour le fonctionnement interne)
                    luckystack.process_frame(array)

                    # Récupérer les statistiques
                    stats = luckystack.get_stats()
                    buffer_fill = stats.get('lucky_buffer_fill', 0)
                    buffer_size = stats.get('lucky_buffer_size', 0)
                    stacks_done = stats.get('lucky_stacks_done', 0)
                    total_frames = stats.get('total_frames', 0)
                    avg_score = stats.get('lucky_avg_score', 0)
                    stats_text1 = f"Lucky: Buffer {buffer_fill}/{buffer_size} | Frames: {total_frames}"
                    stats_text2 = f"Stacks: {stacks_done} | Score moy: {avg_score:.1f}"

                    # Initialiser le compteur de stacks affichés si nécessaire
                    if not hasattr(pygame, '_lucky_last_displayed'):
                        pygame._lucky_last_displayed = 0

                    # CORRECTION: Toujours essayer de récupérer et afficher le preview Lucky
                    # (que ce soit un nouveau stack ou le dernier stack existant)
                    try:
                        # Récupérer le résultat stacké
                        stacked_array = luckystack.get_preview_for_display()

                        if stacked_array is not None:
                            # === MODE RAW: Appliquer ISP + stretch EXTERNE ===
                            # Même traitement que livestack et que le preview stretch
                            if raw_format >= 2:
                                stacked_array = apply_isp_to_preview(stacked_array)
                                if stretch_preset != 0:
                                    stacked_array = astro_stretch(stacked_array)
                                if stacked_array.dtype != np.uint8:
                                    stacked_array = np.clip(stacked_array, 0, 255).astype(np.uint8)

                            # Détecter un nouveau stack seulement pour les sauvegardes PNG
                            is_new_stack = (stacks_done > pygame._lucky_last_displayed)

                            if is_new_stack:
                                # Afficher les infos de traitement une fois au premier stack
                                if not hasattr(pygame, '_luckystack_display_info_shown'):
                                    if raw_format >= 2:
                                        print(f"\n[LUCKYSTACK DISPLAY RAW] Traitement externe:")
                                        print(f"  ✓ apply_isp_to_preview() (WB, gamma, contrast, etc.)")
                                        print(f"  ✓ astro_stretch() (GHS/Arcsinh si activé)")
                                        print(f"  → IDENTIQUE au preview stretch")
                                    else:
                                        print(f"\n[LUCKYSTACK DISPLAY RGB/YUV] Pipeline libastrostack:")
                                        print(f"  ✓ Stretch configuré (méthode: {luckystack.session.config.png_stretch_method if luckystack.session else 'N/A'})")
                                    pygame._luckystack_display_info_shown = True

                                # Mettre à jour le compteur
                                pygame._lucky_last_displayed = stacks_done

                            # Convertir en surface pygame
                            if len(stacked_array.shape) == 3:
                                # Transposer (H,W,C) → (W,H,C) pour pygame
                                transposed = np.swapaxes(stacked_array, 0, 1)
                                # IMPORTANT: pygame attend BGR, échanger R/B pour tous les modes
                                # Ceci aligne le luckystack avec le preview stretch
                                image = pygame.surfarray.make_surface(transposed[:,:,[2,1,0]])
                            else:
                                # MONO
                                image = pygame.surfarray.make_surface(stacked_array.T)

                            # Redimensionner pour l'affichage
                            if stretch_mode == 1:
                                display_modes = pygame.display.list_modes()
                                if display_modes and display_modes != -1:
                                    max_width, max_height = display_modes[0]
                                else:
                                    screen_info = pygame.display.Info()
                                    max_width, max_height = screen_info.current_w, screen_info.current_h
                                image = pygame.transform.scale(image, (max_width, max_height))
                            elif image.get_width() != preview_width or image.get_height() != preview_height:
                                image = pygame.transform.scale(image, (preview_width, preview_height))

                            # Afficher
                            windowSurfaceObj.blit(image, (0, 0))
                            pygame.display.update()

                            # Marquer que l'affichage Lucky est fait
                            livestack_display_done = True

                            # Sauvegarder PNG intermédiaire tous les 2 stacks (si activé ET nouveau stack)
                            if is_new_stack and ls_lucky_save_progress == 1 and stacks_done > 0 and stacks_done % 2 == 0:
                                if not hasattr(pygame, '_lucky_last_saved'):
                                    pygame._lucky_last_saved = 0
                                if stacks_done > pygame._lucky_last_saved:
                                    try:
                                        # Mode RAW: utiliser traitement externe pour cohérence preview/stack
                                        if raw_format >= 2:
                                            save_with_external_processing(luckystack, filename=f"lucky_progress_{stacks_done:04d}", raw_format_name=raw_formats[raw_format])
                                        else:
                                            luckystack.save(filename=f"lucky_progress_{stacks_done:04d}")
                                        pygame._lucky_last_saved = stacks_done
                                        if show_cmds == 1:
                                            print(f"[LUCKYSTACK] PNG intermédiaire sauvegardé: stack #{stacks_done}")
                                    except Exception as e:
                                        if show_cmds == 1:
                                            print(f"[LUCKYSTACK] Erreur save PNG: {e}")

                    except Exception as e:
                        if show_cmds == 1:
                            print(f"[DEBUG] Erreur affichage Lucky: {e}")
                            import traceback
                            traceback.print_exc()

                    # Si aucun stack n'est disponible, le flux vidéo sera affiché par le code normal
                    # (livestack_display_done reste False)

                    # Afficher les stats Lucky par-dessus avec fond noir stable (évite le clignotement)
                    # Le paramètre top=2 dessine à la position absolue, bkgnd_Color=1 dessine fond noir
                    text(0, 0, 2, 2, 1, stats_text1, ft, 1)  # top=2 position absolue, bkgnd_Color=1 fond noir
                    text(0, 1, 2, 2, 1, stats_text2, ft, 1)  # upd=1 force update

                # Traitement normal seulement si LiveStack/LuckyStack n'a pas déjà affiché
                if not livestack_display_done:
                    # *** VALIDATION: Vérifier que array est 3D avant affichage ***
                    if len(array.shape) == 2:
                        # Array 2D (RAW Bayer non débayérisé) - ne devrait pas arriver ici
                        # Mais si c'est le cas, afficher en niveaux de gris comme fallback
                        if not hasattr(pygame, '_raw_2d_warning'):
                            print(f"[WARNING] Array 2D détecté dans fallback display: shape={array.shape}")
                            print(f"  → Conversion en niveaux de gris pour affichage")
                            pygame._raw_2d_warning = True
                        # Convertir en grayscale displayable (normaliser si nécessaire)
                        if array.max() > 255:
                            array = (array.astype(np.float32) / array.max() * 255).astype(np.uint8)
                        else:
                            array = array.astype(np.uint8)
                        # Dupliquer en 3 canaux pour RGB
                        array = np.stack([array, array, array], axis=-1)

                    # Appliquer les réglages ISP en temps réel si mode RAW (preview reflète le stacking)
                    # Toujours appliquer l'ISP quand on est en mode stretch RAW, pas seulement avec le panneau ouvert
                    if stretch_mode == 1 and raw_format >= 2:
                        array = apply_isp_to_preview(array)

                    # Appliquer le stretch astro si le mode est activé ET que le preset n'est pas OFF
                    if stretch_mode == 1 and stretch_preset != 0:
                        array = astro_stretch(array)

                    # Convertir float32 → uint8 pour pygame (si nécessaire)
                    # Le stretch GHS préserve maintenant float32 pour garder la dynamique RAW12/16
                    if array.dtype == np.float32:
                        array = np.clip(array, 0, 255).astype(np.uint8)

                    # Convertir numpy array → pygame surface
                    # Picamera2 retourne (height, width, 3) en RGB
                    # pygame.surfarray.make_surface attend (width, height, 3)
                    # MAIS pygame interprète les couleurs comme (R,G,B) dans l'ordre (0,1,2)
                    # On doit transposer width/height ET échanger R et B pour pygame
                    image = pygame.surfarray.make_surface(np.swapaxes(array, 0, 1)[:,:,[2,1,0]])

                    # Scaling si nécessaire
                    if stretch_mode == 1:
                        # En mode stretch, afficher en VRAI plein écran (cache la barre de tâches)
                        # Obtenir la résolution maximale de l'écran
                        display_modes = pygame.display.list_modes()
                        if display_modes and display_modes != -1:
                            # Prendre la résolution la plus grande (la première de la liste)
                            max_width, max_height = display_modes[0]
                        else:
                            # Fallback si list_modes ne fonctionne pas
                            screen_info = pygame.display.Info()
                            max_width, max_height = screen_info.current_w, screen_info.current_h
                        image = pygame.transform.scale(image, (max_width, max_height))
                    elif image.get_width() != preview_width or image.get_height() != preview_height:
                        if igw/igh > 1.5:
                            image = pygame.transform.scale(image, (preview_width, int(preview_height * 0.75)))
                        else:
                            image = pygame.transform.scale(image, (preview_width, preview_height))

                    # Affichage
                    windowSurfaceObj.blit(image, (0, 0))

                    # Afficher les contrôles de réglage stretch si en mode stretch
                    # Ne PAS afficher pendant le stacking (livestack ou luckystack)
                    if stretch_mode == 1 and not livestack_active and not luckystack_active:
                        # Icône unique: ISP en mode RAW, ADJ en mode RGB/YUV
                        is_raw = (raw_format >= 2)
                        draw_stretch_hand_icon(max_width, max_height, stretch_adjust_mode == 1, is_raw_mode=is_raw)
                        # Bouton STACK pour lancer le live stacking (pas en mode lucky)
                        draw_livestack_button(max_width, max_height, is_raw_mode=is_raw)

                        # Afficher le panneau approprié selon le mode
                        if stretch_adjust_mode == 1:
                            if is_raw:
                                # Mode RAW: panneau ISP complet (ISP + stretch)
                                _raw_slider_rects = draw_raw_controls(max_width, max_height, array)
                            else:
                                # Mode RGB/YUV: panneau stretch uniquement
                                _stretch_slider_rects = draw_stretch_controls(max_width, max_height, array)

        except Exception as e:
            # En cas d'erreur, afficher dans la console
            import traceback
            if show_cmds == 1:
                print(f"Erreur capture Picamera2: {e}")
                traceback.print_exc()
            pass

    # ===== CAPTURE AVEC RPICAM-VID (méthode originale) =====
    else:
        # MÉTHODE OPTIMISÉE POUR FLUIDITÉ
        # Charger l'image la plus récente (pics[0]) pour réduire le délai
        # Garder un petit buffer (3 images) pour éviter suppressions constantes
        pics = glob.glob('/run/shm/*.jpg')
        if len(pics) > 0:
            # Tri alphabétique descendant (test0003, test0002, test0001, test0000)
            pics.sort(reverse=True)
            try:
                # Charger la DERNIÈRE image (pics[0]) pour minimiser le délai
                # Avec --segment 1, les fichiers sont écrits très rapidement
                image = pygame.image.load(pics[0])

                # Garder seulement les 3 images les plus récentes (réduit les I/O)
                # Au lieu de tout supprimer à chaque fois
                if len(pics) > 3:
                    for tt in range(3, len(pics)):
                        try:
                            os.remove(pics[tt])
                        except OSError:
                            pass  # Ignorer si déjà supprimé
            except (pygame.error, OSError):
                # Si pics[0] échoue (en cours d'écriture), essayer pics[1]
                try:
                    if len(pics) > 1:
                        image = pygame.image.load(pics[1])
                except (pygame.error, OSError):
                    pass  # Garder l'ancienne image affichée

            # Convertir pygame surface → numpy array pour le stretch et/ou l'histogramme
            img_array = None
            if stretch_mode == 1:
                img_array = pygame.surfarray.array3d(image)
                # Transposer de (width, height, channels) à (height, width, channels)
                img_array = np.transpose(img_array, (1, 0, 2))

                # Appliquer les réglages ISP en temps réel si mode RAW (preview reflète le stacking)
                # Toujours appliquer l'ISP quand on est en mode stretch RAW
                if raw_format >= 2:
                    img_array = apply_isp_to_preview(img_array)

                # Appliquer le stretch astro si le preset n'est pas OFF
                if stretch_preset != 0:
                    # Appliquer le stretch
                    img_array = astro_stretch(img_array)
                    # Reconvertir en pygame surface
                    image = pygame.surfarray.make_surface(np.swapaxes(img_array, 0, 1))

            # Scaling et affichage
            # En mode zoom, l'image arrive déjà à la bonne taille via ROI, ne pas rescaler
            if stretch_mode == 1:
                # En mode stretch, afficher en VRAI plein écran (cache la barre de tâches)
                # Obtenir la résolution maximale de l'écran
                display_modes = pygame.display.list_modes()
                if display_modes and display_modes != -1:
                    # Prendre la résolution la plus grande (la première de la liste)
                    max_width, max_height = display_modes[0]
                else:
                    # Fallback si list_modes ne fonctionne pas
                    screen_info = pygame.display.Info()
                    max_width, max_height = screen_info.current_w, screen_info.current_h
                image = pygame.transform.scale(image, (max_width, max_height))
            elif zoom == 0:
                if igw/igh > 1.5:
                    image = pygame.transform.scale(image, (preview_width,int(preview_height * 0.75)))
                else:
                    image = pygame.transform.scale(image, (preview_width,preview_height))
            windowSurfaceObj.blit(image, (0,0))

            # Afficher les contrôles de réglage stretch si en mode stretch (rpicam-vid)
            # Ne PAS afficher pendant le stacking (livestack ou luckystack)
            if stretch_mode == 1 and not livestack_active and not luckystack_active:
                # Icône unique: ISP en mode RAW, ADJ en mode RGB/YUV
                is_raw = (raw_format >= 2)
                draw_stretch_hand_icon(max_width, max_height, stretch_adjust_mode == 1, is_raw_mode=is_raw)
                # Bouton STACK pour lancer le live stacking (pas en mode lucky)
                draw_livestack_button(max_width, max_height, is_raw_mode=is_raw)

                # Afficher le panneau approprié selon le mode
                if stretch_adjust_mode == 1 and img_array is not None:
                    if is_raw:
                        # Mode RAW: panneau ISP complet (ISP + stretch)
                        _raw_slider_rects = draw_raw_controls(max_width, max_height, img_array)
                    else:
                        # Mode RGB/YUV: panneau stretch uniquement
                        _stretch_slider_rects = draw_stretch_controls(max_width, max_height, img_array)

    # Ne pas afficher les overlays en mode stretch
    if (zoom > 0 or foc_man == 1 or focus_mode == 1 or histogram > 0) and stretch_mode == 0:
        # Utiliser array3d au lieu de pixels3d pour ne pas verrouiller la surface
        # Cela améliore grandement la fluidité de l'affichage en mode focus et histogram
        image2 = pygame.surfarray.array3d(image)
        # Transposer car array3d retourne (width, height, channels) au lieu de (height, width, channels)
        image2_transposed = np.transpose(image2, (1, 0, 2))
        crop2 = image2_transposed[xy-histarea:xy+histarea,xx-histarea:xx+histarea]
        gray = cv2.cvtColor(crop2,cv2.COLOR_RGB2GRAY)
        
        # Histogramme OPTIMISÉ avec numpy vectorisé (80-95% plus rapide)
        # Ne pas afficher l'histogramme en mode focus
        if histogram > 0 and focus_mode == 0:
            # Calculer les histogrammes avec numpy (100-1000x plus rapide que les boucles Python)
            bins = np.arange(257)  # 0-256 pour np.histogram

            # Initialiser les histogrammes
            rede = greene = bluee = lume = None

            if histogram == 1 or histogram == 5:
                rede, _ = np.histogram(crop2[:,:,0].ravel(), bins=bins)
            if histogram == 2 or histogram == 5:
                greene, _ = np.histogram(crop2[:,:,1].ravel(), bins=bins)
            if histogram == 3 or histogram == 5:
                bluee, _ = np.histogram(crop2[:,:,2].ravel(), bins=bins)
            if histogram == 4 or histogram == 5:
                lume, _ = np.histogram(gray.ravel(), bins=bins)

            # Normalisation linéaire vectorisée (0-100) - évite les warnings et plus robuste
            if lume is not None:
                max_val = lume.max()
                lume = (lume / max_val * 100).astype(int) if max_val > 0 else lume
            if rede is not None:
                max_val = rede.max()
                rede = (rede / max_val * 100).astype(int) if max_val > 0 else rede
            if greene is not None:
                max_val = greene.max()
                greene = (greene / max_val * 100).astype(int) if max_val > 0 else greene
            if bluee is not None:
                max_val = bluee.max()
                bluee = (bluee / max_val * 100).astype(int) if max_val > 0 else bluee

            # Calculer la hauteur du bandeau noir pour utiliser toute la hauteur disponible
            hist_y = int(preview_height * 0.75) + 1
            hist_height = preview_height - hist_y - 2  # -2 pour une petite marge en bas
            # L'histogramme occupe toute la largeur de la zone preview (sans le menu)
            hist_width = preview_width

            # Créer le graphique DIRECTEMENT à la largeur cible (sans scaling)
            # Cela évite les marges natives du scaling pygame
            output = np.zeros((hist_width, hist_height, 3), dtype=np.uint8)

            # Normaliser les valeurs pour la hauteur disponible
            scale_factor = hist_height / 100  # Adapter l'échelle à la nouvelle hauteur

            # Calculer la largeur de chaque bin (256 bins répartis sur hist_width pixels)
            bin_width = hist_width / 256.0

            # Dessiner chaque courbe directement à la bonne largeur
            # IMPORTANT: Dessiner les 256 bins (0-255) pour couvrir toute la largeur
            for i in range(256):
                # Calculer la position x pour ce bin
                x_start = int(i * bin_width)
                x_end = int((i + 1) * bin_width)

                # S'assurer que le dernier bin atteint bien le bord droit
                if i == 255:
                    x_end = hist_width

                if lume is not None and i < len(lume) and lume[i] > 0:
                    if i < 255:
                        y_start = min(int(lume[i] * scale_factor), int(lume[i+1] * scale_factor))
                        y_end = max(int(lume[i] * scale_factor), int(lume[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(lume[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), :] = 255  # Blanc pour luminance

                if rede is not None and i < len(rede) and rede[i] > 0:
                    if i < 255:
                        y_start = min(int(rede[i] * scale_factor), int(rede[i+1] * scale_factor))
                        y_end = max(int(rede[i] * scale_factor), int(rede[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(rede[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), 0] = 255  # Rouge

                if greene is not None and i < len(greene) and greene[i] > 0:
                    if i < 255:
                        y_start = min(int(greene[i] * scale_factor), int(greene[i+1] * scale_factor))
                        y_end = max(int(greene[i] * scale_factor), int(greene[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(greene[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), 1] = 255  # Vert

                if bluee is not None and i < len(bluee) and bluee[i] > 0:
                    if i < 255:
                        y_start = min(int(bluee[i] * scale_factor), int(bluee[i+1] * scale_factor))
                        y_end = max(int(bluee[i] * scale_factor), int(bluee[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(bluee[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), 2] = 255  # Bleu

            graph = pygame.surfarray.make_surface(output)
            graph = pygame.transform.flip(graph, 0, 1)
            graph.set_alpha(180)  # Légèrement plus opaque pour meilleure lisibilité
            # Plus besoin de scaling - déjà à la bonne taille !
            # DEBUG: Cadre rouge pour visualiser les limites de l'histogramme
            pygame.draw.rect(windowSurfaceObj, (255, 0, 0), Rect(0, hist_y, hist_width, hist_height), 2)
            # Afficher l'histogramme sur toute la largeur sans cadre
            windowSurfaceObj.blit(graph, (0, hist_y))
        
        # Nettoyage des variables numpy (array3d ne crée pas de verrou)
        del image2
        del image2_transposed
        del crop2
        
        # ========== METRICS CALCULATION - Optimisé avec frame counter ==========
        # Incrémenter le compteur de frames (variable globale définie ligne 777)
        metrics_frame_counter += 1

        # Calculer les métriques seulement toutes les N frames
        if metrics_frame_counter >= metrics_interval:
            metrics_frame_counter = 0  # Réinitialiser le compteur

            # *** FOCUS METRIC (configurable via menu METRICS) ***
            # Initialiser la variable pour le graphique
            foc = None

            if focus_method > 0:
                foc = calculate_focus(gray, focus_method)

                # Déterminer la couleur du texte selon la qualité du focus
                # Seuils typiques pour astrophotographie (dépendent de la taille de la zone)
                focus_color = 0  # couleur par défaut (gris foncé)
                if foc > 500:
                    focus_color = 1  # vert - excellent focus
                elif foc > 200:
                    focus_color = 2  # jaune - bon focus
                elif foc > 50:
                    focus_color = 3  # rouge - focus moyen/mauvais
                else:
                    focus_color = 3  # rouge - mauvais focus

                text(20,1,focus_color,2,0,f"Focus ({focus_methods[focus_method]}): {int(foc)}",fv* 2,0)

            # *** SNR DISPLAY (configurable via menu METRICS) ***
            if snr_display == 1:
                try:
                    # Convertir la surface pygame en array pour le calcul SNR
                    image_array = pygame.surfarray.array3d(image)
                    # Transposer car pygame utilise (width, height, channels) et opencv utilise (height, width, channels)
                    image_array = np.transpose(image_array, (1, 0, 2))
                    snr_value = calculate_snr(image_array)

                    # Seuils pour ratio linéaire (adapté à l'astrophotographie)
                    snr_color = 0  # couleur par défaut (gris foncé)
                    if snr_value > 20:
                        snr_color = 1  # vert si excellent SNR
                    elif snr_value > 10:
                        snr_color = 2  # jaune si bon SNR
                    elif snr_value > 5:
                        snr_color = 2  # jaune si SNR moyen
                    else:
                        snr_color = 3  # rouge si mauvais SNR

                    # Affichage au format ratio (ex: 15.2:1)
                    text(20,2,snr_color,2,0,"SNR = " + str(round(snr_value, 1)) + ":1",fv* 2,0)
                except:
                    text(20,2,0,2,0,"SNR = N/A",fv* 2,0)

            # *** STAR METRIC (configurable via menu METRICS: OFF/HFR/FWHM) ***
            # Initialiser les variables pour les graphiques
            hfr_val = None
            fwhm_val = None

            if star_metric > 0 and (focus_mode == 1 or histogram > 0):
                # Utiliser la même zone que le réticule (histarea)
                star_val = None
                star_label = "STAR"

                if star_metric == 1:
                    # HFR : Half Flux Radius (robuste aux aigrettes)
                    star_val = calculate_hfr(image, xx, xy, histarea)
                    hfr_val = star_val  # Stocker pour le graphique
                    star_label = "HFR"
                    # Seuils HFR (en pixels, plus petit = meilleur)
                    threshold_excellent = 2
                    threshold_good = 3.5
                    threshold_medium = 5

                elif star_metric == 2:
                    # FWHM : Full Width Half Maximum
                    star_val = calculate_fwhm(image, xx, xy, histarea)
                    fwhm_val = star_val  # Stocker pour le graphique
                    star_label = "FWHM"
                    # Seuils FWHM (en pixels, plus petit = meilleur)
                    threshold_excellent = 5
                    threshold_good = 10
                    threshold_medium = 20

                if star_val is not None:
                    # Déterminer la couleur selon la qualité
                    if star_val < threshold_excellent:
                        star_color = 1  # vert (excellent)
                    elif star_val < threshold_good:
                        star_color = 2  # jaune (bon)
                    elif star_val < threshold_medium:
                        star_color = 2  # jaune (moyen)
                    else:
                        star_color = 3  # rouge (mauvais)

                    # Affichage texte - position adaptée selon le mode
                    # En preview: ligne 3 (juste sous SNR), en focus: ligne 4
                    star_line = 3 if focus_mode == 0 else 4
                    text(20, star_line, star_color, 2, 0, f"{star_label}: {round(star_val, 2)}", fv * 2, 0)
                else:
                    # Afficher "N/A" si pas d'étoile détectée
                    star_line = 3 if focus_mode == 0 else 4
                    text(20, star_line, 0, 2, 0, f"{star_label}: N/A", fv * 2, 0)

            # En mode focus : afficher 2 graphiques élégants dans le bandeau noir
            # Graphique 1 (gauche) : HFR ou FWHM selon star_metric
            # Graphique 2 (droite) : Focus selon focus_method
            # CORRECTION : Afficher seulement en mode focus (focus_mode == 1)
            if focus_mode == 1:
                try:
                    graph_y = int(preview_height * 0.75) + 1
                    graph_height = preview_height - graph_y - 2

                    # Largeur de chaque graphique = moitié du bandeau moins marges
                    graph_width = int((preview_width - 40) / 2)  # -40 pour marges (10+10+10+10)

                    # Graphique 1 (gauche) : Métrique stellaire (HFR ou FWHM selon star_metric)
                    star_surface = None
                    if star_metric == 1 and hfr_val is not None:  # HFR
                        star_surface = update_star_metric_graph(hfr_val, 'HFR')
                    elif star_metric == 2 and fwhm_val is not None:  # FWHM
                        star_surface = update_star_metric_graph(fwhm_val, 'FWHM')

                    if star_surface is not None and alt_dis < 2:
                        graph1_x = 10
                        graph1_resized = pygame.transform.scale(star_surface, (graph_width, graph_height))

                        # Cadre élégant avec couleur gradient
                        pygame.draw.rect(windowSurfaceObj, (100, 255, 200),
                                       Rect(graph1_x - 2, graph_y - 2,
                                            graph1_resized.get_width() + 4,
                                            graph1_resized.get_height() + 4), 2)

                        windowSurfaceObj.blit(graph1_resized, (graph1_x, graph_y))

                    # Graphique 2 (droite) : Focus selon focus_method
                    if focus_method > 0 and foc is not None:
                        focus_surface = update_focus_graph(foc, focus_methods[focus_method])
                        if focus_surface is not None and alt_dis < 2:
                            graph2_x = 10 + graph_width + 20  # Après le premier graphique + marge
                            graph2_resized = pygame.transform.scale(focus_surface, (graph_width, graph_height))

                            # Cadre élégant avec couleur gradient
                            pygame.draw.rect(windowSurfaceObj, (100, 200, 255),
                                           Rect(graph2_x - 2, graph_y - 2,
                                                graph2_resized.get_width() + 4,
                                                graph2_resized.get_height() + 4), 2)

                            windowSurfaceObj.blit(graph2_resized, (graph2_x, graph_y))

                except Exception as e:
                    pass  # Ignorer les erreurs d'affichage des graphiques silencieusement

        # Rectangle rouge et croix d'analyse - SEULEMENT en mode focus
        if focus_mode == 1:
            # Adapter la taille d'affichage du réticule au zoom
            # Le réticule doit être PLUS GRAND à l'écran quand on zoome
            # pour représenter la même zone physique du capteur (car l'image est agrandie)
            zoom_factors = {0: 1.0, 1: 1.6, 2: 2.4, 3: 3.0, 4: 3.9, 5: 5.8}
            zoom_factor = zoom_factors.get(zoom, 1.0)
            histarea_display = int(histarea * zoom_factor)

            # Limiter la taille max pour éviter un réticule trop grand
            histarea_display = min(histarea_display, int(preview_width / 3))

            pygame.draw.rect(windowSurfaceObj,redColor,Rect(xx-histarea_display,xy-histarea_display,histarea_display*2,histarea_display*2),1)
            pygame.draw.line(windowSurfaceObj,(255,255,255),(xx-int(histarea_display/2),xy),(xx+int(histarea_display/2),xy),1)
            pygame.draw.line(windowSurfaceObj,(255,255,255),(xx,xy-int(histarea_display/2)),(xx,xy+int(histarea_display/2)),1)

    # Mode preview (zoom == 0 ET focus_mode == 0) - Ne pas afficher en mode stretch
    if zoom == 0 and focus_mode == 0 and stretch_mode == 0:
        text(0,0,6,2,0,"Preview",fv* 2,0)

    # Mode preview (zoom == 0)
    if zoom == 0:
        #pygame.draw.rect(windowSurfaceObj,blackColor,Rect(0,0,int(preview_width/4.5),int(preview_height/8)),0)
        # Ne pas afficher le texte en mode stretch
        if stretch_mode == 0:
            text(0,0,6,2,0,"Preview",fv* 2,0)
        zxp = (zx -((preview_width/2) / (igw/preview_width)))
        zyp = (zy -((preview_height/2) / (igh/preview_height)))
        zxq = (zx - zxp) * 2
        zyq = (zy - zyp) * 2
        if zxp + zxq > preview_width:
            zx = preview_width - int(zxq/2)
            zxp = (zx -((preview_width/2) / (igw/preview_width)))
            zxq = (zx - zxp) * 2
        if zyp + zyq > preview_height:
            zy = preview_height - int(zyq/2)
            zyp = (zy -((preview_height/2) / (igh/preview_height)))
            zyq = (zy - zyp) * 2
        if zxp < 0:
            zx = int(zxq/2) + 1
            zxp = 0
            zxq = (zx - zxp) * 2
        if zyp < 0:
            zy = int(zyq/2) + 1
            zyp = 0
            zyq = (zy - zyp) * 2
        if preview_width < 800:
            gw = 2
        else:
            gw = 1
        # Ne pas afficher le rectangle de focus en mode stretch (déjà géré plus bas)
        if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and fxz != 1 and zoom == 0 and stretch_mode == 0:
            pygame.draw.rect(windowSurfaceObj,(200,0,0),Rect(int(fxx*preview_width),int(fxy*preview_height*.75),int(fxz*preview_width),int(fyz*preview_height)),1)

    # *** CRUCIAL : Mettre à jour l'affichage pygame ***
    pygame.display.update()
    
    # *** CETTE PARTIE EST CRUCIALE : Mode preview (zoom == 0) ***
    if zoom == 0:
        #pygame.draw.rect(windowSurfaceObj,blackColor,Rect(0,0,int(preview_width/4.5),int(preview_height/8)),0)
        # Ne pas afficher le texte en mode stretch
        if stretch_mode == 0:
            text(0,0,6,2,0,"Preview",fv* 2,0)
        zxp = (zx -((preview_width/2) / (igw/preview_width)))
        zyp = (zy -((preview_height/2) / (igh/preview_height)))
        zxq = (zx - zxp) * 2
        zyq = (zy - zyp) * 2
        if zxp + zxq > preview_width:
            zx = preview_width - int(zxq/2)
            zxp = (zx -((preview_width/2) / (igw/preview_width)))
            zxq = (zx - zxp) * 2
        if zyp + zyq > preview_height:
            zy = preview_height - int(zyq/2)
            zyp = (zy -((preview_height/2) / (igh/preview_height)))
            zyq = (zy - zyp) * 2
        if zxp < 0:
            zx = int(zxq/2) + 1
            zxp = 0
            zxq = (zx - zxp) * 2
        if zyp < 0:
            zy = int(zyq/2) + 1
            zyp = 0
            zyq = (zy - zyp) * 2
        if preview_width < 800:
            gw = 2
        else:
            gw = 1
        # Ne pas afficher le rectangle de focus en mode stretch
        if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and fxz != 1 and zoom == 0 and stretch_mode == 0:
            pygame.draw.rect(windowSurfaceObj,(200,0,0),Rect(int(fxx*preview_width),int(fxy*preview_height*.75),int(fxz*preview_width),int(fyz*preview_height)),1)

        pygame.display.update()

    if buttonSTR.is_pressed:
        type = pygame.MOUSEBUTTONUP
        if str_cap == 2:
            click_event = pygame.event.Event(type, {"button": 3, "pos": (0,0)})
        else:
            click_event = pygame.event.Event(type, {"button": 1, "pos": (0,0)})
        pygame.event.post(click_event)
    
    #check for any mouse button presses
    for event in pygame.event.get():
      #QUIT
      if event.type == QUIT:
          # Arrêter le thread de capture asynchrone
          if capture_thread is not None:
              capture_thread.stop()
          # Arrêter proprement l'extracteur MJPEG
          if mjpeg_extractor is not None:
              mjpeg_extractor.stop()
          if not use_picamera2 and p is not None:
              os.killpg(p.pid, signal.SIGTERM)
          pygame.quit()
      # MOVE HISTAREA
      elif (event.type == MOUSEBUTTONUP):
        mousex, mousey = event.pos

        # Si on est en mode LiveStack actif, un clic quitte ce mode
        if livestack_active:
            livestack_active = False
            stretch_mode = 0  # Désactiver aussi le stretch
            if livestack is not None:
                # Sauvegarder le résultat final avant de stopper (si activé)
                if ls_save_final == 1:
                    try:
                        # Mode RAW: utiliser traitement externe pour cohérence preview/stack
                        if raw_format >= 2:
                            save_with_external_processing(livestack, raw_format_name=raw_formats[raw_format])
                        else:
                            livestack.save(raw_format_name=raw_formats[raw_format])
                        print("[LIVESTACK] Image finale sauvegardée")
                    except Exception as e:
                        print(f"[LIVESTACK] Erreur sauvegarde: {e}")
                else:
                    print("[LIVESTACK] Sauvegarde finale désactivée (ls_save_final=0)")
                livestack.stop()
            print("[LIVESTACK] Mode désactivé (clic)")
            # Restaurer le mode d'affichage normal (avec l'interface)
            if frame == 1:
                if fullscreen == 1:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                else:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
            else:
                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)
            # Effacer l'écran (remplir de noir)
            windowSurfaceObj.fill((0, 0, 0))
            # Redessiner le menu pour restaurer l'affichage normal
            Menu()
            pygame.display.update()

            # *** Reconfigurer caméra pour revenir au mode ISP après le RAW ***
            if raw_format >= 2:
                print(f"[LIVESTACK] Retour mode ISP - reconfiguration caméra...")
                kill_preview_process()
                preview()
                print(f"[LIVESTACK] Caméra reconfigurée en mode normal")

            continue

        # Si on est en mode LuckyStack actif, un clic quitte ce mode
        if luckystack_active:
            luckystack_active = False
            stretch_mode = 0  # Désactiver aussi le stretch
            if luckystack is not None:
                # Sauvegarder le résultat final avant de stopper (si activé)
                if ls_lucky_save_final == 1:
                    try:
                        # Mode RAW: utiliser traitement externe pour cohérence preview/stack
                        if raw_format >= 2:
                            save_with_external_processing(luckystack, raw_format_name=raw_formats[raw_format])
                        else:
                            luckystack.save(raw_format_name=raw_formats[raw_format])
                        print("[LUCKYSTACK] Image finale sauvegardée")
                    except Exception as e:
                        print(f"[LUCKYSTACK] Erreur sauvegarde: {e}")
                else:
                    print("[LUCKYSTACK] Sauvegarde finale désactivée (ls_lucky_save_final=0)")
                luckystack.stop()
            print("[LUCKYSTACK] Mode désactivé (clic)")
            # Restaurer le mode d'affichage normal (avec l'interface)
            if frame == 1:
                if fullscreen == 1:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                else:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
            else:
                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)
            # Effacer l'écran (remplir de noir)
            windowSurfaceObj.fill((0, 0, 0))
            # Redessiner le menu pour restaurer l'affichage normal
            Menu()
            pygame.display.update()

            # *** Reconfigurer caméra pour revenir au mode ISP après le RAW ***
            if raw_format >= 2:
                print(f"[LUCKYSTACK] Retour mode ISP - reconfiguration caméra...")
                kill_preview_process()
                preview()
                print(f"[LUCKYSTACK] Caméra reconfigurée en mode normal")

            continue

        # Si on est en mode stretch, gérer les clics selon le mode d'ajustement
        if stretch_mode == 1:
            # Obtenir les dimensions de l'écran fullscreen
            display_modes = pygame.display.list_modes()
            if display_modes and display_modes != -1:
                fs_width, fs_height = display_modes[0]
            else:
                screen_info = pygame.display.Info()
                fs_width, fs_height = screen_info.current_w, screen_info.current_h

            # Vérifier si clic sur le bouton STACK (lance le live stacking)
            # Ne s'applique pas si livestack ou luckystack déjà actif
            if not livestack_active and not luckystack_active and is_click_on_livestack_button(mousex, mousey, fs_width):
                # Activer Live Stack
                livestack_active = True

                # *** IMPORTANT: Reconfigurer caméra pour mode RAW ***
                if raw_format >= 2:
                    print(f"[LIVESTACK] Mode RAW - reconfiguration caméra...")
                    kill_preview_process()
                    preview()
                    print(f"[LIVESTACK] Caméra reconfigurée pour capture RAW")

                # Réinitialiser les compteurs de session Live Stack
                if hasattr(pygame, '_livestack_last_saved'):
                    delattr(pygame, '_livestack_last_saved')
                if hasattr(pygame, '_livestack_stretch_info_shown'):
                    delattr(pygame, '_livestack_stretch_info_shown')
                if hasattr(pygame, '_livestack_stretch_applied_shown'):
                    delattr(pygame, '_livestack_stretch_applied_shown')
                if hasattr(pygame, '_livestack_no_stretch_shown'):
                    delattr(pygame, '_livestack_no_stretch_shown')
                if hasattr(pygame, '_raw_format_mismatch_warning'):
                    delattr(pygame, '_raw_format_mismatch_warning')
                if hasattr(pygame, '_raw_resolution_mismatch_warning'):
                    delattr(pygame, '_raw_resolution_mismatch_warning')
                if hasattr(pygame, '_stacking_mode_shown'):
                    delattr(pygame, '_stacking_mode_shown')

                # Créer l'instance livestack si nécessaire
                if livestack is None:
                    camera_params = {
                        'exposure': sspeed,
                        'gain': gain,
                        'red': red / 10,
                        'blue': blue / 10,
                        'raw_format': raw_formats[raw_format]
                    }
                    livestack = create_advanced_livestack_session(camera_params)

                    livestack.configure(
                        stacking_method=['mean', 'median', 'kappa_sigma', 'winsorized', 'weighted'][ls_stack_method],
                        kappa=ls_stack_kappa / 10.0,
                        iterations=ls_stack_iterations,
                        alignment_mode=ls_alignment_modes[ls_alignment_mode],
                        enable_qc=bool(ls_enable_qc),
                        max_fwhm=ls_max_fwhm / 10.0 if ls_max_fwhm > 0 else 999.0,
                        min_sharpness=ls_min_sharpness / 1000.0 if ls_min_sharpness > 0 else 0.0,
                        max_drift=float(ls_max_drift) if ls_max_drift > 0 else 999999.0,
                        min_stars=int(ls_min_stars),
                        planetary_enable=bool(ls_planetary_enable),
                        planetary_mode=ls_planetary_mode,
                        planetary_disk_min=ls_planetary_disk_min,
                        planetary_disk_max=ls_planetary_disk_max,
                        planetary_disk_threshold=ls_planetary_threshold,
                        planetary_disk_margin=ls_planetary_margin,
                        planetary_disk_ellipse=bool(ls_planetary_ellipse),
                        planetary_window=planetary_windows[ls_planetary_window],
                        planetary_upsample=ls_planetary_upsample,
                        planetary_highpass=bool(ls_planetary_highpass),
                        planetary_roi_center=bool(ls_planetary_roi_center),
                        planetary_corr=ls_planetary_corr / 100.0,
                        planetary_max_shift=float(ls_planetary_max_shift),
                        lucky_enable=False,
                        lucky_buffer_size=ls_lucky_buffer,
                        lucky_keep_percent=float(ls_lucky_keep),
                        lucky_score_method=['laplacian', 'gradient', 'sobel', 'tenengrad'][ls_lucky_score],
                        lucky_stack_method=['mean', 'median', 'sigma_clip'][ls_lucky_stack],
                        lucky_align_enabled=bool(ls_lucky_align),
                        lucky_score_roi_percent=float(ls_lucky_roi),
                        png_stretch=['off', 'ghs', 'asinh'][stretch_preset],
                        png_factor=stretch_factor / 10.0,
                        png_clip_low=0.0 if stretch_preset == 1 else stretch_p_low / 10.0,
                        png_clip_high=100.0 if stretch_preset == 1 else stretch_p_high / 100.0,
                        ghs_D=ghs_D / 10.0,
                        ghs_b=ghs_b / 10.0,
                        ghs_SP=ghs_SP / 100.0,
                        ghs_LP=ghs_LP / 100.0,
                        ghs_HP=ghs_HP / 100.0,
                        preview_refresh=ls_preview_refresh,
                        save_dng="none",
                    )

                is_raw_mode = (raw_format >= 2)
                video_format_map = {0: 'yuv420', 1: 'xrgb8888', 2: 'raw12', 3: 'raw16'}
                livestack.configure(
                    isp_enable=False,
                    isp_config_path=None,
                    video_format=video_format_map.get(raw_format, 'yuv420'),
                    png_stretch='off' if is_raw_mode else 'ghs',
                    ghs_D=ghs_D / 10.0,
                    ghs_b=ghs_b / 10.0,
                    ghs_SP=ghs_SP / 100.0,
                    ghs_LP=ghs_LP / 100.0,
                    ghs_HP=ghs_HP / 100.0,
                )

                livestack.camera_params['raw_format'] = raw_formats[raw_format]
                livestack.reset()
                livestack.start()
                pygame._livestack_last_saved = 0

                qc_status = "ON" if ls_enable_qc else "OFF"
                print(f"[LIVESTACK] Mode activé depuis STRETCH (QC: {qc_status})")
                # Désactiver le mode ajustement
                stretch_adjust_mode = 0
                _stretch_slider_rects = {}
                _raw_slider_rects = {}
                continue

            # Vérifier si clic sur l'icône unique (ISP en mode RAW, ADJ en mode RGB/YUV)
            if is_click_on_stretch_hand_icon(mousex, mousey, fs_width):
                stretch_adjust_mode = 1 - stretch_adjust_mode  # Toggle
                if stretch_adjust_mode == 1:
                    # Activation
                    if raw_format >= 2:
                        # Mode RAW: charger les valeurs ISP depuis la session
                        load_isp_from_session()
                    _stretch_slider_rects = {}
                    _raw_slider_rects = {}
                else:
                    # Désactivation - effacer les slider_rects
                    _stretch_slider_rects = {}
                    _raw_slider_rects = {}
                continue

            # Si mode ajustement actif, vérifier si clic sur un contrôle
            if stretch_adjust_mode == 1:
                if raw_format >= 2:
                    # Mode RAW: utiliser les contrôles ISP
                    if _raw_slider_rects:
                        if handle_raw_slider_click(mousex, mousey, _raw_slider_rects):
                            continue  # Contrôle modifié, ne pas quitter
                else:
                    # Mode RGB/YUV: utiliser les contrôles stretch
                    if _stretch_slider_rects:
                        if handle_stretch_slider_click(mousex, mousey, _stretch_slider_rects):
                            continue  # Slider modifié, ne pas quitter

                # Clic ailleurs que sur contrôle ou icône - ignorer (rester en mode ajustement)
                continue

            # Mode ajustement inactif - un clic quitte le mode stretch
            stretch_mode = 0
            stretch_adjust_mode = 0
            raw_adjust_mode = 0
            _stretch_slider_rects = {}
            _raw_slider_rects = {}
            # Restaurer le mode d'affichage normal (avec l'interface)
            if frame == 1:
                if fullscreen == 1:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                else:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
            else:
                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)
            # Effacer l'écran (remplir de noir)
            windowSurfaceObj.fill((0, 0, 0))
            # Redessiner le menu pour restaurer l'affichage normal
            Menu()
            pygame.display.update()

            # Si on était en mode RAW preview (sans stacking), revenir en mode ISP normal
            if raw_format >= 2 and not (livestack_active or luckystack_active):
                if capture_thread is not None:
                    capture_thread.set_capture_params({'type': 'main'})
                    if show_cmds == 1:
                        print(f"[STRETCH] Sortie mode RAW preview - capture_thread reconfiguré en mode ISP")

            continue

        # Permettre le déplacement du réticule même avec menu ouvert si on est en mode focus
        if mousex < preview_width and mousey < preview_height and mousex != 0 and mousey != 0 and event.button != 3 and (menu == 0 or focus_mode == 1):
            # Calculer histarea_display pour les limites (même logique que l'affichage du réticule)
            zoom_factors = {0: 1.0, 1: 1.6, 2: 2.4, 3: 3.0, 4: 3.9, 5: 5.8}
            zoom_factor = zoom_factors.get(zoom, 1.0)
            histarea_display = int(histarea * zoom_factor)
            histarea_display = min(histarea_display, int(preview_width / 3))

            xx = mousex
            xx = min(xx,preview_width - histarea_display)
            xx = max(xx,histarea_display)
            xy = mousey
            if igw/igh > 1.5 and zoom < 5:
                xy = min(xy,int(preview_height * .75) - histarea_display)
            else:
                xy = min(xy,preview_height - histarea_display)
            xy = max(xy,histarea_display)
            if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and mousex < preview_width and mousey < preview_height *.75 and zoom == 0 and (v3_f_mode == 0 or v3_f_mode == 2):
                fxx = (xx - 25)/preview_width
                xy  = min(xy,int((preview_height - 25) * .75))
                fxy = ((xy - 20) * 1.3333)/preview_height
                fxz = 50/preview_width
                fyz = fxz
                #if fxz != 1 and menu == 0:
                #    text(0,3,3,1,1,"Spot",fv,7)
            elif ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam ==6)) or Pi_Cam == 8) and zoom == 0:
                fxx = 0
                fxy = 0
                fxz = 1
                fzy = 1
                if (v3_f_mode == 0 or v3_f_mode == 2) and menu == 0:
                    text(0,3,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
            if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0:
                restart = 1
        
        # external trigger
        if mousex == 0 and mousey == 0:
            str_btn = 1
            
        # determine button pressed
        if mousex > preview_width or str_btn == 1:
            button_row = int(mousey/bh)
            if mousex > preview_width + bw/2:
                button_pos = 1
            else:
                button_pos = 0
                      
            # capture on STR button press
            if str_btn == 1:
                if str_cap == 0:
                    button_row = 0
                elif str_cap == 1 or str_cap == 2:
                    button_row = 1
                elif str_cap == 3:
                    button_row = 2
                str_btn = 0
              
            if button_row == 0 and menu > 0:
                menu = 0
                Menu()
            # MENU 0
            elif menu == 0: 
                if button_row == 0:
                    # TAKE STILL
                    still = 1
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                        # Attendre que le processus de preview se termine complètement
                        poll = p.poll()
                        while poll == None:
                            poll = p.poll()
                            time.sleep(0.1)

                    button(0,0,1,4)
                    if os.path.exists("PiLibtext.txt"):
                         os.remove("PiLibtext.txt")
                    text(0,0,2,0,1,"    CAPTURING",ft,0)
                    text(0,0,6,2,1,"Please Wait, taking still ...",int(fv*1.7),1)
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%y%m%d%H%M%S")
                    if extns[extn] != 'raw':
                        fname =  pic_dir + str(timestamp) + '.' + extns2[extn]
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-still"
                        else:
                            datastr = "rpicam-still"
                        datastr += " --camera " + str(camera) + " -e " + extns[extn] + " -n "
                        datastr += "-t " + str(timet) + " -o " + fname
                    else:
                        fname =  pic_dir + str(timestamp) + '.' + extns2[extn]
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-still"
                        else:
                             datastr = "rpicam-still"
                        datastr += " --camera " + str(camera) + " -r -n "
                        datastr += "-t " + str(timet) + " -o " + fname
                    datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100) 
                    if mode == 0:
                        datastr += " --shutter " + str(sspeed)
                    else:
                        datastr += " --exposure " + str(modes[mode])
                    if ev != 0:
                        datastr += " --ev " + str(ev)
                    if sspeed > 1000000 and mode == 0:
                        datastr += " --gain " + str(gain) + " --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                    else:    
                        datastr += " --gain " + str(gain)
                        if awb == 0:
                            datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --awb " + awbs[awb]
                    datastr += " --metering " + meters[meter]
                    datastr += " --saturation " + str(saturation/10)
                    datastr += " --sharpness " + str(sharpness/10)
                    datastr += " --quality " + str(quality)
                    if vflip == 1:
                        datastr += " --vflip"
                    if hflip == 1:
                        datastr += " --hflip"
                    datastr += " --denoise " + denoises[denoise]
                    if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                    if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"
                    if Pi_Cam == 4 and scientific == 1:
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                    if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                        if v3_f_mode == 1:
                            if Pi_Cam == 3:
                                datastr += " --lens-position " + str(v3_focus/100)
                            if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                datastr += " --lens-position " + str(focus/100)
                    elif (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0 and fxz == 1:
                        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode] + " --autofocus-on-capture"
                    if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1)or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0 and fxz != 1:
                        datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                    if Pi_Cam == 3 or Pi == 5:
                        datastr += " --hdr " + v3_hdrs_cli[v3_hdr]
                    if (Pi_Cam == 6 or Pi_Cam == 8) and mode == 0 and button_pos == 1:
                        datastr += " --width 4624 --height 3472 " # 16MP superpixel mode for higher light sensitivity
                    elif Pi_Cam == 6 or Pi_Cam == 8:
                        if Pi != 5 and lo_res == 1:
                            datastr += " --width 4624 --height 3472"
                        elif Pi_Cam == 6:
                            datastr += " --width 9152 --height 6944"
                        elif Pi_Cam == 8:
                            datastr += " --width 9248 --height 6944"
                    if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop
                        sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                        if sensor_mode:
                            datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                            datastr += f" --width {sensor_mode[0]} --height {sensor_mode[1]}"
                    elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel (zoom 3 désactivé)
                        # Arrondir à un nombre PAIR pour compatibilité formats vidéo
                        zws = int(igw * zfs[zoom])
                        zhs = int(igh * zfs[zoom])
                        if zws % 2 != 0:
                            zws -= 1  # Forcer pair
                        if zhs % 2 != 0:
                            zhs -= 1  # Forcer pair
                        zxo = ((igw-zws)/2)/igw
                        zyo = ((igh-zhs)/2)/igh
                        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                        datastr += " --width " + str(zws) + " --height " + str(zhs)
                    # Supprimé: ancien zoom manuel (zoom == 5)
                    if False and zoom == 5:
                        zxo = ((igw/2)-(preview_width/2))/igw
                        if igw/igh > 1.5:
                            zyo = ((igh/2)-((preview_height * .75)/2))/igh
                        else:
                            zyo = ((igh/2)-(preview_height/2))/igh
                        if igw/igh > 1.5:
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                        else:
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                    datastr += " --metadata - --metadata-format txt >> PiLibtext.txt"
                    if show_cmds == 1:
                        print (datastr)

                    # MODE PICAMERA2 : Capture directe
                    if use_picamera2 and picam2 is not None:
                        try:
                            # Déterminer la configuration de capture selon l'extension
                            if extns[extn] == 'raw' or extns2[extn] == 'dng':
                                # Capture RAW - utiliser le stream 'raw'
                                picam2.capture_file(fname, name='raw', format=extns[extn])
                            else:
                                # Capture JPEG/PNG/BMP - utiliser le stream 'main'
                                picam2.capture_file(fname, name='main')

                            # Attendre que le fichier soit créé
                            import time
                            timeout = 5  # 5 secondes max
                            start_wait = time.time()
                            while not os.path.exists(fname) and (time.time() - start_wait) < timeout:
                                time.sleep(0.1)

                        except Exception as e:
                            print(f"Erreur capture Picamera2: {e}")
                            # En cas d'erreur, afficher un message
                            text(0,0,3,2,1,"Erreur capture",int(fv*1.5),1)

                    # MODE RPICAM-STILL : Commande système
                    else:
                        os.system(datastr)
                        while not os.path.exists(fname):
                            pass
                    if extns2[extn] == 'jpg' or extns2[extn] == 'bmp' or extns2[extn] == 'png':
                        image = pygame.image.load(fname)
                        if igw/igh > 1.5: 
                            image = pygame.transform.scale(image, (preview_width,int(preview_height * 0.75)))
                        else:
                            image = pygame.transform.scale(image, (preview_width,preview_height))
                        windowSurfaceObj.blit(image, (0,0))
                    dgain = 0
                    again = 0
                    etime = 0
                    if os.path.exists("PiLibtext.txt"):
                        with open("PiLibtext.txt", "r") as file:
                            line = file.readline()
                            check = line.split("=")
                            if check[0] == "DigitalGain":
                                dgain = check[1][:-1]
                            if check[0] == "AnalogueGain":
                                again = check[1][:-1]
                            if check[0] == "ExposureTime":
                                etime = check[1][:-1]
                            while line:
                                line = file.readline()
                                check = line.split("=")
                                if check[0] == "DigitalGain":
                                    dgain = check[1][:-1]
                                if check[0] == "AnalogueGain":
                                    again = check[1][:-1]
                                if check[0] == "ExposureTime":
                                    etime = check[1][:-1]
                    text(0,22,6,2,1,"Ana Gain: " + str(again) + " Dig Gain: " + str(dgain) + " Exp Time: " + str(etime) +"uS",int(fv*1.5),1)
                    text(0,0,6,2,1,fname,int(fv*1.5),1)
                    pygame.display.update()
                    time.sleep(2)
                    pygame.draw.rect(windowSurfaceObj,blackColor,Rect(0,int(preview_height * .75),preview_width,preview_height /4),0)
                    still = 0
                    menu = 0
                    Menu()
                    restart = 2
                        
                elif button_row == 1 and event.button != 3:

                    # TAKE VIDEO
                    video = 1
                    picam2_was_paused = False  # Initialiser pour éviter NameError
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                        # Attendre que le processus de preview se termine complètement
                        poll = p.poll()
                        while poll == None:
                            poll = p.poll()
                            time.sleep(0.1)

                    button(0,1,1,3)
                    if Pi == 5:
                        text(0,1,2,0,1,"    RECORDING",ft,1)
                    else:
                        text(0,1,2,0,1,"       STOP ",ft,0)
                    text(0,0,6,2,1,"Please Wait, taking video ...",int(fv*1.7),1)
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%y%m%d%H%M%S")
                    # Déterminer l'extension du fichier
                    if codecs[codec].startswith('ser_'):
                        vname = vid_dir + str(timestamp) + ".ser"
                    else:
                        vname = vid_dir + str(timestamp) + "." + codecs2[codec]

                    if codecs[codec].startswith('ser_'):
                        # ==== CAPTURE POUR FORMAT SER ====
                        # Déterminer le format SER selon le codec choisi
                        if codecs[codec] == 'ser_yuv':
                            ser_format = 0  # YUV420
                        elif codecs[codec] == 'ser_rgb':
                            ser_format = 1  # RGB888
                        elif codecs[codec] == 'ser_xrgb':
                            ser_format = 2  # XRGB8888

                        # Choix de la méthode selon ser_format:
                        # 0 = YUV420 (rpicam-vid)
                        # 1 = RGB888 (Picamera2)
                        # 2 = XRGB8888 (Picamera2)

                        if ser_format == 0:
                            # ==== MÉTHODE YUV420 (rpicam-vid) ====
                            text(0,0,6,2,1,"Recording YUV420 for SER...",int(fv*1.7),1)

                            # Construire la commande pour capturer en YUV420
                            if lver != "bookwo" and lver != "trixie":
                                datastr = "libcamera-vid"
                            else:
                                datastr = "rpicam-vid"

                            # Capturer en fichier YUV420 sur le NVMe (même répertoire que les vidéos finales)
                            temp_video = vid_dir + "ser_temp_video.yuv"

                            datastr += " --camera " + str(camera) + " -t " + str(vlen * 1000)
                            datastr += " -o " + temp_video
                            datastr += " --codec yuv420"

                            # IMPORTANT: Pour un vrai ROI (méthode Test2), l'ordre doit être:
                            # --codec yuv420 --mode ... --roi ... --width ... --height ...

                            # Gestion du ROI (Region Of Interest) - DOIT venir AVANT --width/--height
                            if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop natif (pas de ROI)
                                sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                                if sensor_mode:
                                    # Utiliser directement le mode sensor natif sans ROI
                                    datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                            elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel (zoom 3 désactivé)
                                # ROI centré sur le mode capteur natif (vrai ROI)
                                # Déterminer la profondeur de bits selon la caméra
                                if Pi_Cam == 4:  # Pi HQ
                                    sensor_bits = 12
                                else:
                                    sensor_bits = 10

                                # Arrondir à un nombre PAIR pour compatibilité YUV420
                                zws = int(igw * zfs[zoom])
                                zhs = int(igh * zfs[zoom])
                                if zws % 2 != 0:
                                    zws -= 1  # Forcer pair
                                if zhs % 2 != 0:
                                    zhs -= 1  # Forcer pair
                                zxo = ((igw-zws)/2)/igw
                                zyo = ((igh-zhs)/2)/igh
                                datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits) + "  --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                            # Supprimé: ancien zoom manuel (zoom == 5)
                            elif False and zoom == 5:
                                # ROI manuel centré basé sur preview_width/height
                                # Déterminer la profondeur de bits selon la caméra
                                if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                                    sensor_bits = 12
                                else:
                                    sensor_bits = 10

                                zxo = ((igw/2)-(preview_width/2))/igw
                                if igw/igh > 1.5:
                                    zyo = ((igh/2)-((preview_height * .75)/2))/igh
                                else:
                                    zyo = ((igh/2)-(preview_height/2))/igh

                                datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits)
                                if igw/igh > 1.5:
                                    datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                                else:
                                    datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)

                            # Résolution et dimensions
                            if Pi_Cam == 10 and zoom > 0:  # IMX585: utiliser résolution du mode sensor
                                sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                                if sensor_mode:
                                    actual_vwidth = sensor_mode[0]
                                    actual_vheight = sensor_mode[1]
                                    datastr += f" --width {sensor_mode[0]} --height {sensor_mode[1]}"
                                else:
                                    # Fallback si sensor_mode non disponible
                                    actual_vwidth = vwidth
                                    actual_vheight = vheight
                                    datastr += f" --width {vwidth} --height {vheight}"
                            elif zoom > 0 and zoom <= 5:  # Autres caméras - Zoom fixe avec ROI
                                # Utiliser la résolution choisie par l'utilisateur (vwidth/vheight)
                                # Si c'est une résolution native (affichée en vert), pas de crop
                                # Sinon, utiliser zws/zhs (résolution ROI calculée)
                                actual_vwidth = vwidth
                                actual_vheight = vheight

                                # Ajouter --width et --height
                                datastr += " --width " + str(actual_vwidth) + " --height " + str(actual_vheight)
                            # Supprimé: ancien zoom manuel (zoom == 5)
                            elif False and zoom == 5:
                                # Zoom manuel : utiliser preview_width/height
                                if igw/igh > 1.5:
                                    actual_vwidth = preview_width
                                    actual_vheight = int(preview_height * .75)
                                else:
                                    actual_vwidth = preview_width
                                    actual_vheight = preview_height
                                datastr += " --width " + str(actual_vwidth) + " --height " + str(actual_vheight)
                            else:
                                # Pas de zoom : utiliser la résolution du mode sensor
                                if Pi_Cam == 10:
                                    sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                                    if sensor_mode:
                                        actual_vwidth = sensor_mode[0]
                                        actual_vheight = sensor_mode[1]
                                        datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                                    else:
                                        actual_vwidth = vwidth
                                        actual_vheight = vheight
                                else:
                                    actual_vwidth = vwidth
                                    actual_vheight = vheight
                                datastr += " --width " + str(actual_vwidth) + " --height " + str(actual_vheight)

                            if mode != 0:
                                datastr += " --framerate " + str(fps)
                            else:
                                speed7 = sspeed
                                speed7 = max(speed7,int((1/fps)*1000000))
                                datastr += " --framerate " + str(max(1, min(180, int(1000000/speed7))))

                            if vpreview == 0:
                                datastr += " -n "
                            datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)

                            if mode == 0:
                                datastr += " --shutter " + str(sspeed)
                            else:
                                datastr += " --exposure " + modes[mode]

                            datastr += " --gain " + str(gain)

                            if ev != 0:
                                datastr += " --ev " + str(ev)

                            if awb == 0:
                                datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                            else:
                                datastr += " --awb " + awbs[awb]

                            datastr += " --metering " + meters[meter]
                            datastr += " --saturation " + str(saturation/10)
                            datastr += " --sharpness " + str(sharpness/10)
                            datastr += " --denoise " + denoises[denoise]

                            if vflip == 1:
                                datastr += " --vflip"
                            if hflip == 1:
                                datastr += " --hflip"

                            if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                                datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                            if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
                                datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"

                            if show_cmds == 1:
                                print(datastr)

                            # Vérifier l'espace disque disponible avant d'enregistrer
                            # Calculer la taille estimée du fichier YUV420
                            duration_sec = vlen / 1000.0
                            if mode != 0:
                                actual_fps = fps
                            else:
                                speed7 = sspeed
                                speed7 = max(speed7,int((1/fps)*1000000))
                                actual_fps = max(1, min(180, int(1000000/speed7)))

                            num_frames = int(duration_sec * actual_fps)
                            # YUV420 = 1.5 bytes per pixel (Y + U/4 + V/4)
                            estimated_size = int(actual_vwidth * actual_vheight * 1.5 * num_frames)

                            try:
                                import shutil
                                stat = shutil.disk_usage(vid_dir)
                                free_space = stat.free
                                print(f"[DEBUG YUV] Estimated size: {estimated_size/(1024*1024):.1f} MB, Free space (NVMe): {free_space/(1024*1024):.1f} MB")

                                if free_space < estimated_size * 1.1:  # 10% de marge
                                    error_msg = f"Espace disque insuffisant ! Nécessaire: {estimated_size/(1024*1024):.1f} MB, Disponible: {free_space/(1024*1024):.1f} MB"
                                    print(f"[ERROR] {error_msg}")
                                    text(0,0,6,2,1,error_msg,int(fv*1.7),1)
                                    time.sleep(3)
                                    # Ne pas continuer l'enregistrement
                                    continue  # Retourner au menu principal
                            except Exception as disk_check_error:
                                print(f"[WARNING] Could not check disk space: {disk_check_error}")

                            # Arrêter temporairement Picamera2 pour libérer la caméra
                            picam2_was_paused = pause_picamera2()

                            try:
                                print(f"[DEBUG SER] Starting SER video recording: {vname}")
                                print(f"[DEBUG SER] Temp file: {temp_video}")
                                print(f"[DEBUG SER] Command: {datastr}")

                                # Capturer la vidéo
                                os.system(datastr)

                                print(f"[DEBUG SER] Recording finished")
                                print(f"[DEBUG SER] Temp file exists: {os.path.exists(temp_video)}")
                                if os.path.exists(temp_video):
                                    print(f"[DEBUG SER] Temp file size: {os.path.getsize(temp_video)} bytes")

                                # Convertir RAW unpacked (SRGGB16) en SER avec le script qui marche !
                                text(0,0,6,2,1,"Converting SRGGB16 to RGB24...",int(fv*1.7),1)

                                # Calculer le FPS réel
                                if mode != 0:
                                    actual_fps = fps
                                else:
                                    speed7 = sspeed
                                    speed7 = max(speed7,int((1/fps)*1000000))
                                    actual_fps = max(1, min(180, int(1000000/speed7)))

                                # Vérifier que le fichier vidéo YUV420 existe
                                if not os.path.exists(temp_video):
                                    error_msg = "ERREUR: Fichier YUV420 non créé ! (Possible manque d'espace disque)"
                                    print(f"[ERROR] {error_msg}")
                                    text(0,0,6,2,1,error_msg,int(fv*1.7),1)
                                    time.sleep(3)
                                else:
                                    # Avec résolutions standards, pas besoin de détection automatique
                                    # La résolution est exactement celle spécifiée dans actual_vwidth/actual_vheight
                                    print(f"[DEBUG SER] Converting with standard resolution: {actual_vwidth}×{actual_vheight}")

                                    # Vérification critique: calculer taille attendue du fichier
                                    file_size = os.path.getsize(temp_video)
                                    duration_sec = vlen / 1000.0
                                    num_frames = int(duration_sec * actual_fps)
                                    expected_size = int(actual_vwidth * actual_vheight * 1.5 * num_frames)

                                    print(f"[DEBUG SER] File size: {file_size} bytes, Expected: {expected_size} bytes, Frames: {num_frames}")

                                    # Vérifier si le fichier est trop petit (signe d'erreur d'enregistrement)
                                    if expected_size > 0 and file_size < expected_size * 0.5:  # Moins de 50% de la taille attendue
                                        error_msg = f"ERREUR: Fichier YUV420 corrompu ou incomplet ({file_size/(1024*1024):.1f}/{expected_size/(1024*1024):.1f} MB)"
                                        print(f"[ERROR] {error_msg}")
                                        text(0,0,6,2,1,error_msg,int(fv*1.7),1)

                                        # Nettoyer le fichier corrompu
                                        try:
                                            os.remove(temp_video)
                                            print("[DEBUG] Fichier corrompu supprimé")
                                        except:
                                            pass

                                        time.sleep(3)
                                    elif expected_size > 0 and abs(file_size - expected_size) / expected_size > 0.1:  # Plus de 10% de différence
                                        print(f"[DEBUG SER] Warning: File size mismatch ({100*abs(file_size-expected_size)/expected_size:.1f}% difference)")
                                        print(f"[DEBUG SER] This may indicate incorrect resolution or frame count")

                                        # Convertir quand même mais avec un avertissement
                                        text(0,0,6,2,1,f"Converting YUV420 to SER ({actual_vwidth}×{actual_vheight}) @ {actual_fps} fps...",int(fv*1.7),1)

                                        success, frame_count, msg = convert_yuv420_to_ser(
                                            temp_video, vname, actual_vwidth, actual_vheight, fps=actual_fps
                                        )

                                        if success:
                                            text(0,0,6,2,1,vname + f" - {frame_count} frames @ {actual_fps} fps",int(fv*1.5),1)
                                        else:
                                            text(0,0,6,2,1,f"Conversion error: {msg}",int(fv*1.7),1)

                                        # Supprimer le fichier vidéo temporaire
                                        try:
                                            os.remove(temp_video)
                                        except:
                                            pass

                                        time.sleep(2)
                                    else:
                                        # Taille correcte, convertir normalement
                                        text(0,0,6,2,1,f"Converting YUV420 to SER ({actual_vwidth}×{actual_vheight}) @ {actual_fps} fps...",int(fv*1.7),1)

                                        success, frame_count, msg = convert_yuv420_to_ser(
                                            temp_video, vname, actual_vwidth, actual_vheight, fps=actual_fps
                                        )

                                        if success:
                                            text(0,0,6,2,1,vname + f" - {frame_count} frames @ {actual_fps} fps",int(fv*1.5),1)
                                        else:
                                            text(0,0,6,2,1,f"Conversion error: {msg}",int(fv*1.7),1)

                                        # Supprimer le fichier vidéo temporaire
                                        try:
                                            os.remove(temp_video)
                                        except:
                                            pass

                                        time.sleep(2)

                            except Exception as e:
                                print(f"[ERROR] YUV420 SER recording failed: {e}")
                                import traceback
                                traceback.print_exc()
                                text(0,0,6,2,1,f"YUV420 SER error: {str(e)}",int(fv*1.7),1)
                                time.sleep(3)

                            finally:
                                # Redémarrer Picamera2 si il avait été mis en pause
                                if picam2_was_paused:
                                    resume_picamera2()

                        elif ser_format in [1, 2]:
                            # ==== MÉTHODE RGB888/XRGB8888 (Picamera2) ====
                            format_name = ser_formats[ser_format]  # "RGB888" ou "XRGB8888"
                            text(0,0,6,2,1,f"Recording {format_name} for SER...",int(fv*1.7),1)

                            # Déterminer bytes_per_pixel selon le format
                            bytes_per_pixel = 3 if ser_format == 1 else 4

                            # Déterminer la résolution de capture selon le zoom (modes IMX585)
                            # IMPORTANT: Toujours utiliser vwidth/vheight (résolution choisie par l'utilisateur)
                            # pour éviter le crop non désiré sur les résolutions natives
                            actual_vwidth, actual_vheight = vwidth, vheight

                            # Calculer le FPS réel
                            if mode != 0:
                                actual_fps = fps
                            else:
                                speed7 = sspeed
                                speed7 = max(speed7,int((1/fps)*1000000))
                                actual_fps = max(1, min(180, int(1000000/speed7)))

                            # Calculer la durée et le nombre de frames
                            # IMPORTANT: vlen est en SECONDES (pas en millisecondes)
                            duration_sec = vlen  # vlen déjà en secondes
                            num_frames = int(duration_sec * actual_fps)

                            print(f"[DEBUG SER RGB] vlen={vlen}s, fps={actual_fps}, expected frames={num_frames}")

                            # Fichier temporaire pour le flux RGB brut sur le NVMe (même répertoire que les vidéos finales)
                            temp_rgb = vid_dir + "ser_temp_rgb.raw"

                            # Vérifier l'espace disque disponible avant d'enregistrer
                            estimated_size = num_frames * actual_vwidth * actual_vheight * bytes_per_pixel
                            try:
                                import shutil
                                stat = shutil.disk_usage(vid_dir)
                                free_space = stat.free
                                print(f"[DEBUG RGB] Estimated size: {estimated_size/(1024*1024):.1f} MB, Free space (NVMe): {free_space/(1024*1024):.1f} MB")

                                if free_space < estimated_size * 1.1:  # 10% de marge
                                    error_msg = f"Espace disque insuffisant ! Nécessaire: {estimated_size/(1024*1024):.1f} MB, Disponible: {free_space/(1024*1024):.1f} MB"
                                    print(f"[ERROR] {error_msg}")
                                    text(0,0,6,2,1,error_msg,int(fv*1.7),1)
                                    time.sleep(3)
                                    # Ne pas continuer l'enregistrement
                                    picam2_was_paused = pause_picamera2()
                                    if picam2_was_paused:
                                        resume_picamera2()
                                    continue  # Sortir de cette section et retourner au menu
                            except Exception as disk_check_error:
                                print(f"[WARNING] Could not check disk space: {disk_check_error}")

                            # Pause Picamera2 pour libérer la caméra
                            picam2_was_paused = pause_picamera2()

                            # Créer une instance temporaire de Picamera2 pour la capture RGB
                            temp_picam2 = None

                            try:
                                # Importer Picamera2
                                from picamera2 import Picamera2

                                # Déterminer le fichier de tuning ISP (même que rpicam-vid)
                                tuning_file = None
                                if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                                    tuning_file = "/home/" + Home_Files[0] + "/imx290a.json"
                                elif Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
                                    tuning_file = "/home/" + Home_Files[0] + "/imx585_lowlight.json"

                                # Créer une nouvelle instance temporaire avec le fichier de tuning
                                print(f"[DEBUG SER RGB] Creating temporary Picamera2 instance on camera {camera}")
                                if tuning_file:
                                    print(f"[DEBUG SER RGB] Using tuning file: {tuning_file}")
                                    temp_picam2 = Picamera2(camera, tuning=Picamera2.load_tuning_file(tuning_file))
                                else:
                                    temp_picam2 = Picamera2(camera)

                                # Capturer directement en RGB (pas d'intermédiaire YUV pour éviter perte qualité)
                                if ser_format == 1:
                                    capture_format = "RGB888"
                                else:
                                    capture_format = "XRGB8888"

                                # Configuration pour IMX585 avec mode sensor spécifique
                                if Pi_Cam == 10:
                                    sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                                    config = temp_picam2.create_video_configuration(
                                        main={"size": (actual_vwidth, actual_vheight), "format": capture_format},
                                        raw={"size": sensor_mode} if sensor_mode else None
                                    )
                                else:
                                    config = temp_picam2.create_video_configuration(
                                        main={"size": (actual_vwidth, actual_vheight), "format": capture_format}
                                    )

                                # Appliquer les paramètres caméra
                                if mode == 0:
                                    config["controls"]["ExposureTime"] = sspeed
                                    config["controls"]["AnalogueGain"] = gain

                                temp_picam2.configure(config)

                                # ==========================================
                                # PROFIL "SPEED" POUR HAUTE VITESSE SER
                                # ==========================================
                                # Sauvegarder les paramètres ISP actuels
                                saved_isp_params = {
                                    'brightness': brightness,
                                    'contrast': contrast,
                                    'saturation': saturation,
                                    'sharpness': sharpness
                                }

                                # Appliquer profil "Speed" : désactiver les traitements ISP lourds
                                cam_controls = {}
                                cam_controls["Brightness"] = 0.0       # Neutre (pas de boost)
                                cam_controls["Contrast"] = 1.0         # Neutre (×1)
                                cam_controls["Saturation"] = 1.0       # Neutre (×1)
                                cam_controls["Sharpness"] = 0.0        # DÉSACTIVÉ (sharpening = lent)

                                if awb == 0:
                                    cam_controls["ColourGains"] = (red/10.0, blue/10.0)

                                # Désactiver denoise via NoiseReductionMode
                                try:
                                    from picamera2.controls import NoiseReductionModeEnum
                                    cam_controls["NoiseReductionMode"] = NoiseReductionModeEnum.Off
                                except:
                                    pass  # Si NoiseReductionMode non disponible, ignorer

                                temp_picam2.set_controls(cam_controls)

                                print(f"[DEBUG SER RGB] Starting {format_name} capture (ISP): {actual_vwidth}x{actual_vheight} @ {actual_fps} fps")
                                print(f"[DEBUG SER RGB] Duration: {duration_sec}s ({num_frames} frames)")
                                print(f"[DEBUG SER RGB] Using ISP 'SPEED' profile (sharpness=0, denoise=off, neutral colors)")
                                print(f"[DEBUG SER RGB] Using optimized callback mode for high-speed capture")

                                # Importer numpy pour inverser les canaux
                                import numpy as np
                                import threading

                                # État partagé pour le callback (utiliser un dict pour éviter problème nonlocal)
                                capture_state = {
                                    'frame_count': 0,
                                    'write_error': None,
                                    'output_file': None,
                                    'lock': threading.Lock()
                                }

                                # Fonction de callback pour capturer et écrire les frames rapidement
                                # OPTIMISÉ : écriture directe sans copie ni conversion (conversion faite après)
                                def frame_callback(request):
                                    if capture_state['write_error'] is not None:
                                        return  # Arrêter si erreur détectée

                                    if capture_state['frame_count'] >= num_frames:
                                        return  # Arrêter si nombre de frames atteint

                                    try:
                                        # IMPORTANT: Utiliser make_array().tobytes() au lieu de make_buffer()
                                        # car make_buffer() inclut le stride (padding des lignes), ce qui
                                        # provoque des lignes décalées dans convert_rgb888_to_ser()
                                        frame_array = request.make_array("main")
                                        frame_buffer = frame_array.tobytes()

                                        # Écrire directement sur disque sans conversion
                                        # La conversion RGB→BGR sera faite après pendant convert_rgb888_to_ser
                                        with capture_state['lock']:
                                            if capture_state['output_file'] is not None:
                                                capture_state['output_file'].write(frame_buffer)
                                                capture_state['frame_count'] += 1

                                                # Afficher progression tous les 50 frames
                                                if capture_state['frame_count'] % 50 == 0:
                                                    text(0,0,6,2,1,f"Recording {format_name}: {capture_state['frame_count']}/{num_frames} frames",int(fv*1.7),1)

                                    except (OSError, IOError) as write_err:
                                        capture_state['write_error'] = write_err
                                        print(f"\n[ERROR] Erreur d'écriture à la frame {capture_state['frame_count']}: {write_err}")
                                    except Exception as e:
                                        capture_state['write_error'] = e
                                        print(f"\n[ERROR] Erreur dans callback: {e}")

                                try:
                                    # Ouvrir le fichier temporaire
                                    capture_state['output_file'] = open(temp_rgb, "wb")

                                    # Configurer le framerate dans les contrôles
                                    # FrameDurationLimits = (min_duration_us, max_duration_us)
                                    # Pour forcer le fps: min=max = 1000000/fps microseconds
                                    frame_duration_us = int(1000000 / actual_fps)
                                    controls_fps = {"FrameDurationLimits": (frame_duration_us, frame_duration_us)}
                                    temp_picam2.set_controls(controls_fps)

                                    # Démarrer la capture avec callback
                                    temp_picam2.start()

                                    # Enregistrer le callback pour chaque frame
                                    temp_picam2.post_callback = frame_callback

                                    start_time = time.time()

                                    # Attendre la durée de capture ou le nombre de frames
                                    while capture_state['frame_count'] < num_frames and capture_state['write_error'] is None:
                                        elapsed = time.time() - start_time
                                        if elapsed > duration_sec + 2:  # +2 sec de marge
                                            break
                                        time.sleep(0.1)  # Petit sleep pour ne pas surcharger le CPU

                                    elapsed_time = time.time() - start_time
                                    actual_capture_fps = capture_state['frame_count'] / elapsed_time if elapsed_time > 0 else 0
                                    print(f"[DEBUG SER RGB] Capture terminée: {capture_state['frame_count']} frames en {elapsed_time:.2f}s ({actual_capture_fps:.1f} fps réels)")

                                except (OSError, IOError) as file_error:
                                    # Erreur lors de l'ouverture du fichier
                                    capture_state['write_error'] = file_error
                                    print(f"[ERROR] Impossible d'ouvrir le fichier temporaire: {file_error}")

                                finally:
                                    # Arrêter la capture
                                    temp_picam2.post_callback = None
                                    temp_picam2.stop()
                                    temp_picam2.close()

                                    # Fermer le fichier
                                    if capture_state['output_file'] is not None:
                                        capture_state['output_file'].close()

                                # Vérifier s'il y a eu une erreur d'écriture
                                if capture_state['write_error'] is not None:
                                    # Une erreur s'est produite pendant l'enregistrement
                                    error_msg = f"ERREUR d'enregistrement: {str(capture_state['write_error'])}"

                                    # Identifier le type d'erreur
                                    if "No space left" in str(capture_state['write_error']) or "Errno 28" in str(capture_state['write_error']):
                                        error_msg = f"ERREUR: Espace disque insuffisant ! ({capture_state['frame_count']}/{num_frames} frames enregistrées)"
                                    elif isinstance(capture_state['write_error'], (OSError, IOError)):
                                        error_msg = f"ERREUR I/O: {str(capture_state['write_error'])} ({capture_state['frame_count']}/{num_frames} frames enregistrées)"

                                    print(f"\n[ERROR] {error_msg}")
                                    text(0,0,6,2,1,error_msg,int(fv*1.7),1)

                                    # Nettoyer le fichier temporaire corrompu
                                    try:
                                        if os.path.exists(temp_rgb):
                                            os.remove(temp_rgb)
                                            print("[DEBUG] Fichier temporaire corrompu supprimé")
                                    except Exception as cleanup_err:
                                        print(f"[WARNING] Impossible de supprimer le fichier temporaire: {cleanup_err}")

                                    time.sleep(3)
                                else:
                                    # Pas d'erreur, capture réussie
                                    print(f"[DEBUG SER RGB] Capture finished: {capture_state['frame_count']} frames")
                                    print(f"[DEBUG SER RGB] Temp file size: {os.path.getsize(temp_rgb)} bytes")

                                    # Convertir RGB → SER
                                    text(0,0,6,2,1,f"Converting {format_name} to SER...",int(fv*1.7),1)

                                    success, final_frame_count, msg = convert_rgb888_to_ser(
                                        temp_rgb, vname, actual_vwidth, actual_vheight,
                                        fps=actual_fps, bytes_per_pixel=bytes_per_pixel
                                    )

                                    if success:
                                        text(0,0,6,2,1,vname + f" - {final_frame_count} frames @ {actual_fps} fps",int(fv*1.5),1)
                                    else:
                                        text(0,0,6,2,1,f"Conversion error: {msg}",int(fv*1.7),1)

                                    # Supprimer le fichier temporaire
                                    try:
                                        os.remove(temp_rgb)
                                    except:
                                        pass

                                    time.sleep(2)

                            except Exception as e:
                                print(f"[ERROR] RGB capture failed: {e}")
                                import traceback
                                traceback.print_exc()
                                text(0,0,6,2,1,f"RGB capture error: {str(e)}",int(fv*1.7),1)
                                time.sleep(2)

                            finally:
                                # Nettoyer l'instance temporaire si elle existe
                                if temp_picam2 is not None:
                                    try:
                                        temp_picam2.stop()
                                    except:
                                        pass
                                    try:
                                        temp_picam2.close()
                                    except:
                                        pass

                                # Redémarrer Picamera2 en mode preview
                                if picam2_was_paused:
                                    resume_picamera2()


                    elif codecs2[codec] != 'raw' and not codecs[codec].startswith('ser_'):
                        # Code existant pour les autres formats (sauf SER et RAW)
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-vid"


                    if codecs2[codec] != 'raw' and not codecs[codec].startswith('ser_'):
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-vid"
                        else:
                            datastr = "rpicam-vid"
                        datastr += " --camera " + str(camera) + " -t " + str(vlen * 1000)
                        # Ajouter framerate AVANT -o (output)
                        if mode != 0:
                            datastr += " --framerate " + str(fps)
                        else:
                            speed7 = sspeed
                            speed7 = max(speed7,int((1/fps)*1000000))
                            datastr += " --framerate " + str(max(1, min(180, int(1000000/speed7))))
                        if codecs[codec] != 'h264' and codecs[codec] != 'mp4':
                            datastr += " --codec " + codecs[codec]
                        elif codecs[codec] != 'mp4':
                            prof = h264profiles[profile].split(" ")
                            #datastr += " --profile " + str(prof[0]) + " --level " + str(prof[1])
                            datastr += " --level " + str(prof[1])
                    elif codecs2[codec] == 'raw':
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-raw"
                        else:
                            datastr = "rpicam-raw"
                        datastr += " --camera " + str(camera) + " -t " + str(vlen * 1000) + " -o " + vname + " --framerate " + str(fps)

                    # Le code ci-dessous ne doit PAS s'exécuter pour les codecs 'ser_*' qui construisent leur propre commande
                    if not codecs[codec].startswith('ser_'):
                        if vpreview == 0:
                            datastr += " -n "
                        datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)

                        # IMPORTANT: Pour un vrai ROI (méthode Test2), ajouter --mode et --roi AVANT --width/--height
                        if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop natif (pas de ROI)
                            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                            if sensor_mode:
                                # Utiliser directement le mode sensor natif sans ROI
                                datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                        elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel (zoom 3 désactivé)
                            # Déterminer la profondeur de bits selon la caméra
                            if Pi_Cam == 4:  # Pi HQ
                                sensor_bits = 12
                            else:
                                sensor_bits = 10

                            # Arrondir à un nombre PAIR pour compatibilité YUV420
                            zws = int(igw * zfs[zoom])
                            zhs = int(igh * zfs[zoom])
                            if zws % 2 != 0:
                                zws -= 1  # Forcer pair
                            if zhs % 2 != 0:
                                zhs -= 1  # Forcer pair
                            zxo = ((igw-zws)/2)/igw
                            zyo = ((igh-zhs)/2)/igh
                            datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits) + "  --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        elif False and zoom == 5:
                            # Déterminer la profondeur de bits selon la caméra
                            if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                                sensor_bits = 12
                            else:
                                sensor_bits = 10

                            zxo = ((igw/2)-(preview_width/2))/igw
                            if igw/igh > 1.5:
                                zyo = ((igh/2)-((preview_height * .75)/2))/igh
                            else:
                                zyo = ((igh/2)-(preview_height/2))/igh

                            datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits)
                            if igw/igh > 1.5:
                                datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width / .75)/igw) + "," + str(preview_height/igh)
                            else:
                                datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)

                        # Résolution et dimensions - SEULEMENT si ROI non appliqué
                        # IMPORTANT: Quand --roi est utilisé, NE PAS spécifier --width/--height
                        # Le ROI définit déjà la taille de sortie (crop sans rescale = vrai zoom)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        if False and zoom == 5:
                            if igw/igh > 1.5:
                                datastr += " --width " + str(preview_width) + " --height " + str(int(preview_height * .75))
                            else:
                                datastr += " --width " + str(preview_width) + " --height " + str(preview_height)
                        elif Pi_Cam == 10 and zoom > 0:  # IMX585: ajouter --width et --height du mode sensor
                            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                            if sensor_mode:
                                datastr += f" --width {sensor_mode[0]} --height {sensor_mode[1]}"
                        elif zoom > 0 and zoom <= 5:  # Autres caméras - Zoom fixe avec ROI (zoom 3 désactivé)
                            # NE RIEN FAIRE : le ROI a déjà défini la taille de sortie
                            pass
                        elif Pi_Cam == 4 and vwidth == 2028:
                            datastr += " --mode 2028:1520:12"
                        elif Pi_Cam == 3 and vwidth == 2304 and codec == 0:
                            datastr += " --mode 2304:1296:10 --width 2304 --height 1296"
                        elif Pi_Cam == 3 and vwidth == 2028 and codec == 0:
                            datastr += " --mode 2028:1520:10 --width 2028 --height 1520"
                        elif Pi_Cam == 10:
                            # Déterminer le mode sensor basé sur vwidth/vheight ou zoom
                            if zoom > 0:
                                sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                            else:
                                # Pas de zoom: choisir mode basé sur résolution demandée
                                # Mapping résolution → mode
                                if vwidth == 3856 and vheight == 2180:
                                    sensor_mode = (3856, 2180)  # Mode 1 native
                                elif vwidth == 2880 and vheight == 2160:
                                    sensor_mode = (2880, 2160)  # Mode 2 crop
                                elif vwidth == 1920 and vheight == 1080:
                                    sensor_mode = (1920, 1080)  # Mode 3 crop
                                elif vwidth == 1280 and vheight == 720:
                                    sensor_mode = (1280, 720)   # Mode 4 crop
                                elif vwidth == 800 and vheight == 600:
                                    sensor_mode = (800, 600)    # Mode 5 crop
                                else:
                                    # Fallback: utiliser mode binning
                                    sensor_mode = (1928, 1090)  # Mode 0 binning

                            if sensor_mode:
                                datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                            datastr += f" --width {vwidth} --height {vheight}"
                        else:
                            datastr += " --width " + str(vwidth) + " --height " + str(vheight)

                        # Ajouter les paramètres communs à tous les cas (zoom ou pas)
                        if mode == 0:
                            datastr += " --shutter " + str(sspeed)
                        else:
                            datastr += " --exposure " + modes[mode]
                        datastr += " --gain " + str(gain)
                        if ev != 0:
                            datastr += " --ev " + str(ev)
                        if awb == 0:
                            datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --awb " + awbs[awb]
                        datastr += " --metering " + meters[meter]
                        datastr += " --saturation " + str(saturation/10)
                        datastr += " --sharpness " + str(sharpness/10)
                        datastr += " --denoise "    + denoises[denoise]
                        if vflip == 1:
                            datastr += " --vflip"
                        if hflip == 1:
                            datastr += " --hflip"
                        if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                        if Pi_Cam == 4 and scientific == 1:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                        if Pi_Cam == 5 and foc_man == 1 and Pi == 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx519mf.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx519mf.json"
                        elif Pi_Cam == 5  and foc_man == 1 and Pi != 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx519mff.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx519mff.json"
                        if Pi_Cam == 6  and foc_man == 1 and Pi == 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/arducam_64mf.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/arducam_64mf.json"
                        if Pi_Cam == 6  and foc_man == 1 and Pi != 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/arducam_64mff.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/arducam_64mff.json"
                        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                            if v3_f_mode == 1:
                                if Pi_Cam == 3:
                                    datastr += " --lens-position " + str(v3_focus/100)
                                if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                    datastr += " --lens-position " + str(focus/100)
                        if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0 and fxx != 0 and v3_f_mode != 1:
                            datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                        if (Pi_Cam == 3 and v3_af == 1) and v3_f_speed != 0:
                            datastr += " --autofocus-speed " + v3_f_speeds[v3_f_speed]
                        if (Pi_Cam == 3 and v3_af == 1) and v3_f_range != 0:
                            datastr += " --autofocus-range " + v3_f_ranges[v3_f_range]
                        if Pi_Cam == 3 or Pi == 5:
                            datastr += " --hdr " + v3_hdrs_cli[v3_hdr]
                        datastr += " -p 0,0," + str(preview_width) + "," + str(preview_height)
                        # Ajouter le fichier de sortie (-o) APRES tous les autres paramètres
                        datastr += " -o " + vname
                        # ROI déjà ajouté plus haut (après brightness/contrast) pour respecter l'ordre de Test2
                        if show_cmds == 1:
                            print (datastr)

                        # Arrêter temporairement Picamera2 pour libérer la caméra
                        picam2_was_paused = pause_picamera2()

                        print(f"[DEBUG] Starting video recording: {vname}")
                        print(f"[DEBUG] Command: {datastr}")
                        print(f"[DEBUG] Pi={Pi}, codec={codecs[codec]}")

                        if Pi == 5 and codecs[codec] == 'mp4':
                            print("[DEBUG] Using os.system() for MP4 on Pi 5")
                            os.system(datastr)
                        else:
                            print("[DEBUG] Using subprocess.Popen()")
                            p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                        start_video = time.monotonic()
                        stop = 0
                        while (time.monotonic() - start_video < vlen or vlen == 0) and stop == 0:
                          if vlen != 0:
                              vlength = int(vlen - (time.monotonic()-start_video))
                          else:
                              vlength = int(time.monotonic()-start_video)
                          td = timedelta(seconds=vlength)
                          text(0,1,1,1,1,str(td),fv,0)
                          for event in pygame.event.get():
                              if (event.type == MOUSEBUTTONUP):
                                  mousex, mousey = event.pos
                                  # stop video recording
                                  if mousex > preview_width:
                                      button_row = int((mousey)/bh)
                                      if mousex > preview_width + (bw/2):
                                          button_pos = 1
                                      else:
                                          button_pos = 0
                                  if button_row == 1:
                                      if p is not None:
                                          os.killpg(p.pid, signal.SIGTERM)
                                      stop = 1

                    print(f"[DEBUG] Video recording stopped")
                    print(f"[DEBUG] Checking if file exists: {vname}")
                    print(f"[DEBUG] File exists: {os.path.exists(vname)}")
                    if os.path.exists(vname):
                        print(f"[DEBUG] File size: {os.path.getsize(vname)} bytes")

                    text(0,0,6,2,1,vname,int(fv*1.5),1)
                    time.sleep(1)

                    # Redémarrer Picamera2 si il avait été mis en pause
                    if picam2_was_paused:
                        text(0,0,2,2,1,"Redémarrage Picamera2...",int(fv*1.5),1)
                        pygame.display.update()
                        resume_picamera2()
                        time.sleep(0.5)

                    # Post-traitement pour Pi 5 avec MP4/H264 (correction timestamps)
                    if Pi == 5 and (codecs[codec] == 'mp4' or codecs[codec] == 'h264'):
                        text(0,0,2,2,1,"Post-traitement ffmpeg en cours...",int(fv*1.5),1)
                        pygame.display.update()

                        # Calculer le framerate utilisé
                        if mode != 0:
                            fps_used = fps
                        else:
                            speed7 = sspeed
                            speed7 = max(speed7, int((1/fps)*1000000))
                            fps_used = max(1, min(180, int(1000000/speed7)))

                        # Appeler la fonction de correction des timestamps
                        success = fix_video_timestamps(vname, fps_used, quality_preset="ultrafast")

                        if success:
                            text(0,0,1,2,1,"Vidéo corrigée avec succès",int(fv*1.5),1)
                        else:
                            text(0,0,3,2,1,"Avertissement: correction timestamps échouée",int(fv*1.5),1)

                        pygame.display.update()
                        time.sleep(2)

                    td = timedelta(seconds=vlen)
                    text(0,1,3,1,1,str(td),fv,0)
                    video = 0
                    menu = 0
                    Menu()
                    restart = 2
                                       
                elif button_row == 1 and event.button == 3:
                    # STREAM VIDEO
                    stream = 1
                    picam2_was_paused = False  # Initialiser pour éviter NameError
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                        # Attendre que le processus de preview se termine complètement
                        poll = p.poll()
                        while poll == None:
                            poll = p.poll()
                            time.sleep(0.1)

                    button(0,1,1,3)
                    text(0,1,2,0,1,"           STOP ",ft,0)
                    text(0,0,6,2,1,"Please Wait, streaming video ...",int(fv*1.7),1)
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%y%m%d%H%M%S")
                    vname =  vid_dir + str(timestamp) + "." + codecs2[codec]
                    if lver != "bookwo" and lver != "trixie":
                        datastr = "libcamera-vid "
                    else:
                        datastr = "rpicam-vid "
                    datastr += "--camera " + str(camera) + " -t " + str(vlen * 1000)
                    # Ajouter framerate AVANT les options de sortie
                    if mode != 0:
                        datastr += " --framerate " + str(fps)
                    else:
                        speed7 = sspeed
                        speed7 = max(speed7,int((1/fps)*1000000))
                        datastr += " --framerate " + str(max(1, min(180, int(1000000/speed7))))
                    # Ajouter les options de sortie APRES framerate
                    if stream_type == 0:
                        datastr += " --inline --listen -o tcp://0.0.0.0:" + str(stream_port)
                    elif stream_type == 1:
                        datastr += " --inline -o udp://" + udp_ip_addr + ":" + str(stream_port)
                    prof = h264profiles[profile].split(" ")
                    #datastr += " --profile " + str(prof[0]) + " --level " + str(prof[1])
                    datastr += " --level " + str(prof[1])
                    if vpreview == 0:
                        datastr += " -n "
                    datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)

                    # IMPORTANT: Pour un vrai ROI (méthode Test2), ajouter --mode et --roi AVANT --width/--height
                    if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop natif (pas de ROI)
                        sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                        if sensor_mode:
                            # Utiliser directement le mode sensor natif sans ROI
                            datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                    elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel (zoom 3 désactivé)
                        # Déterminer la profondeur de bits selon la caméra
                        if Pi_Cam == 4:  # Pi HQ
                            sensor_bits = 12
                        else:
                            sensor_bits = 10

                        zws = int(igw * zfs[zoom])
                        zhs = int(igh * zfs[zoom])
                        zxo = ((igw-zws)/2)/igw
                        zyo = ((igh-zhs)/2)/igh
                        datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits) + "  --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                    # Supprimé: ancien zoom manuel (zoom == 5)
                    elif False and zoom == 5:
                        # Déterminer la profondeur de bits selon la caméra
                        if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                            sensor_bits = 12
                        else:
                            sensor_bits = 10

                        zxo = ((igw/2)-(preview_width/2))/igw
                        if igw/igh > 1.5:
                            zyo = ((igh/2)-((preview_height * .75)/2))/igh
                        else:
                            zyo = ((igh/2)-(preview_height/2))/igh

                        datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits)
                        if igw/igh > 1.5:
                            datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width / .75)/igw) + "," + str(preview_height/igh)
                        else:
                            datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)

                    # Résolution et dimensions - SEULEMENT si ROI non appliqué
                    # IMPORTANT: Quand --roi est utilisé, NE PAS spécifier --width/--height
                    # Supprimé: ancien zoom manuel (zoom == 5)
                    if False and zoom == 5:
                        datastr += " --width " + str(preview_width) + " --height " + str(preview_height)
                    elif Pi_Cam == 10 and zoom > 0:  # IMX585: ajouter --width et --height du mode sensor
                        sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                        if sensor_mode:
                            datastr += f" --width {sensor_mode[0]} --height {sensor_mode[1]}"
                    elif zoom > 0 and zoom <= 5:  # Autres caméras - Zoom fixe avec ROI (zoom 3 désactivé)
                        # NE RIEN FAIRE : le ROI a déjà défini la taille de sortie
                        pass
                    elif Pi_Cam == 4 and vwidth == 2028:
                        datastr += " --mode 2028:1520:12"
                    elif Pi_Cam == 3 and vwidth == 2304 and codec == 0:
                        datastr += " --mode 2304:1296:10 --width 2304 --height 1296"
                    elif Pi_Cam == 3 and vwidth == 2028 and codec == 0:
                        datastr += " --mode 2028:1520:10 --width 2028 --height 1520"
                    elif Pi_Cam == 10:
                        # Déterminer le mode sensor basé sur vwidth/vheight ou zoom
                        if zoom > 0:
                            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                        else:
                            # Pas de zoom: choisir mode basé sur résolution demandée
                            # Mapping résolution → mode
                            if vwidth == 3856 and vheight == 2180:
                                sensor_mode = (3856, 2180)  # Mode 1 native
                            elif vwidth == 2880 and vheight == 2160:
                                sensor_mode = (2880, 2160)  # Mode 2 crop
                            elif vwidth == 1920 and vheight == 1080:
                                sensor_mode = (1920, 1080)  # Mode 3 crop
                            elif vwidth == 1280 and vheight == 720:
                                sensor_mode = (1280, 720)   # Mode 4 crop
                            elif vwidth == 800 and vheight == 600:
                                sensor_mode = (800, 600)    # Mode 5 crop
                            else:
                                # Fallback: utiliser mode binning
                                sensor_mode = (1928, 1090)  # Mode 0 binning

                        if sensor_mode:
                            datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                        datastr += f" --width {vwidth} --height {vheight}"
                    else:
                        datastr += " --width " + str(vwidth) + " --height " + str(vheight)
                    if mode == 0:
                        datastr += " --shutter " + str(sspeed)
                    else:
                        datastr += " --exposure " + modes[mode]
                    datastr += " --gain " + str(gain)
                    if ev != 0:
                        datastr += " --ev " + str(ev)
                    if awb == 0:
                        datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                    else:
                        datastr += " --awb " + awbs[awb]
                    datastr += " --metering " + meters[meter]
                    datastr += " --saturation " + str(saturation/10)
                    datastr += " --sharpness " + str(sharpness/10)
                    datastr += " --denoise "    + denoises[denoise]
                    if vflip == 1:
                        datastr += " --vflip"
                    if hflip == 1:
                        datastr += " --hflip"
                    if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                    if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"
                    if Pi_Cam == 4 and scientific == 1:
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                    if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6) ) or Pi_Cam == 8:
                        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                        if v3_f_mode == 1:
                            if Pi_Cam == 3:
                                datastr += " --lens-position " + str(v3_focus/100)
                            if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                datastr += " --lens-position " + str(focus/100)
                    if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6) ) or Pi_Cam == 8)  and zoom == 0 and fxx != 0 and v3_f_mode != 1:
                        datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                    if (Pi_Cam == 3 and v3_af == 1) and v3_f_speed != 0:
                        datastr += " --autofocus-speed " + v3_f_speeds[v3_f_speed]
                    if (Pi_Cam == 3 and v3_af == 1) and v3_f_range != 0:
                        datastr += " --autofocus-range " + v3_f_ranges[v3_f_range]
                    if Pi_Cam == 3 or Pi == 5:
                        datastr += " --hdr " + v3_hdrs_cli[v3_hdr]
                    datastr += " -p 0,0," + str(preview_width) + "," + str(preview_height)
                    # ROI déjà ajouté plus haut (après brightness/contrast) pour respecter l'ordre de Test2
                    if stream_type == 2:
                        data = "#rtp{sdp=rtsp://:" + str(stream_port) + "/stream1}"
                        datastr += " --inline -o | cvlc stream:///dev/stdin --sout '" + data + "' :demux=h264" ###
                    if show_cmds == 1:
                        print (datastr)

                    # Arrêter temporairement Picamera2 pour libérer la caméra
                    picam2_was_paused = pause_picamera2()

                    p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                    start_video = time.monotonic()
                    stop = 0
                    while (time.monotonic() - start_video < vlen or vlen == 0) and stop == 0:
                        if vlen != 0:
                            vlength = int(vlen - (time.monotonic()-start_video))
                        else:
                            vlength = int(time.monotonic()-start_video)
                        td = timedelta(seconds=vlength)
                        text(0,1,1,1,1,str(td),fv,0)
                        for event in pygame.event.get():
                            if (event.type == MOUSEBUTTONUP):
                                mousex, mousey = event.pos
                                # stop video streaming
                                if mousex > preview_width:
                                    button_row = int((mousey)/bh)
                                    if mousex > preview_width + (bw/2):
                                        button_pos = 1
                                    else:
                                        button_pos = 0
                                if button_row == 1:
                                   if p is not None:
                                       os.killpg(p.pid, signal.SIGTERM)
                                   stop = 1

                    # Redémarrer Picamera2 si il avait été mis en pause
                    if picam2_was_paused:
                        resume_picamera2()

                    td = timedelta(seconds=vlen)
                    text(0,1,3,1,1,str(td),fv,0)
                    stream = 0
                    menu = 0
                    Menu()
                    restart = 2
                        
                elif button_row == 2:
                    # TAKE TIMELAPSE
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                    # Fermer Picamera2 si actif pour libérer la caméra pour rpicam-still
                    pause_picamera2()
                    restart = 1
                    timelapse = 1
                    button(0,2,1,4)
                    text(0,2,2,0,1,"           STOP",ft,0)
                    tcount = 0
                      
                    if tinterval > 0 and mode != 0: # normal mode
                        text(0,0,6,2,1,"Please Wait, taking Timelapse ...",int(fv*1.7),1)
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%y%m%d%H%M%S")
                        count = 0
                        # Forcer .jpg pour Allsky
                        if allsky_mode > 0:
                            fname =  pic_dir + str(timestamp) + '_%04d.jpg'
                        else:
                            fname =  pic_dir + str(timestamp) + '_%04d.' + extns2[extn]
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-still"
                        else:
                            datastr = "rpicam-still"
                        if extns[extn] != 'raw':
                            datastr += " --camera " + str(camera) + " -e " + extns[extn] + " -s -t 0 -o " + fname
                            datastr += " -n"
                        else:
                            datastr += " --camera " + str(camera) + " -r -s -t 0 -o " + fname 
                            datastr += " -n"
                        datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
                        if mode == 0:
                            datastr += " --shutter " + str(sspeed)
                        else:
                            datastr += " --exposure " + modes[mode]
                        if ev != 0:
                            datastr += " --ev " + str(ev)
                        if sspeed > 1000000 and mode == 0:
                            datastr += " --gain " + str(gain) + " --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --gain " + str(gain)
                            if awb == 0:
                                datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                            else:
                                datastr += " --awb " + awbs[awb]
                        datastr += " --metering " + meters[meter]
                        datastr += " --saturation " + str(saturation/10)
                        datastr += " --sharpness " + str(sharpness/10)
                        datastr += " --quality " + str(quality)
                        datastr += " --denoise "    + denoises[denoise]
                        if vflip == 1:
                            datastr += " --vflip"
                        if hflip == 1:
                            datastr += " --hflip"
                        if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                        if Pi_Cam == 4 and scientific == 1:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                            datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                            if v3_f_mode == 1:
                                if Pi_Cam == 3 and v3_af == 1:
                                    datastr += " --lens-position " + str(v3_focus/100)
                                if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                    datastr += " --lens-position " + str(focus/100)
                        elif (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0 and fxz == 1:
                            datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode] + " --autofocus-on-capture"
                        if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0:
                            datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                        if Pi_Cam == 3 or Pi == 5:
                            datastr += " --hdr " + v3_hdrs_cli[v3_hdr]
                        if (Pi_Cam == 6 or Pi_Cam == 8) and mode == 0 and button_pos == 3:
                            datastr += " --width 4624 --height 3472 " # 16MP superpixel mode for higher light sensitivity
                        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
                            if Pi != 5 and lo_res == 1:
                                datastr += " --width 4624 --height 3472"
                            elif Pi_Cam == 6:
                                datastr += " --width 9152 --height 6944"
                            elif Pi_Cam == 8:
                                datastr += " --width 9248 --height 6944"
                        # Zoom fixe 1x à 6x
                        if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop natif (pas de ROI)
                            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                            if sensor_mode:
                                # Utiliser directement le mode sensor natif sans ROI
                                datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                                datastr += f" --width {sensor_mode[0]} --height {sensor_mode[1]}"
                        elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel (zoom 3 désactivé)
                            # Arrondir à un nombre PAIR pour compatibilité formats vidéo
                            zws = int(igw * zfs[zoom])
                            zhs = int(igh * zfs[zoom])
                            if zws % 2 != 0:
                                zws -= 1  # Forcer pair
                            if zhs % 2 != 0:
                                zhs -= 1  # Forcer pair
                            zxo = ((igw-zws)/2)/igw
                            zyo = ((igh-zhs)/2)/igh
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                            datastr += " --width " + str(zws) + " --height " + str(zhs)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        if False and zoom == 5:
                            zxo = ((igw/2)-(preview_width/2))/igw
                            if igw/igh > 1.5:
                                zyo = ((igh/2)-((preview_height * .75)/2))/igh
                            else:
                                zyo = ((igh/2)-(preview_height/2))/igh
                            if igw/igh > 1.5:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                            else:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                        p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                        if show_cmds == 1:
                            print (datastr)
                        start_timelapse = time.monotonic()
                        start2 = time.monotonic()
                        stop = 0
                        pics3 = []
                        count = 0
                        old_count = 0

                        # ALLSKY: Initialiser contrôleur Mean Target si mode Auto-Gain
                        allsky_controller = None
                        original_gain = gain  # Sauvegarder gain initial
                        if allsky_mode == 2:
                            allsky_controller = AllskyMeanController(
                                mean_target=allsky_mean_target / 100.0,
                                mean_threshold=allsky_mean_threshold / 100.0,
                                max_gain=allsky_max_gain
                            )
                            print(f"[Allsky] Auto-Gain activé: target={allsky_mean_target/100.0:.2f}, threshold={allsky_mean_threshold/100.0:.2f}, max_gain={allsky_max_gain}")

                        while count < tshots and stop == 0:
                            if time.monotonic() - start2 >= tinterval:
                                if lver != "bookwo" and lver != "trixie":
                                    os.system('pkill -SIGUSR1 libcamera-still')
                                else:
                                    os.system('pkill -SIGUSR1 rpicam-still')
                                start2 = time.monotonic()
                                text(0,0,6,2,1,"Please Wait, taking Timelapse ..."  + " " + str(count+1),int(fv*1.7),1)
                                show = 0
                                while count == old_count:
                                    time.sleep(0.1)
                                    pics3 = glob.glob(pic_dir + "*.*")
                                    counts = []
                                    for xu in range(0,len(pics3)):
                                        ww = pics3[xu].split("/")
                                        if ww[-1][0:12] == timestamp:
                                            counts.append(pics3[xu])
                                    count = len(counts)
                                    counts.sort()
                                    for event in pygame.event.get():
                                        if event.type == QUIT:
                                            # Ignorer QUIT pendant timelapse (écran en veille, etc.)
                                            pass
                                        elif (event.type == MOUSEBUTTONUP):
                                            mousex, mousey = event.pos
                                            # stop timelapse
                                            if mousex > preview_width:
                                                button_row = int((mousey)/bh)
                                            if button_row == 2:
                                                if p is not None:
                                                    os.killpg(p.pid, signal.SIGTERM)
                                                text(0,2,3,1,1,str(tshots),fv,12)
                                                stop = 1
                                                count = tshots
                                                                                        
                                old_count = count

                                # ALLSKY: Appliquer stretch si activé
                                if allsky_mode > 0 and count > 0 and allsky_apply_stretch == 1:
                                    latest_jpeg = counts[-1]
                                    apply_stretch_to_jpeg(latest_jpeg, stretch_preset)

                                # ALLSKY: Ajuster gain automatiquement
                                if allsky_controller is not None and count > 0:
                                    latest_jpeg = counts[-1]
                                    measured_mean = allsky_controller.calculate_mean(latest_jpeg)
                                    if measured_mean is not None:
                                        new_gain = allsky_controller.update(gain, measured_mean)
                                        if new_gain != gain:
                                            print(f"[Allsky] Auto-Gain: mean={measured_mean:.3f}, {gain} -> {new_gain}")
                                            gain = new_gain
                                            # Mettre à jour la commande libcamera-still avec le nouveau gain
                                            # Note: Le nouveau gain sera utilisé pour la PROCHAINE capture

                                text(0,2,1,1,1,str(tshots - count),fv,0)
                                tdur = tinterval * (tshots - count)
                                td = timedelta(seconds=tdur)
                            time.sleep(0.1)
                            if buttonSTR.is_pressed: # 
                                type = pygame.MOUSEBUTTONUP
                                if str_cap == 2:
                                    click_event = pygame.event.Event(type, {"button": 3, "pos": (0,0)})
                                else:
                                    click_event = pygame.event.Event(type, {"button": 1, "pos": (0,0)})
                                pygame.event.post(click_event)
                            for event in pygame.event.get():
                                if event.type == QUIT:
                                    # Ignorer QUIT pendant timelapse (écran en veille, etc.)
                                    pass
                                elif (event.type == MOUSEBUTTONUP):
                                    mousex, mousey = event.pos
                                    # stop timelapse or capture STILL
                                    if mousex > preview_width:
                                        button_row = int((mousey)/bh)
                                    if button_row == 2:
                                        if p is not None:
                                            os.killpg(p.pid, signal.SIGTERM)
                                        text(0,2,3,1,1,str(tshots),fv,12)
                                        stop = 1
                                        count = tshots
                                    if button_row == 0:
                                        if lver != "bookwo" and lver != "trixie":
                                            os.system('pkill -SIGUSR1 libcamera-still')
                                        else:
                                            os.system('pkill -SIGUSR1 rpicam-still')
                                        text(0,0,3,0,1,"CAPTURE",ft,7)
                        if lver != "bookwo" and lver != "trixie":
                            os.system('pkill -SIGUSR2 libcamera-still')
                        else:
                            os.system('pkill -SIGUSR2 rpicam-still')

                        # ALLSKY: Assembler vidéo si mode activé
                        if allsky_mode > 0 and stop == 0:
                            video_filename = pic_dir + timestamp + "_allsky.mp4"
                            text(0,0,6,2,1,"Assemblage vidéo Allsky...", int(fv*1.7), 1)
                            pygame.display.update()

                            success = assemble_allsky_video(pic_dir, timestamp, allsky_video_fps, video_filename)

                            if success:
                                print(f"[Allsky] Vidéo créée : {video_filename}")
                                text(0,0,6,2,1,f"Vidéo créée: {video_filename}", int(fv*1.5), 1)
                                pygame.display.update()
                                time.sleep(2)

                                # Nettoyer JPEGs si activé
                                if allsky_cleanup_jpegs == 1:
                                    text(0,0,6,2,1,"Nettoyage JPEGs...", int(fv*1.7), 1)
                                    pygame.display.update()
                                    for jpeg_path in counts:
                                        try:
                                            os.remove(jpeg_path)
                                        except Exception as e:
                                            print(f"[Allsky] Erreur suppression {jpeg_path}: {e}")
                                    print(f"[Allsky] {len(counts)} JPEGs supprimés")
                            else:
                                print("[Allsky] Échec assemblage - JPEGs conservés")
                                text(0,0,6,2,1,"Échec assemblage vidéo", int(fv*1.5), 1)
                                pygame.display.update()
                                time.sleep(2)

                        # ALLSKY: Restaurer gain initial
                        if allsky_controller is not None:
                            gain = original_gain
                            print(f"[Allsky] Gain restauré: {gain}")

                    elif tinterval > 0 and mode == 0: # manual mode
                        text(0,0,6,2,1,"Please Wait, taking Timelapse ...",int(fv*1.7),1)
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%y%m%d%H%M%S")
                        start2 = time.monotonic()
                        stop = 0
                        pics3 = []
                        count = 0
                        old_count = 0
                        trig = 1
                        p = None

                        # ALLSKY: Initialiser contrôleur Mean Target si mode Auto-Gain
                        allsky_controller = None
                        original_gain = gain  # Sauvegarder gain initial
                        if allsky_mode == 2:
                            allsky_controller = AllskyMeanController(
                                mean_target=allsky_mean_target / 100.0,
                                mean_threshold=allsky_mean_threshold / 100.0,
                                max_gain=allsky_max_gain
                            )
                            print(f"[Allsky] Auto-Gain activé: target={allsky_mean_target/100.0:.2f}, threshold={allsky_mean_threshold/100.0:.2f}, max_gain={allsky_max_gain}")

                        while count < tshots and stop == 0:
                            if time.monotonic() - start2 > tinterval:
                                start2 = time.monotonic()
                                if p is not None:
                                    poll = p.poll()
                                    while poll == None:
                                        poll = p.poll()
                                        time.sleep(0.1)
                                # Forcer .jpg pour Allsky (format %04d pour cohérence avec mode normal)
                                if allsky_mode > 0:
                                    fname =  pic_dir + str(timestamp) + "_%04d.jpg" % count
                                else:
                                    fname =  pic_dir + str(timestamp) + "_" + str(count) + "." + extns2[extn]
                                if lver != "bookwo" and lver != "trixie":
                                    datastr = "libcamera-still"
                                else:
                                    datastr = "rpicam-still"
                                if extns[extn] != 'raw':
                                    datastr += " --camera " + str(camera) + " -e " + extns[extn] + " -t " + str(timet) + " -o " + fname + " -n"
                                else:
                                    datastr += " --camera " + str(camera) + " -r -t 1000 -o " + fname + " -n " 
                                datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
                                datastr += " --shutter " + str(sspeed)
                                if ev != 0:
                                    datastr += " --ev " + str(ev)
                                if sspeed > 1000000 and mode == 0:
                                    datastr += " --gain " + str(gain) + " --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                                else:
                                    datastr += " --gain " + str(gain)
                                    if awb == 0:
                                        datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                                    else:
                                        datastr += " --awb " + awbs[awb]
                                datastr += " --metering " + meters[meter]
                                datastr += " --saturation " + str(saturation/10)
                                datastr += " --sharpness " + str(sharpness/10)
                                datastr += " --quality " + str(quality)
                                datastr += " --denoise "    + denoises[denoise]
                                if vflip == 1:
                                    datastr += " --vflip"
                                if hflip == 1:
                                    datastr += " --hflip"
                                if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                                    datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                                if Pi_Cam == 4 and scientific == 1:
                                    if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                        datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                                    if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                        datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                                if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                                    datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                                    if v3_f_mode == 1:
                                        if Pi_Cam == 3 and v3_af == 1:
                                            datastr += " --lens-position " + str(v3_focus/100)
                                        if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                            datastr += " --lens-position " + str(focus/100)
                                elif (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0 and fxz == 1:
                                    datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode] + " --autofocus-on-capture"
                                if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8)  and zoom == 0:
                                    datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                                if Pi_Cam == 3 or Pi == 5:
                                    datastr += " --hdr " + v3_hdrs_cli[v3_hdr]
                                if (Pi_Cam == 6 or Pi_Cam == 8) and mode == 0 and button_pos == 3:
                                    datastr += " --width 4624 --height 3472 " # 16MP superpixel mode for higher light sensitivity
                                elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
                                    if Pi != 5 and lo_res == 1:
                                        datastr += " --width 4624 --height 3472"
                                    elif Pi_Cam == 6:
                                        datastr += " --width 9152 --height 6944"
                                    elif Pi_Cam == 8:
                                        datastr += " --width 9248 --height 6944"
                                # Zoom fixe 1x à 6x
                                if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop natif (pas de ROI)
                                    sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                                    if sensor_mode:
                                        # Utiliser directement le mode sensor natif sans ROI
                                        datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                                elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel
                                    zws = int(igw * zfs[zoom])
                                    zhs = int(igh * zfs[zoom])
                                    zxo = ((igw-zws)/2)/igw
                                    zyo = ((igh-zhs)/2)/igh
                                    datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                                # Supprimé: ancien zoom manuel (zoom == 5)
                                if False and zoom == 5:
                                    zxo = ((igw/2)-(preview_width/2))/igw
                                    if igw/igh > 1.5:
                                        zyo = ((igh/2)-((preview_height * .75)/2))/igh
                                    else:
                                        zyo = ((igh/2)-(preview_height/2))/igh
                                    if igw/igh > 1.5:
                                        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                                    else:
                                        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                                if show_cmds == 1:
                                    print (datastr)
                                p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                                text(0,0,6,2,1,"Please Wait, taking Timelapse ..."  + " " + str(count+1),int(fv*1.7),1)
                                show = 0
                                while count == old_count:
                                    time.sleep(0.1)
                                    pics3 = glob.glob(pic_dir + "*.*")
                                    counts = []
                                    for xu in range(0,len(pics3)):
                                        ww = pics3[xu].split("/")
                                        if ww[-1][0:12] == timestamp:
                                            counts.append(pics3[xu])
                                    count = len(counts)
                                    counts.sort()
                                    if (extns2[extn] == 'jpg' or extns2[extn] == 'bmp' or extns2[extn] == 'png') and count > 0 and show == 0:
                                        image = pygame.image.load(counts[count-1])
                                        # Zoom manuel supprimé: simplifié la condition
                                        if (Pi_Cam != 3 and Pi_Cam != 10 and Pi_Cam != 15):
                                            catSurfacesmall = pygame.transform.scale(image, (preview_width,preview_height))
                                        else:
                                            catSurfacesmall = pygame.transform.scale(image, (preview_width,int(preview_height * 0.75)))
                                        windowSurfaceObj.blit(catSurfacesmall, (0, 0))
                                        text(0,0,6,2,1,counts[count-1],int(fv*1.5),1)
                                        pygame.display.update()
                                        show == 1
                                    for event in pygame.event.get():
                                        if event.type == QUIT:
                                            # Ignorer QUIT pendant timelapse (écran en veille, etc.)
                                            pass
                                        elif (event.type == MOUSEBUTTONUP):
                                            mousex, mousey = event.pos
                                            # stop timelapse
                                            if mousex > preview_width:
                                                button_row = int((mousey)/bh)
                                            if button_row == 2:
                                                if p is not None:
                                                    os.killpg(p.pid, signal.SIGTERM)
                                                text(0,2,3,1,1,str(tshots),fv,0)
                                                stop = 1
                                                count = tshots

                                old_count = count

                                # ALLSKY: Appliquer stretch si activé
                                if allsky_mode > 0 and count > 0 and allsky_apply_stretch == 1 and len(counts) > 0:
                                    latest_jpeg = counts[-1]
                                    apply_stretch_to_jpeg(latest_jpeg, stretch_preset)

                                # ALLSKY: Ajuster gain automatiquement
                                if allsky_controller is not None and count > 0 and len(counts) > 0:
                                    latest_jpeg = counts[-1]
                                    measured_mean = allsky_controller.calculate_mean(latest_jpeg)
                                    if measured_mean is not None:
                                        new_gain = allsky_controller.update(gain, measured_mean)
                                        if new_gain != gain:
                                            print(f"[Allsky] Auto-Gain: mean={measured_mean:.3f}, {gain} -> {new_gain}")
                                            gain = new_gain
                                            # Note: Le nouveau gain sera utilisé pour la PROCHAINE capture

                                text(0,2,1,1,1,str(tshots - count),fv,12)
                                tdur = tinterval * (tshots - count)
                                td = timedelta(seconds=tdur)
                            time.sleep(0.1)
                            for event in pygame.event.get():
                                if event.type == QUIT:
                                    # Ignorer QUIT pendant timelapse (écran en veille, etc.)
                                    pass
                                elif (event.type == MOUSEBUTTONUP):
                                    mousex, mousey = event.pos
                                    # stop timelapse
                                    if mousex > preview_width:
                                        button_row = int((mousey)/bh)
                                    if button_row == 2:
                                        if p is not None:
                                            os.killpg(p.pid, signal.SIGTERM)
                                        text(0,2,3,1,1,str(tshots),fv,0)
                                        stop = 1
                                        count = tshots

                        # ALLSKY: Assembler vidéo si mode activé
                        if allsky_mode > 0 and stop == 0:
                            video_filename = pic_dir + timestamp + "_allsky.mp4"
                            text(0,0,6,2,1,"Assemblage vidéo Allsky...", int(fv*1.7), 1)
                            pygame.display.update()

                            success = assemble_allsky_video(pic_dir, timestamp, allsky_video_fps, video_filename)

                            if success:
                                print(f"[Allsky] Vidéo créée : {video_filename}")
                                text(0,0,6,2,1,f"Vidéo créée: {video_filename}", int(fv*1.5), 1)
                                pygame.display.update()
                                time.sleep(2)

                                # Nettoyer JPEGs si activé
                                if allsky_cleanup_jpegs == 1:
                                    text(0,0,6,2,1,"Nettoyage JPEGs...", int(fv*1.7), 1)
                                    pygame.display.update()
                                    for jpeg_path in counts:
                                        try:
                                            os.remove(jpeg_path)
                                        except Exception as e:
                                            print(f"[Allsky] Erreur suppression {jpeg_path}: {e}")
                                    print(f"[Allsky] {len(counts)} JPEGs supprimés")
                            else:
                                print("[Allsky] Échec assemblage - JPEGs conservés")
                                text(0,0,6,2,1,"Échec assemblage vidéo", int(fv*1.5), 1)
                                pygame.display.update()
                                time.sleep(2)

                        # ALLSKY: Restaurer gain initial
                        if allsky_controller is not None:
                            gain = original_gain
                            print(f"[Allsky] Gain restauré: {gain}")

                    elif tinterval == 0:
                        if tduration == 0:
                            tduration = 1
                        text(0,0,6,2,1,"Please Wait, taking Timelapse ...",int(fv*1.7),1)
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%y%m%d%H%M%S")
                        fname =  pic_dir + str(timestamp) + '_%04d.' + extns2[extn]
                        if codecs2[codec] != 'raw':
                            if lver != "bookwo" and lver != "trixie":
                                datastr = "libcamera-vid"
                            else:
                                datastr = "rpicam-vid"
                            datastr += " --camera " + str(camera) + " -n --codec mjpeg -t " + str(tduration*1000) + " --segment 400"
                            # Ajouter le framerate AVANT les autres paramètres
                            if mode == 0:
                                # Permettre FPS < 1 pour longues expositions (0.01 fps min)
                                if sspeed > 0:
                                    calc_fps = max(min(1000000/sspeed, 180), 0.01)
                                else:
                                    calc_fps = 30
                                datastr += " --framerate " + str(calc_fps)
                            else:
                                datastr += " --framerate " + str(fps)
                        else:
                            fname =  pic_dir + str(timestamp) + '_%04d.' + codecs2[codec]
                            if lver != "bookwo" and lver != "trixie":
                                datastr = "libcamera-raw"
                            else:
                                datastr = "rpicam-raw"
                            datastr += " --camera " + str(camera) + " -n -t " + str(tduration*1000) + " --segment 400"
                            # Ajouter le framerate pour raw aussi
                            if mode == 0:
                                # Permettre FPS < 1 pour longues expositions (0.01 fps min)
                                if sspeed > 0:
                                    calc_fps = max(min(1000000/sspeed, 180), 0.01)
                                else:
                                    calc_fps = 30
                                datastr += " --framerate " + str(calc_fps)
                            else:
                                datastr += " --framerate " + str(fps)
                        if zoom > 0:
                            if igw/igh > 1.5:
                                datastr += " --width " + str(int(preview_width)) + " --height " + str(int(preview_height * .75))
                            else:
                                datastr += " --width " + str(preview_width) + " --height " + str(preview_height)
                        else:
                            datastr += " --width " + str(vwidth) + " --height " + str(vheight)
                        datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
                        if mode == 0:
                            datastr += " --shutter " + str(sspeed)
                        else:
                            datastr += " --exposure " + str(modes[mode])
                        if ev != 0:
                            datastr += " --ev " + str(ev)
                        if sspeed > 5000000 and mode == 0 and (Pi_Cam < 5 or Pi_Cam == 7):
                            datastr += " --gain 1 --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --gain " + str(gain)
                            if awb == 0:
                                datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                            else:
                                datastr += " --awb " + awbs[awb]
                        datastr += " --metering "   + meters[meter]
                        datastr += " --saturation " + str(saturation/10)
                        datastr += " --sharpness "  + str(sharpness/10)
                        datastr += " --denoise "    + denoises[denoise]
                        if vflip == 1:
                            datastr += " --vflip"
                        if hflip == 1:
                            datastr += " --hflip"
                        if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                        if Pi_Cam == 4 and scientific == 1:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                            datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                            if v3_f_mode == 1:
                                if Pi_Cam == 3:
                                    datastr += " --lens-position " + str(v3_focus/100)
                                if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                    datastr += " --lens-position " + str(focus/100)
                        if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6) ) or Pi_Cam == 8) and zoom == 0:
                            datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                        if Pi_Cam == 3 or Pi == 5:
                            datastr += " --hdr " + v3_hdrs_cli[v3_hdr]
                        # Zoom fixe 1x à 6x
                        if Pi_Cam == 10 and zoom > 0:  # IMX585 - Hardware crop natif (pas de ROI)
                            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                            if sensor_mode:
                                # Utiliser directement le mode sensor natif sans ROI
                                datastr += f" --mode {sensor_mode[0]}:{sensor_mode[1]}:12"
                        elif zoom > 0 and zoom <= 5:  # Autres caméras - ROI logiciel (zoom 3 désactivé)
                            # Arrondir à un nombre PAIR pour compatibilité formats vidéo
                            zws = int(igw * zfs[zoom])
                            zhs = int(igh * zfs[zoom])
                            if zws % 2 != 0:
                                zws -= 1  # Forcer pair
                            if zhs % 2 != 0:
                                zhs -= 1  # Forcer pair
                            zxo = ((igw-zws)/2)/igw
                            zyo = ((igh-zhs)/2)/igh
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        if False and zoom == 5:
                            zxo = ((igw/2)-(preview_width/2))/igw
                            if igw/igh > 1.5:
                                zyo = ((igh/2)-((preview_height * .75)/2))/igh
                            else:
                                zyo = ((igh/2)-(preview_height/2))/igh
                            if igw/igh > 1.5:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width / .75)/igw) + "," + str(preview_height/igh)
                            else:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                        # Ajouter le fichier de sortie à la fin
                        datastr += " -o " + fname
                        if show_cmds == 1:
                            print (datastr)
                        p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                        start_timelapse = time.monotonic()
                        stop = 0
                        while time.monotonic() - start_timelapse < tduration+1 and stop == 0:
                            tdur = int(tduration - (time.monotonic() - start_timelapse))
                            td = timedelta(seconds=tdur)
                            text(0,2,1,1,1,str(td),fv,0)
                        # Attendre que le processus rpicam-vid se termine complètement
                        if p is not None:
                            print("[DEBUG] Waiting for rpicam-vid to finish...")
                            p.wait()
                            print("[DEBUG] rpicam-vid finished, waiting for camera release...")
                            time.sleep(2.0)  # Délai supplémentaire pour libération complète de la caméra
                    # Redémarrer Picamera2 si nécessaire après le timelapse
                    resume_picamera2()
                    timelapse = 0
                    menu = 0
                    Menu()
                    restart = 2 
                        
                elif button_row == 3:
                    # LIVE STACK - Active/désactive le mode Live Stacking
                    # Désactiver Lucky Stack si actif (mutuellement exclusif)
                    if luckystack_active:
                        luckystack_active = False
                        if luckystack is not None:
                            luckystack.stop()

                    if not livestack_active:
                        # Activer Live Stack
                        livestack_active = True

                        # *** IMPORTANT: Reconfigurer caméra pour mode RAW ***
                        # preview() vérifiera livestack_active et configurera le capture_thread en mode 'raw'
                        if raw_format >= 2:
                            print(f"[LIVESTACK] Mode RAW - reconfiguration caméra...")
                            kill_preview_process()
                            preview()  # Reconfigure avec stream RAW et capture_thread en mode 'raw'
                            print(f"[LIVESTACK] Caméra reconfigurée pour capture RAW")

                        # Réinitialiser les compteurs de session Live Stack
                        if hasattr(pygame, '_livestack_last_saved'):
                            delattr(pygame, '_livestack_last_saved')
                        if hasattr(pygame, '_livestack_stretch_info_shown'):
                            delattr(pygame, '_livestack_stretch_info_shown')
                        if hasattr(pygame, '_livestack_stretch_applied_shown'):
                            delattr(pygame, '_livestack_stretch_applied_shown')
                        if hasattr(pygame, '_livestack_no_stretch_shown'):
                            delattr(pygame, '_livestack_no_stretch_shown')
                        # Réinitialiser les flags de validation RAW
                        if hasattr(pygame, '_raw_format_mismatch_warning'):
                            delattr(pygame, '_raw_format_mismatch_warning')
                        if hasattr(pygame, '_raw_resolution_mismatch_warning'):
                            delattr(pygame, '_raw_resolution_mismatch_warning')
                        if hasattr(pygame, '_stacking_mode_shown'):
                            delattr(pygame, '_stacking_mode_shown')

                        # Activer le mode stretch pour affichage fullscreen
                        stretch_mode = 1

                        # Passer en mode fullscreen (comme pour STRETCH)
                        display_modes = pygame.display.list_modes()
                        if display_modes and display_modes != -1:
                            max_width, max_height = display_modes[0]
                        else:
                            screen_info = pygame.display.Info()
                            max_width, max_height = screen_info.current_w, screen_info.current_h
                        windowSurfaceObj = pygame.display.set_mode((max_width, max_height), pygame.FULLSCREEN, 24)

                        # Créer l'instance livestack si nécessaire
                        if livestack is None:
                            # Récupérer les paramètres actuels de la caméra
                            camera_params = {
                                'exposure': sspeed,  # Déjà en microsecondes
                                'gain': gain,
                                'red': red / 10,  # Divisé par 10 pour ColourGains
                                'blue': blue / 10,
                                'raw_format': raw_formats[raw_format]  # YUV420/XRGB8888/SRGGB12/SRGGB16
                            }
                            # Utiliser le module avancé
                            livestack = create_advanced_livestack_session(camera_params)

                            # Configurer avec paramètres utilisateur
                            livestack.configure(
                                # Stacking method
                                stacking_method=['mean', 'median', 'kappa_sigma', 'winsorized', 'weighted'][ls_stack_method],
                                kappa=ls_stack_kappa / 10.0,
                                iterations=ls_stack_iterations,

                                # Alignement
                                alignment_mode=ls_alignment_modes[ls_alignment_mode],

                                # Contrôle qualité
                                enable_qc=bool(ls_enable_qc),
                                max_fwhm=ls_max_fwhm / 10.0 if ls_max_fwhm > 0 else 999.0,
                                min_sharpness=ls_min_sharpness / 1000.0 if ls_min_sharpness > 0 else 0.0,
                                max_drift=float(ls_max_drift) if ls_max_drift > 0 else 999999.0,
                                min_stars=int(ls_min_stars),  # 0 = désactivé pour microscopie

                                # Planétaire (optionnel)
                                planetary_enable=bool(ls_planetary_enable),
                                planetary_mode=ls_planetary_mode,
                                planetary_disk_min=ls_planetary_disk_min,
                                planetary_disk_max=ls_planetary_disk_max,
                                planetary_disk_threshold=ls_planetary_threshold,
                                planetary_disk_margin=ls_planetary_margin,
                                planetary_disk_ellipse=bool(ls_planetary_ellipse),
                                planetary_window=planetary_windows[ls_planetary_window],
                                planetary_upsample=ls_planetary_upsample,
                                planetary_highpass=bool(ls_planetary_highpass),
                                planetary_roi_center=bool(ls_planetary_roi_center),
                                planetary_corr=ls_planetary_corr / 100.0,
                                planetary_max_shift=float(ls_planetary_max_shift),

                                # Lucky Imaging (désactivé en mode LIVE STACK)
                                lucky_enable=False,
                                lucky_buffer_size=ls_lucky_buffer,
                                lucky_keep_percent=float(ls_lucky_keep),
                                lucky_score_method=['laplacian', 'gradient', 'sobel', 'tenengrad'][ls_lucky_score],
                                lucky_stack_method=['mean', 'median', 'sigma_clip'][ls_lucky_stack],
                                lucky_align_enabled=bool(ls_lucky_align),
                                lucky_score_roi_percent=float(ls_lucky_roi),

                                # PNG avec stretch selon paramètres interface
                                png_stretch=['off', 'ghs', 'asinh'][stretch_preset],
                                png_factor=stretch_factor / 10.0,
                                # Pour GHS: pas de normalisation par percentiles (clip 0-100 = désactivé)
                                # Pour Arcsinh: normalisation active (stretch_p_low/high)
                                png_clip_low=0.0 if stretch_preset == 1 else stretch_p_low / 10.0,
                                png_clip_high=100.0 if stretch_preset == 1 else stretch_p_high / 100.0,
                                ghs_D=ghs_D / 10.0,
                                ghs_b=ghs_b / 10.0,
                                ghs_SP=ghs_SP / 100.0,
                                ghs_LP=ghs_LP / 100.0,
                                ghs_HP=ghs_HP / 100.0,
                                preview_refresh=ls_preview_refresh,
                                save_dng="none",

                                # ISP (Image Signal Processor)
                            )

                        # Debug ISP avant passage à configure (EN DEHORS du if pour être toujours exécuté)
                        print(f"[DEBUG AVANT CONFIGURE] isp_enable variable = {isp_enable} (type: {type(isp_enable)})")
                        print(f"[DEBUG AVANT CONFIGURE] isp_config_path variable = {isp_config_path}")
                        print(f"[DEBUG AVANT CONFIGURE] raw_format variable = {raw_format} ({raw_formats[raw_format]})")

                        # === NOUVELLE APPROCHE RAW: ISP/Stretch appliqué EXTERNE pour cohérence preview/stacking ===
                        # Pour RAW (raw_format >= 2): libastrostack empile les frames BRUTES
                        # L'ISP + stretch est appliqué dans RPiCamera2.py avec apply_isp_to_preview() + astro_stretch()
                        # Ceci garantit que le preview et le résultat stacké sont IDENTIQUES
                        #
                        # Pour RGB/YUV (raw_format < 2): pas de traitement ISP (déjà fait par ISP matériel)

                        is_raw_mode = (raw_format >= 2)

                        if is_raw_mode:
                            print(f"[LIVESTACK RAW] Mode RAW détecté - ISP/stretch seront appliqués EXTERNE")
                            print(f"  → libastrostack: stacking brut uniquement (isp_enable=False, stretch=none)")
                            print(f"  → RPiCamera2: appliquera apply_isp_to_preview() + astro_stretch()")
                        else:
                            if isp_enable == 1:
                                print(f"[INFO] ISP logiciel ignoré pour {raw_formats[raw_format]} (déjà traité par ISP matériel)")

                        # Passer les paramètres à libastrostack
                        # Pour RAW: désactiver ISP et stretch interne (traitement externe)
                        # Pour RGB/YUV: pas d'ISP software nécessaire
                        video_format_map = {0: 'yuv420', 1: 'xrgb8888', 2: 'raw12', 3: 'raw16'}
                        livestack.configure(
                            isp_enable=False,  # ISP externe pour RAW, pas nécessaire pour RGB/YUV
                            isp_config_path=None,
                            video_format=video_format_map.get(raw_format, 'yuv420'),
                            png_stretch='off' if is_raw_mode else 'ghs',  # Stretch externe pour RAW
                            # Paramètres GHS pour mode RGB/YUV (libastrostack les applique)
                            ghs_D=ghs_D / 10.0,
                            ghs_b=ghs_b / 10.0,
                            ghs_SP=ghs_SP / 100.0,
                            ghs_LP=ghs_LP / 100.0,
                            ghs_HP=ghs_HP / 100.0,
                        )

                        # Mettre à jour le format RAW actuel avant de démarrer
                        livestack.camera_params['raw_format'] = raw_formats[raw_format]

                        # Réinitialiser les compteurs avant de redémarrer
                        livestack.reset()

                        # Démarrer la session
                        livestack.start()

                        # Pour RAW mode, l'ISP/stretch sera appliqué externe (dans le code d'affichage)
                        # Plus besoin de apply_isp_to_session() car libastrostack n'a pas d'ISP actif

                        # Réinitialiser compteur de sauvegarde PNG
                        pygame._livestack_last_saved = 0

                        # Afficher le mode activé
                        qc_status = "ON" if ls_enable_qc else "OFF"
                        print(f"[LIVESTACK] Mode activé (QC: {qc_status})")
                    else:
                        # Désactiver Live Stack
                        livestack_active = False
                        stretch_mode = 0  # Désactiver aussi le stretch

                        if livestack is not None:
                            # Sauvegarder le résultat final avant de stopper (si activé)
                            if ls_save_final == 1:
                                try:
                                    # Mode RAW: utiliser traitement externe pour cohérence avec preview
                                    if raw_format >= 2:
                                        save_with_external_processing(livestack, raw_format_name=raw_formats[raw_format])
                                    else:
                                        # Mode RGB/YUV: utiliser libastrostack normalement
                                        livestack.save(raw_format_name=raw_formats[raw_format])
                                    print("[LIVESTACK] Image finale sauvegardée")
                                except Exception as e:
                                    print(f"[LIVESTACK] Erreur sauvegarde: {e}")
                            else:
                                print("[LIVESTACK] Sauvegarde finale désactivée (ls_save_final=0)")
                            livestack.stop()

                        # Restaurer le mode d'affichage normal (avec l'interface)
                        if frame == 1:
                            if fullscreen == 1:
                                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                            else:
                                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
                        else:
                            windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)

                        # Effacer l'écran et redessiner le menu
                        windowSurfaceObj.fill((0, 0, 0))
                        Menu()
                        pygame.display.update()
                        print("[LIVESTACK] Mode désactivé")

                        # *** Reconfigurer caméra pour revenir au mode ISP après le RAW ***
                        if raw_format >= 2 and not luckystack_active:
                            print(f"[LIVESTACK] Retour mode ISP - reconfiguration caméra...")
                            kill_preview_process()
                            preview()
                            print(f"[LIVESTACK] Caméra reconfigurée en mode normal")

                elif button_row == 4:
                    # LUCKY STACK - Active/désactive le mode Lucky Imaging
                    # Désactiver Live Stack si actif (mutuellement exclusif)
                    if livestack_active:
                        livestack_active = False
                        if livestack is not None:
                            livestack.stop()

                    if not luckystack_active:
                        # Vérifier compatibilité RAW pour IMX585
                        if Pi_Cam == 10 and raw_format >= 2:  # Mode RAW Bayer
                            sensor_mode = get_imx585_sensor_mode(zoom, use_native_sensor_mode == 1)
                            if sensor_mode not in imx585_validated_raw_modes:
                                print(f"\n⚠️  AVERTISSEMENT: Résolution {sensor_mode} non validée en mode RAW")
                                print(f"  Résolutions RAW validées:")
                                for res, desc in imx585_validated_raw_modes.items():
                                    print(f"    • {res[0]}x{res[1]} - {desc}")
                                print(f"  Pour de meilleurs résultats, utilisez:")
                                print(f"    - Zoom 0 (fullframe) ou Zoom 2 (1920x1080)")
                                print(f"    - OU passez en mode XRGB8888 ISP (raw_format=1)")
                                print()

                        # Activer Lucky Stack
                        luckystack_active = True

                        # *** IMPORTANT: Reconfigurer caméra pour mode RAW ***
                        # preview() vérifiera luckystack_active et configurera le capture_thread en mode 'raw'
                        if raw_format >= 2:
                            print(f"[LUCKYSTACK] Mode RAW - reconfiguration caméra...")
                            kill_preview_process()
                            preview()  # Reconfigure avec stream RAW et capture_thread en mode 'raw'
                            print(f"[LUCKYSTACK] Caméra reconfigurée pour capture RAW")

                        # Réinitialiser les compteurs de session Lucky Stack
                        if hasattr(pygame, '_lucky_last_displayed'):
                            delattr(pygame, '_lucky_last_displayed')
                        if hasattr(pygame, '_lucky_cached_image'):
                            delattr(pygame, '_lucky_cached_image')
                        if hasattr(pygame, '_lucky_last_saved'):
                            delattr(pygame, '_lucky_last_saved')
                        if hasattr(pygame, '_lucky_resolution_check'):
                            delattr(pygame, '_lucky_resolution_check')
                        if hasattr(pygame, '_lucky_stack_resolution_debug'):
                            delattr(pygame, '_lucky_stack_resolution_debug')
                        # Réinitialiser les flags de validation RAW
                        if hasattr(pygame, '_raw_format_mismatch_warning'):
                            delattr(pygame, '_raw_format_mismatch_warning')
                        if hasattr(pygame, '_raw_resolution_mismatch_warning'):
                            delattr(pygame, '_raw_resolution_mismatch_warning')
                        if hasattr(pygame, '_stacking_mode_shown'):
                            delattr(pygame, '_stacking_mode_shown')

                        # Activer le mode stretch pour affichage fullscreen
                        stretch_mode = 1

                        # Passer en mode fullscreen (comme pour LIVE STACK)
                        display_modes = pygame.display.list_modes()
                        if display_modes and display_modes != -1:
                            max_width, max_height = display_modes[0]
                        else:
                            screen_info = pygame.display.Info()
                            max_width, max_height = screen_info.current_w, screen_info.current_h
                        windowSurfaceObj = pygame.display.set_mode((max_width, max_height), pygame.FULLSCREEN, 24)

                        # Créer l'instance luckystack si nécessaire
                        if luckystack is None:
                            # Récupérer les paramètres actuels de la caméra
                            camera_params = {
                                'exposure': sspeed,  # Déjà en microsecondes
                                'gain': gain,
                                'red': red / 10,  # Divisé par 10 pour ColourGains
                                'blue': blue / 10,
                                'raw_format': raw_formats[raw_format]  # YUV420/XRGB8888/SRGGB12/SRGGB16
                            }
                            luckystack = create_advanced_livestack_session(camera_params)

                            # Configurer avec paramètres Lucky Imaging
                            luckystack.configure(
                                # Stacker avancé
                                stacking_method=['mean', 'median', 'kappa_sigma', 'winsorized', 'weighted'][ls_stack_method],
                                kappa=ls_stack_kappa / 10.0,
                                iterations=ls_stack_iterations,

                                # Contrôle qualité DÉSACTIVÉ pour Lucky Imaging
                                # (Lucky fait déjà sa propre sélection via scoring)
                                enable_qc=False,

                                # Planétaire (peut être activé avec Lucky)
                                planetary_enable=bool(ls_planetary_enable),
                                planetary_mode=ls_planetary_mode,
                                planetary_disk_min=ls_planetary_disk_min,
                                planetary_disk_max=ls_planetary_disk_max,
                                planetary_disk_threshold=ls_planetary_threshold,
                                planetary_disk_margin=ls_planetary_margin,
                                planetary_disk_ellipse=bool(ls_planetary_ellipse),
                                planetary_window=planetary_windows[ls_planetary_window],
                                planetary_upsample=ls_planetary_upsample,
                                planetary_highpass=bool(ls_planetary_highpass),
                                planetary_roi_center=bool(ls_planetary_roi_center),
                                planetary_corr=ls_planetary_corr / 100.0,
                                planetary_max_shift=float(ls_planetary_max_shift),

                                # Lucky Imaging (activé)
                                lucky_enable=True,  # Toujours activé en mode Lucky Stack
                                lucky_buffer_size=ls_lucky_buffer,
                                lucky_keep_percent=float(ls_lucky_keep),
                                lucky_score_method=['laplacian', 'gradient', 'sobel', 'tenengrad'][ls_lucky_score],
                                lucky_stack_method=['mean', 'median', 'sigma_clip'][ls_lucky_stack],
                                lucky_align_enabled=bool(ls_lucky_align),
                                lucky_score_roi_percent=float(ls_lucky_roi),

                                # PNG avec stretch selon paramètres interface
                                png_stretch=['off', 'ghs', 'asinh'][stretch_preset],
                                png_factor=stretch_factor / 10.0,
                                # Pour GHS: pas de normalisation par percentiles (clip 0-100 = désactivé)
                                # Pour Arcsinh: normalisation active (stretch_p_low/high)
                                png_clip_low=0.0 if stretch_preset == 1 else stretch_p_low / 10.0,
                                png_clip_high=100.0 if stretch_preset == 1 else stretch_p_high / 100.0,
                                ghs_D=ghs_D / 10.0,
                                ghs_b=ghs_b / 10.0,
                                ghs_SP=ghs_SP / 100.0,
                                ghs_LP=ghs_LP / 100.0,
                                ghs_HP=ghs_HP / 100.0,
                                preview_refresh=ls_preview_refresh,
                                save_dng="none"
                            )

                        # === NOUVELLE APPROCHE RAW: ISP/Stretch appliqué EXTERNE (même que LiveStack) ===
                        is_raw_mode = (raw_format >= 2)

                        if is_raw_mode:
                            print(f"[LUCKYSTACK RAW] Mode RAW détecté - ISP/stretch seront appliqués EXTERNE")
                            print(f"  → libastrostack: stacking brut uniquement (isp_enable=False, stretch=none)")
                            print(f"  → RPiCamera2: appliquera apply_isp_to_preview() + astro_stretch()")
                        else:
                            if isp_enable == 1:
                                print(f"[INFO] ISP logiciel ignoré pour {raw_formats[raw_format]} (déjà traité par ISP matériel)")

                        # Configuration pour mode externe ISP/stretch
                        video_format_map = {0: 'yuv420', 1: 'xrgb8888', 2: 'raw12', 3: 'raw16'}
                        luckystack.configure(
                            isp_enable=False,  # ISP externe pour RAW
                            isp_config_path=None,
                            video_format=video_format_map.get(raw_format, 'yuv420'),
                            png_stretch='off' if is_raw_mode else 'ghs',  # Stretch externe pour RAW
                            # Mettre à jour les paramètres Lucky modifiés entre deux sessions
                            lucky_stack_method=['mean', 'median', 'sigma_clip'][ls_lucky_stack],
                            lucky_buffer_size=ls_lucky_buffer,
                            lucky_keep_percent=float(ls_lucky_keep),
                            lucky_score_method=['laplacian', 'gradient', 'sobel', 'tenengrad'][ls_lucky_score],
                            lucky_align_enabled=bool(ls_lucky_align),
                            lucky_score_roi_percent=float(ls_lucky_roi),
                            # Paramètres GHS pour mode RGB/YUV
                            ghs_D=ghs_D / 10.0,
                            ghs_b=ghs_b / 10.0,
                            ghs_SP=ghs_SP / 100.0,
                            ghs_LP=ghs_LP / 100.0,
                            ghs_HP=ghs_HP / 100.0,
                        )

                        # Mettre à jour le format RAW actuel avant de démarrer
                        luckystack.camera_params['raw_format'] = raw_formats[raw_format]

                        # Réinitialiser les compteurs avant de redémarrer
                        luckystack.reset()

                        # Démarrer la session
                        luckystack.start()

                        # Pour RAW mode, l'ISP/stretch sera appliqué externe (dans le code d'affichage)
                        # Plus besoin de apply_isp_to_session() car libastrostack n'a pas d'ISP actif

                        # Réinitialiser les compteurs d'affichage et de sauvegarde
                        pygame._lucky_last_displayed = 0
                        pygame._lucky_cached_image = None
                        pygame._lucky_last_saved = 0

                        # Réinitialiser les flags de debug pour voir les infos à chaque activation
                        if hasattr(pygame, '_lucky_resolution_check'):
                            delattr(pygame, '_lucky_resolution_check')
                        if hasattr(pygame, '_lucky_stack_resolution_debug'):
                            delattr(pygame, '_lucky_stack_resolution_debug')
                        if hasattr(pygame, '_lucky_stretch_info_shown'):
                            delattr(pygame, '_lucky_stretch_info_shown')
                        if hasattr(pygame, '_lucky_stretch_applied_shown'):
                            delattr(pygame, '_lucky_stretch_applied_shown')
                        if hasattr(pygame, '_lucky_no_stretch_shown'):
                            delattr(pygame, '_lucky_no_stretch_shown')
                        # Réinitialiser le flag de debug pour voir la méthode de stacking
                        if luckystack is not None and luckystack.lucky_stacker is not None:
                            if hasattr(luckystack.lucky_stacker, '_stack_method_shown'):
                                delattr(luckystack.lucky_stacker, '_stack_method_shown')

                        print("[LUCKYSTACK] Mode activé")
                    else:
                        # Désactiver Lucky Stack
                        luckystack_active = False
                        stretch_mode = 0  # Désactiver aussi le stretch

                        if luckystack is not None:
                            # Sauvegarder le résultat final avant de stopper (si activé)
                            if ls_lucky_save_final == 1:
                                try:
                                    # Mode RAW: utiliser traitement externe pour cohérence avec preview
                                    if raw_format >= 2:
                                        save_with_external_processing(luckystack, raw_format_name=raw_formats[raw_format])
                                    else:
                                        # Mode RGB/YUV: utiliser libastrostack normalement
                                        luckystack.save(raw_format_name=raw_formats[raw_format])
                                    print("[LUCKYSTACK] Image finale sauvegardée")
                                except Exception as e:
                                    print(f"[LUCKYSTACK] Erreur sauvegarde: {e}")
                            else:
                                print("[LUCKYSTACK] Sauvegarde finale désactivée (ls_lucky_save_final=0)")
                            luckystack.stop()

                        # Restaurer le mode d'affichage normal (avec l'interface)
                        if frame == 1:
                            if fullscreen == 1:
                                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                            else:
                                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
                        else:
                            windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)

                        # Effacer l'écran et redessiner le menu
                        windowSurfaceObj.fill((0, 0, 0))
                        Menu()
                        pygame.display.update()
                        print("[LUCKYSTACK] Mode désactivé")

                        # *** Reconfigurer caméra pour revenir au mode ISP après le RAW ***
                        if raw_format >= 2 and not livestack_active:
                            print(f"[LUCKYSTACK] Retour mode ISP - reconfiguration caméra...")
                            kill_preview_process()
                            preview()
                            print(f"[LUCKYSTACK] Caméra reconfigurée en mode normal")

                elif button_row == 5:
                    # STRETCH - Active le mode stretch astro pour le preview
                    stretch_mode = 1
                    # Si format RAW sélectionné, configurer capture_thread pour capturer en RAW
                    # Cela permet de prévisualiser les images telles qu'elles seront stackées
                    if raw_format >= 2 and capture_thread is not None:
                        capture_thread.set_capture_params({'type': 'raw'})
                        if show_cmds == 1:
                            print(f"[STRETCH] Mode RAW preview activé - capture_thread configuré en RAW")
                    # Passer en mode fullscreen pour couvrir aussi la barre de tâches
                    display_modes = pygame.display.list_modes()
                    if display_modes and display_modes != -1:
                        max_width, max_height = display_modes[0]
                    else:
                        screen_info = pygame.display.Info()
                        max_width, max_height = screen_info.current_w, screen_info.current_h
                    windowSurfaceObj = pygame.display.set_mode((max_width, max_height), pygame.FULLSCREEN, 24)

                elif button_row == 6:
                    menu = 1
                    Menu()

                elif button_row == 7:
                    menu = 2
                    Menu()

                elif button_row == 8:
                    # EXIT
                    kill_preview_process()
                    if picam2 is not None:
                        try:
                            picam2.stop()
                            picam2.close()
                        except:
                            pass
                    pygame.display.quit()
                    sys.exit()
                   
                                           
            # MENU 1
            elif menu == 1:
              if button_row == 1:
                  menu = 3
                  Menu()
              elif button_row == 2:
                  menu = 5
                  Menu()
              elif button_row == 3:
                  menu = 6
                  Menu()
              elif button_row == 6:
                  menu = 8
                  Menu()
              elif button_row == 7:
                  # LUCKY STACK Settings - menu placeholder
                  menu = 9  # Nouveau menu pour LUCKY STACK Settings
                  Menu()
              elif button_row == 8:
                  menu = 7  # STRETCH Settings (décalé)
                  Menu()

              elif button_row == 4:
                # ZOOM
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'zoom':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    zoom = int(((mousex-preview_width) / bw) * (pmax+1-pmin))###
                    # Zoom manuel supprimé: simplifié la condition
                    if igw/igh > 1.5 and alt_dis == 0:
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height))
                elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    zoom = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    # Zoom manuel supprimé: simplifié la condition
                    if igw/igh > 1.5:
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height/4))
                elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    zoom = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                # Zoom manuel supprimé: simplifié la condition
                elif (alt_dis == 0 and mousex > preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 1):
                    zoom +=1
                    zoom = min(zoom,pmax)
                elif alt_dis == 0 and mousex < preview_width + (bw/2)  and zoom > 0:
                    zoom -=1
                    # Zoom manuel supprimé: simplifié la condition
                    if igw/igh > 1.5 and alt_dis == 0:
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height))
                # Synchroniser la résolution vidéo avec le zoom
                sync_video_resolution_with_zoom()
                print(zoom)
                if zoom < 2:
                    if zoom == 0:
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        draw_Vbar(0,4,greyColor,'zoom',zoom)
                    else:
                        button(0,4,1,9)
                        text(0,4,2,0,1,"ZOOMED",ft,0)
                        text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
                        draw_Vbar(0,4,dgryColor,'zoom',zoom)

                    if foc_man == 0 and focus_mode == 0:
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                    # determine if camera native format
                    vw = 0
                    x = 0
                    while x < len(vwidths2) and vw == 0:
                        if vwidth == vwidths2[x]:
                             if vheight == vheights2[x]:
                                vw = 1
                        x += 1
                    
                else:
                    button(0,4,1,9)
                    text(0,4,2,0,1,"ZOOMED",ft,0)
                    text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
                    draw_Vbar(0,4,dgryColor,'zoom',zoom)

                # Maintenir le bouton Focus actif si on est en mode focus
                if focus_mode == 1:
                    # NE PAS recentrer le réticule - l'utilisateur peut l'avoir déplacé
                    # xx = int(preview_width/2)
                    # xy = int(preview_height/2)
                    button(0,5,1,9)
                    text(0,5,3,0,1,"FOCUS",ft,0)

                if zoom > 0:
                    fxx = 0
                    fxy = 0
                    fxz = 1
                    fyz = 1
                    if (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0:
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                restart = 1
                time.sleep(.2)
                                         
                             
              elif button_row == 5:
                # FOCUS
                if (Pi_Cam == 3 and v3_af == 1):
                    for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'v3_focus':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                if Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8:
                    if Pi_Cam == 5:
                      for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'v5_focus':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                    if Pi_Cam == 6 or Pi_Cam == 8:
                      for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'v6_focus':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                # arducam manual focus slider
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + (bh/3)) and ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and foc_man == 1:
                    focus = int(((mousex-preview_width) / bw) * pmax)
                    if Pi_Cam == 5:
                        draw_Vbar(0,5,dgryColor,'v5_focus',focus)
                    elif Pi_Cam == 6 or Pi_Cam == 8:
                        draw_Vbar(0,5,dgryColor,'v6_focus',focus)
                    v3_focus = focus
                    restart = 1
                    text(0,5,3,0,1,'<<< ' + str(focus) + ' >>>',fv,0)
                #arducam manual focus buttons    
                elif mousex > preview_width and mousey > ((button_row)*bh) + (bh/3) and mousey < ((button_row)*bh) + (bh/1.5) and ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and foc_man == 1:
                    if button_pos == 2:
                        focus -= 10
                        focus = max(focus,pmin)
                    elif button_pos == 3:
                        focus += 10
                        focus = min(focus,pmax)
                    if Pi_Cam == 5:
                        draw_Vbar(0,3,dgryColor,'v5_focus',focus)
                    elif Pi_Cam == 6 or Pi_Cam == 8:
                        draw_Vbar(0,3,dgryColor,'v6_focus',focus)
                    v3_focus = focus
                    restart = 1
                    text(0,5,3,0,1,'<<< ' + str(focus) + ' >>>',fv,0)
                # Pi v3 manual focus slider
                elif (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + (bh/3)) and (Pi_Cam == 3 and v3_af == 1) and foc_man == 1:
                    v3_focus = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) + pmin
                    draw_Vbar(0,5,dgryColor,'v3_focus',v3_focus-pmin)
                    fd = 1/(v3_focus/100)
                    text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
                    restart = 1
                # Pi v3 manual focus buttons
                elif mousex > preview_width and mousey > ((button_row)*bh) + (bh/3) and mousey < ((button_row)*bh) + (bh/1.5) and (Pi_Cam == 3 and v3_af == 1)  and foc_man == 1:
                    if button_pos == 2:
                        v3_focus -= 1
                        v3_focus = max(v3_focus,pmin)
                    elif button_pos == 3:
                        v3_focus += 1
                        v3_focus = min(v3_focus,pmax)
                    draw_Vbar(0,3,dgryColor,'v3_focus',v3_focus-pmin)
                    fd = 1/(v3_focus/100)
                    text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
                    restart = 1

                elif alt_dis == 0:
                    # determine if camera native format
                    vw = 0
                    x = 0
                    while x < len(vwidths2) and vw == 0:
                        if vwidth == vwidths2[x]:
                             if vheight == vheights2[x]:
                                vw = 1
                        x += 1
                    # FOCUS button NON AF camera (ajout IMX585 = Pi_Cam 10)
                    if (Pi_Cam < 3 or Pi_Cam == 4 or Pi_Cam == 7 or Pi_Cam == 9 or Pi_Cam == 10 or (Pi_Cam ==3 and v3_af == 0)) and focus_mode == 0:
                        zoom = 4
                        sync_video_resolution_with_zoom()
                        focus_mode = 1
                        # Recentrer le réticule (activation initiale du mode focus)
                        xx = int(preview_width/2)
                        xy = int(preview_height/2)
                        button(0,5,1,9)
                        text(0,5,3,0,1,"FOCUS",ft,0)
                        button(0,4,1,9)
                        text(0,4,2,0,1,"ZOOMED",ft,0)
                        text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
                        draw_Vbar(0,4,dgryColor,'zoom',zoom)
                        time.sleep(0.25)
                        restart = 1
                    # CANCEL FOCUS NON AF camera (ajout IMX585 = Pi_Cam 10)
                    elif (Pi_Cam < 3 or Pi_Cam == 4 or Pi_Cam == 7 or Pi_Cam == 9 or Pi_Cam == 10 or (Pi_Cam ==3 and v3_af == 0)) and focus_mode == 1:
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height/4))
                        button(0,5,0,9)
                        button(0,4,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,"",fv,7)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        draw_Vbar(0,4,greyColor,'zoom',zoom)
                        restart = 1
                    # Pi V3 manual focus
                    elif Pi_Cam == 3 and v3_af == 1 and v3_f_mode == 0:
                        focus_mode = 1
                        v3_f_mode = 1
                        foc_man = 1
                        # Recentrer le réticule (activation initiale du mode focus)
                        xx = int(preview_width/2)
                        xy = int(preview_height/2)
                        restart = 1
                        button(0,5,1,9)
                        time.sleep(0.25)
                        draw_Vbar(0,5,dgryColor,'v3_focus',v3_focus-pmin)
                        fd = 1/(v3_focus/100)
                        text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,0)
                        time.sleep(0.25)
                    # ARDUCAM manual focus
                    elif ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and v3_f_mode == 0:
                        focus_mode = 1
                        v3_f_mode = 1
                        foc_man = 1
                        # Recentrer le réticule (activation initiale du mode focus)
                        xx = int(preview_width/2)
                        xy = int(preview_height/2)
                        text(0,5,3,0,1,'<<< ' + str(focus) + ' >>>',fv,0)
                        if Pi_Cam == 5:
                            draw_Vbar(0,5,dgryColor,'v5_focus',focus)
                        if Pi_Cam == 6 or Pi_Cam == 8:
                            draw_Vbar(0,5,dgryColor,'v6_focus',focus)
                        text(0,5,3,1,1,"manual",fv,0)
                        time.sleep(0.25)
                        restart = 1
                    # ARDUCAM cancel manual focus
                    elif ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and foc_man == 1:
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        foc_man = 0
                        if Pi_Cam == 8:
                            v3_f_mode = 2 # continuous focus
                        else:
                            v3_f_mode = 0
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        fxx = 0
                        fxy = 0
                        fxz = 1
                        fyz = 0.75
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        time.sleep(0.25)
                        restart = 1
                    # Pi V3 cancel manual focus
                    elif (Pi_Cam == 3 and v3_af == 1)  and v3_f_mode == 1:
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        v3_f_mode = 2 # continuous focus
                        foc_man = 0
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        fxx = 0
                        fxy = 0
                        fxz = 1
                        fyz = 0.75
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,0,preview_width,preview_height))
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        time.sleep(0.25)
                        restart = 1
                    # AF camera to AUTO
                    elif ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8))) and v3_f_mode == 2:
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        v3_f_mode = 0 # auto focus
                        foc_man = 0
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        fxx = 0
                        fxy = 0
                        fxz = 1
                        fyz = 0.75
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,0,preview_width,preview_height))
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        time.sleep(0.25)
                        restart = 1
                time.sleep(.25)
                
                              
              elif button_row == 6 and Pi_Cam == 3 and v3_af == 1:
                # V3 FOCUS SPEED 
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'v3_f_speed':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    v3_f_speed = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    v3_f_speed = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    v3_f_speed = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        v3_f_speed-=1
                        v3_f_speed = max(v3_f_speed,pmin)
                    else:
                        v3_f_speed +=1
                        v3_f_speed = min(v3_f_speed,pmax)
                text(0,7,3,1,1,v3_f_speeds[v3_f_speed],fv,7)
                draw_bar(0,6,greyColor,'v3_f_speed',v3_f_speed)
                restart = 1
                time.sleep(.25)
                
              elif button_row == 7 and Pi_Cam == 3 and v3_af == 1:
                # V3 FOCUS RANGE 
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'v3_f_range':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    v3_f_range = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    v3_f_range = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    v3_f_range = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        v3_f_range-=1
                        v3_f_range = max(v3_f_range,pmin)
                    else:
                        v3_f_range +=1
                        v3_f_range = min(v3_f_range,pmax)
                text(0,7,3,1,1,v3_f_ranges[v3_f_range],fv,7)
                draw_Vbar(0,7,greyColor,'v3_f_range',v3_f_range)
                restart = 1
                time.sleep(.25)
            
            elif menu == 2:
              if button_row == 1 and cam1 != "1":
                # SWITCH CAMERA
                camera += 1
                if camera > max_camera:
                    camera = 0
                text(0,1,3,1,1,str(camera),fv,7)
                if not use_picamera2 and p is not None:
                    poll = p.poll()
                    if poll == None:
                        os.killpg(p.pid, signal.SIGTERM)
                focus_mode = 0
                reset_fwhm_history()
                reset_hfr_history()
                v3_f_mode = 0 
                foc_man = 0
                fxx = 0
                fxy = 0
                fxz = 1
                fyz = 1
                Camera_Version()
                Menu()
                restart = 1
                
              elif button_row == 2:
                # Accès menu METRICS Settings
                menu = 10
                Menu()

              elif button_row == 3:
                # HISTOGRAM
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'histogram':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    histogram = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    histogram = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    histogram = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        histogram -=1
                        histogram = max(histogram,pmin)
                    else:
                        histogram +=1
                        histogram = min(histogram,pmax)
                text(0,3,3,1,1,histograms[histogram],fv,7)
                draw_bar(0,3,greyColor,'histogram',histogram)
                time.sleep(.25)

              elif button_row == 4:
                # HISTOGRAM SIZE
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'histarea':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    histarea = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    histarea = max(histarea,pmin)
                elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    histarea = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    histarea = max(histarea,pmin)
                elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    histarea = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    histarea = max(histarea,pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        histarea -=1
                        histarea = max(histarea,pmin)
                    else:
                        histarea +=1
                        histarea = min(histarea,pmax)
                if xx - histarea < 0 or xy - histarea < 0:
                    histarea = old_histarea
                if xy + histarea > preview_height or xx + histarea > preview_width:
                    histarea = old_histarea
                if (Pi_Cam == 3 and v3_af == 1) and (xy + histarea > preview_height * 0.75 or xx + histarea > preview_width):
                    histarea = old_histarea
                text(0,4,3,1,1,str(histarea),fv,7)
                draw_Vbar(0,4,greyColor,'histarea',histarea)
                old_histarea = histarea
                time.sleep(.25)

              elif button_row == 5:
                # VERTICAL FLIP
                vflip +=1
                if vflip > 1:
                    vflip = 0
                text(0,5,3,1,1,str(vflip),fv,7)
                restart = 1
                time.sleep(.25)

              elif button_row == 6:
                # HORIZONTAL FLIP
                hflip += 1
                if hflip > 1:
                    hflip = 0
                text(0,6,3,1,1,str(hflip),fv,7)
                restart = 1
                time.sleep(.25)

              elif button_row == 7:
                # RAW Format (YUV420/XRGB8888/SRGGB12/SRGGB16)
                if alt_dis == 0 and mousex < preview_width + (bw/2):
                    raw_format -= 1
                    if raw_format < 0:
                        raw_format = 3
                else:
                    raw_format += 1
                    if raw_format > 3:
                        raw_format = 0
                text(0,7,3,1,1,raw_formats[raw_format],fv,7)
                draw_Vbar(0,7,greyColor,'raw_format',raw_format)
                # Mettre à jour camera_params si livestack/luckystack actif
                if livestack is not None and livestack_active:
                    livestack.camera_params['raw_format'] = raw_formats[raw_format]
                if luckystack is not None and luckystack_active:
                    luckystack.camera_params['raw_format'] = raw_formats[raw_format]
                time.sleep(0.25)

              elif button_row == 8:
                # SENSOR MODE (Native/Binning)
                use_native_sensor_mode = 1 - use_native_sensor_mode  # Toggle 0/1
                if use_native_sensor_mode == 0:
                    text(0,8,3,1,1,"Binning",fv,7)
                else:
                    text(0,8,3,1,1,"Native",fv,7)
                draw_Vbar(0,8,greyColor,'use_native_sensor_mode',use_native_sensor_mode)
                restart = 1  # Forcer recréation de la caméra
                time.sleep(.25)

              elif button_row == 9:
                   # SAVE CONFIG
                   text(0,9,3,0,1,"SAVE Config",fv,7)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = raw_format  # Remplace timet (fixé à 100ms)
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   config[42] = ghs_D
                   config[43] = ghs_b
                   config[44] = ghs_SP
                   config[45] = ghs_LP
                   config[46] = ghs_HP
                   config[47] = ghs_preset
                   config[48] = ls_preview_refresh
                   config[49] = ls_alignment_mode
                   config[50] = ls_enable_qc
                   config[51] = ls_max_fwhm
                   config[52] = ls_min_sharpness
                   config[53] = ls_max_drift
                   config[54] = ls_min_stars
                   config[55] = ls_stack_method
                   config[56] = ls_stack_kappa
                   config[57] = ls_stack_iterations
                   config[58] = ls_planetary_enable
                   config[59] = ls_planetary_mode
                   config[60] = ls_planetary_disk_min
                   config[61] = ls_planetary_disk_max
                   config[62] = ls_planetary_threshold
                   config[63] = ls_planetary_margin
                   config[64] = ls_planetary_ellipse
                   config[65] = ls_planetary_window
                   config[66] = ls_planetary_upsample
                   config[67] = ls_planetary_highpass
                   config[68] = ls_planetary_roi_center
                   config[69] = ls_planetary_corr
                   config[70] = ls_planetary_max_shift
                   config[71] = ls_lucky_buffer
                   config[72] = ls_lucky_keep
                   config[73] = ls_lucky_score
                   config[74] = ls_lucky_stack
                   config[75] = ls_lucky_align
                   config[76] = ls_lucky_roi
                   config[77] = use_native_sensor_mode
                   config[78] = focus_method
                   config[79] = star_metric
                   config[80] = snr_display
                   config[81] = metrics_interval
                   config[82] = ls_lucky_save_progress
                   config[83] = isp_enable
                   config[84] = allsky_mode
                   config[85] = allsky_mean_target
                   config[86] = allsky_mean_threshold
                   config[87] = allsky_video_fps
                   config[88] = allsky_max_gain
                   config[89] = allsky_apply_stretch
                   config[90] = allsky_cleanup_jpegs
                   config[91] = ls_save_progress
                   config[92] = ls_save_final
                   config[93] = ls_lucky_save_final
                   config[94] = fix_bad_pixels
                   config[95] = fix_bad_pixels_sigma
                   config[96] = fix_bad_pixels_min_adu
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,9,2,0,1,"SAVE CONFIG",fv,7)

            # MENU 3
            elif menu == 3:
              if button_row == 9:
                  menu = 4
                  Menu()
              elif button_row == 1:
                # MODE
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'mode':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    mode = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    mode = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    mode = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        mode -=1
                        mode  = max(mode ,pmin)
                    else:
                        mode  +=1
                        mode = min(mode ,pmax)
                if mode == 0:
                    text(0,2,5,0,1,"Shutter S",ft,10)
                    draw_bar(0,2,lgrnColor,'speed',speed)
                    if shutters[speed] < 0:
                        text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
                    else:
                        text(0,2,3,1,1,str(shutters[speed]),fv,10)
                    if gain == 0:
                        gain = 1
                        text(0,3,5,0,1,"Gain    A/D",ft,10)
                        if gain <= mag:
                            text(0,3,3,1,1,str(gain) + " :  " + str(gain) + "/1",fv,10)
                        else:
                            text(0,3,3,1,1,str(gain) + " :  " + str(int(mag)) + "/" + str(((gain/mag)*10)/10)[0:3],fv,10)
                        draw_bar(0,3,lgrnColor,'gain',gain)
                else:
                    text(0,2,5,0,1,"eV",ft,10)
                    text(0,2,3,1,1,str(ev),fv,10)
                    draw_bar(0,2,lgrnColor,'ev',ev)
                    gain = 0
                    text(0,3,5,0,1,"Gain ",ft,10)
                    text(0,3,3,1,1,"Auto",fv,10)
                    draw_bar(0,3,lgrnColor,'gain',gain)
                text(0,1,3,1,1,modes[mode],fv,10)
                draw_bar(0,2,lgrnColor,'mode',mode)
                td = timedelta(seconds=tinterval)
                if tinterval > 0:
                    tduration = tinterval * tshots
                if mode == 0 and tinterval == 0 :
                    speed = 15
                    custom_sspeed = 0  # Réinitialiser car on passe en mode manuel
                    shutter = shutters[speed]
                    if shutter < 0:
                        shutter = abs(1/shutter)
                    sspeed = int(shutter * 1000000)
                    if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
                        sspeed +=1
                    if shutters[speed] < 0:
                        text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
                    else:
                        text(0,2,3,1,1,str(shutters[speed]),fv,10)
                    draw_bar(0,2,lgrnColor,'speed',speed)

                time.sleep(.25)
                restart = 1

              elif button_row == 2:
                # SHUTTER SPEED or EV (dependent on MODE set)
                if mode == 0 :
                    for f in range(0,len(still_limits)-1,3):
                        if still_limits[f] == 'speed':
                            pmin = still_limits[f+1]
                            pmax = max_speed

                    # Vérifier si c'est un clic sur le slider (zone de la barre)
                    clicked_on_slider = False
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        clicked_on_slider = True
                    elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                        clicked_on_slider = True
                    elif (mousey > preview_height * .75 and mousey < preview_height * .75  + int(bh/3)) and alt_dis == 2:
                        clicked_on_slider = True

                    # Détecter le double-clic sur le slider
                    current_time = time.time()
                    is_double_click = False
                    if clicked_on_slider and last_click_row == button_row:
                        time_diff = current_time - last_click_time
                        print(f"[DEBUG SPEED] Double-clic détecté? time_diff={time_diff:.3f}s, DELAY={DOUBLE_CLICK_DELAY}")
                        if time_diff < DOUBLE_CLICK_DELAY:
                            is_double_click = True
                            print("[DEBUG SPEED] DOUBLE-CLIC CONFIRMÉ!")

                    # Mettre à jour les variables de détection du double-clic
                    if clicked_on_slider:
                        print(f"[DEBUG SPEED] Clic sur slider détecté, button_row={button_row}")
                        last_click_time = current_time
                        last_click_row = button_row
                        last_click_x = mousex
                        last_click_y = mousey

                    # Si double-clic, afficher le dialogue de saisie PRECISE en millisecondes
                    if is_double_click:
                        print("[DEBUG SPEED] Ouverture du dialogue de saisie PRECISE en ms...")
                        # Afficher la valeur actuelle (custom ou standard)
                        if custom_sspeed > 0:
                            current_ms = custom_sspeed / 1000.0
                        else:
                            current_ms = shutter_index_to_ms(speed)
                        # Calculer les limites en ms
                        min_ms = shutter_index_to_ms(pmin)  # ~0.25ms pour 1/4000s
                        max_ms = shutter_index_to_ms(pmax)  # Dépend de max_speed
                        new_ms = numeric_input_dialog(f"Expo en ms ({min_ms:.1f}-{max_ms:.0f})", round(current_ms, 1), 0, int(max_ms) + 1)
                        if new_ms is not None:
                            # Stocker la valeur PRECISE en microsecondes
                            custom_sspeed = int(float(new_ms) * 1000)
                            # Trouver l'index le plus proche pour l'affichage du slider uniquement
                            speed = ms_to_shutter_index(float(new_ms))
                            speed = max(speed, pmin)
                            speed = min(speed, pmax)
                            print(f"[DEBUG SPEED] Nouvelle expo PRECISE: {new_ms}ms = {custom_sspeed}µs (slider index {speed})")
                        # Réinitialiser pour éviter un triple-clic
                        last_click_time = 0
                        last_click_row = -1
                    elif clicked_on_slider:
                        # Clic simple sur le slider - réinitialiser custom_sspeed
                        custom_sspeed = 0
                        if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                            speed = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                        elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                            speed = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                        elif (mousey > preview_height * .75 and mousey < preview_height * .75  + int(bh/3)) and alt_dis == 2:
                            speed = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    else:
                        # Clic sur les boutons +/- ou zone centrale (main)
                        # Zone divisée en 3: gauche (0-33%) = "-", centre (33%-66%) = "main", droite (66%-100%) = "+"
                        click_zone_x = mousex - preview_width
                        if alt_dis == 0:
                            if click_zone_x < bw/3:
                                # Zone gauche: diminuer - réinitialiser custom_sspeed
                                custom_sspeed = 0
                                speed -= 1
                                speed = max(speed, pmin)
                            elif click_zone_x < 2*bw/3:
                                # Zone centrale: ouvrir le pavé numérique pour saisie PRECISE en ms
                                print("[DEBUG SPEED] Clic sur zone centrale (main) - ouverture du pavé numérique")
                                # Afficher la valeur actuelle (custom ou standard)
                                if custom_sspeed > 0:
                                    current_ms = custom_sspeed / 1000.0
                                else:
                                    current_ms = shutter_index_to_ms(speed)
                                min_ms = shutter_index_to_ms(pmin)
                                max_ms = shutter_index_to_ms(pmax)
                                new_ms = numeric_input_dialog(f"Expo en ms ({min_ms:.1f}-{max_ms:.0f})", round(current_ms, 1), 0, int(max_ms) + 1)
                                if new_ms is not None:
                                    # Stocker la valeur PRECISE en microsecondes
                                    custom_sspeed = int(float(new_ms) * 1000)
                                    # Trouver l'index le plus proche pour l'affichage du slider uniquement
                                    speed = ms_to_shutter_index(float(new_ms))
                                    speed = max(speed, pmin)
                                    speed = min(speed, pmax)
                                    print(f"[DEBUG SPEED] Nouvelle expo PRECISE: {new_ms}ms = {custom_sspeed}µs (slider index {speed})")
                            else:
                                # Zone droite: augmenter - réinitialiser custom_sspeed
                                custom_sspeed = 0
                                speed += 1
                                speed = min(speed, pmax)
                        elif alt_dis > 0 and button_pos == 0:
                            custom_sspeed = 0
                            speed -= 1
                            speed = max(speed, pmin)
                        else:
                            custom_sspeed = 0
                            speed += 1
                            speed = min(speed, pmax)

                    # Calculer sspeed: utiliser custom_sspeed si défini, sinon depuis shutters[speed]
                    if custom_sspeed > 0:
                        sspeed = custom_sspeed
                        # Afficher la valeur en ms
                        if sspeed >= 1000000:
                            text(0,2,3,1,1,f"{sspeed/1000000:.1f}s",fv,10)
                        else:
                            text(0,2,3,1,1,f"{sspeed/1000:.1f}ms",fv,10)
                    else:
                        shutter = shutters[speed]
                        if shutter < 0:
                            shutter = abs(1/shutter)
                        sspeed = int(shutter * 1000000)
                        if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
                            sspeed +=1
                        if shutters[speed] < 0:
                            text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
                        else:
                            text(0,2,3,1,1,str(shutters[speed]),fv,10)
                    draw_bar(0,2,lgrnColor,'speed',speed)
                    if tinterval > 0:
                        tinterval = int(sspeed/1000000)
                        tinterval = max(tinterval,1)
                        td = timedelta(seconds=tinterval)
                        tduration = tinterval * tshots
                        td = timedelta(seconds=tduration)

                    time.sleep(.25)
                    restart = 1
                else:
                    # EV
                    for f in range(0,len(still_limits)-1,3):
                        if still_limits[f] == 'ev':
                            pmin = still_limits[f+1]
                            pmax = still_limits[f+2]

                    # Vérifier si c'est un clic sur le slider (zone de la barre)
                    clicked_on_slider = False
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        clicked_on_slider = True
                    elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                        clicked_on_slider = True
                    elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                        clicked_on_slider = True

                    # Détecter le double-clic sur le slider
                    current_time = time.time()
                    is_double_click = False
                    if clicked_on_slider and last_click_row == button_row:
                        time_diff = current_time - last_click_time
                        print(f"[DEBUG EV] Double-clic détecté? time_diff={time_diff:.3f}s, DELAY={DOUBLE_CLICK_DELAY}")
                        if time_diff < DOUBLE_CLICK_DELAY:
                            is_double_click = True
                            print("[DEBUG EV] DOUBLE-CLIC CONFIRMÉ!")

                    # Mettre à jour les variables de détection du double-clic
                    if clicked_on_slider:
                        print(f"[DEBUG EV] Clic sur slider détecté, button_row={button_row}")
                        last_click_time = current_time
                        last_click_row = button_row
                        last_click_x = mousex
                        last_click_y = mousey

                    # Si double-clic, afficher le dialogue de saisie
                    if is_double_click:
                        print("[DEBUG EV] Ouverture du dialogue de saisie...")
                        new_ev = numeric_input_dialog("Saisir EV (" + str(pmin) + " a " + str(pmax) + ")", ev, pmin, pmax)
                        if new_ev is not None:
                            ev = int(new_ev)
                            print(f"[DEBUG EV] Nouvelle valeur EV saisie: {ev}")
                        # Réinitialiser pour éviter un triple-clic
                        last_click_time = 0
                        last_click_row = -1
                    elif clicked_on_slider:
                        # Clic simple sur le slider
                        if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                            ev = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) + pmin
                        elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                            ev = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin
                        elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                            ev = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin
                    else:
                        # Clic sur les boutons +/- ou zone centrale (main)
                        # Zone divisée en 3: gauche (0-33%) = "-", centre (33%-66%) = "main", droite (66%-100%) = "+"
                        click_zone_x = mousex - preview_width
                        if alt_dis == 0:
                            # Mode normal: calculer la position relative dans la zone de bouton
                            if click_zone_x < bw/3:
                                # Zone gauche: diminuer
                                ev -= 1
                                ev = max(ev, pmin)
                            elif click_zone_x < 2*bw/3:
                                # Zone centrale: ouvrir le pavé numérique
                                print("[DEBUG EV] Clic sur zone centrale (main) - ouverture du pavé numérique")
                                new_ev = numeric_input_dialog("Saisir EV (" + str(pmin) + " a " + str(pmax) + ")", ev, pmin, pmax)
                                if new_ev is not None:
                                    ev = int(new_ev)
                                    print(f"[DEBUG EV] Nouvelle valeur EV saisie: {ev}")
                            else:
                                # Zone droite: augmenter
                                ev += 1
                                ev = min(ev, pmax)
                        elif alt_dis > 0 and button_pos == 0:
                            ev -= 1
                            ev = max(ev, pmin)
                        else:
                            ev += 1
                            ev = min(ev, pmax)
                    text(0,2,3,1,1,str(ev),fv,10)
                    draw_bar(0,2,lgrnColor,'ev',ev)
                    time.sleep(0.25)
                    restart = 1

              elif button_row == 3:
                # GAIN
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'gain':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]

                # Vérifier si c'est un clic sur le slider (zone de la barre)
                clicked_on_slider = False
                if (mousex > preview_width and mousex < preview_width + bw and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    clicked_on_slider = True
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    clicked_on_slider = True
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    clicked_on_slider = True

                # Détecter le double-clic sur le slider
                current_time = time.time()
                is_double_click = False
                if clicked_on_slider and last_click_row == button_row:
                    time_diff = current_time - last_click_time
                    print(f"[DEBUG GAIN] Double-clic détecté? time_diff={time_diff:.3f}s, DELAY={DOUBLE_CLICK_DELAY}")
                    if time_diff < DOUBLE_CLICK_DELAY:
                        is_double_click = True
                        print("[DEBUG GAIN] DOUBLE-CLIC CONFIRMÉ!")

                # Mettre à jour les variables de détection du double-clic
                if clicked_on_slider:
                    print(f"[DEBUG GAIN] Clic sur slider détecté, button_row={button_row}")
                    last_click_time = current_time
                    last_click_row = button_row
                    last_click_x = mousex
                    last_click_y = mousey

                # Si double-clic, afficher le dialogue de saisie
                if is_double_click:
                    print("[DEBUG GAIN] Ouverture du dialogue de saisie...")
                    new_gain = numeric_input_dialog("Saisir le Gain (1-" + str(pmax) + ")", gain, 1, pmax)
                    if new_gain is not None:
                        gain = int(new_gain)
                        print(f"[DEBUG GAIN] Nouveau gain saisi: {gain}")
                    # Réinitialiser pour éviter un triple-clic
                    last_click_time = 0
                    last_click_row = -1
                elif clicked_on_slider:
                    # Clic simple sur le slider : ajuster la valeur
                    if (mousex > preview_width and mousex < preview_width + bw and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        slider_val = min(int(((mousex-preview_width) / bw) * (pmax+1-pmin)), pmax)
                        gain = slider_to_gain_nonlinear(slider_val, pmax)
                    elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                        slider_val = min(int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)), pmax)
                        gain = slider_to_gain_nonlinear(slider_val, pmax)
                    elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                        slider_val = min(int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)), pmax)
                        gain = slider_to_gain_nonlinear(slider_val, pmax)
                else:
                    # Clic sur les boutons +/- ou zone centrale (main)
                    # Zone divisée en 3: gauche (0-33%) = "-", centre (33%-66%) = "main", droite (66%-100%) = "+"
                    click_zone_x = mousex - preview_width
                    if alt_dis == 0:
                        if click_zone_x < bw/3:
                            # Zone gauche: diminuer
                            if gain <= 1000:
                                gain = max(gain - 1, 1)
                            else:
                                gain = max(gain - 10, 1000)
                        elif click_zone_x < 2*bw/3:
                            # Zone centrale: ouvrir le pavé numérique
                            print("[DEBUG GAIN] Clic sur zone centrale (main) - ouverture du pavé numérique")
                            new_gain = numeric_input_dialog("Saisir le Gain (1-" + str(pmax) + ")", gain, 1, pmax)
                            if new_gain is not None:
                                gain = int(new_gain)
                                print(f"[DEBUG GAIN] Nouveau gain saisi: {gain}")
                        else:
                            # Zone droite: augmenter
                            if gain < 1000:
                                gain = min(gain + 1, pmax)
                            else:
                                gain = min(gain + 10, pmax)
                    elif alt_dis > 0 and button_pos == 0:
                        if gain <= 1000:
                            gain = max(gain - 1, 1)
                        else:
                            gain = max(gain - 10, 1000)
                    else:
                        if gain < 1000:
                            gain = min(gain + 1, pmax)
                        else:
                            gain = min(gain + 10, pmax)
                if gain > 0:
                    text(0,3,5,0,1,"Gain    A/D",ft,10)
                    if gain <= mag:
                        text(0,3,3,1,1,str(gain) + " :  " + str(gain) + "/1",fv,10)
                    else:
                        text(0,3,3,1,1,str(gain) + " :  " + str(int(mag)) + "/" + str(((gain/mag)*10)/10)[0:3],fv,10)
                else:
                    if gain == 0:
                        text(0,3,5,0,1,"Gain ",ft,10)
                    else:
                        text(0,3,5,0,1,"Gain    A/D",ft,10)
                    text(0,3,3,1,1,"Auto",fv,10)
                time.sleep(.25)
                draw_bar(0,3,lgrnColor,'gain',gain)
                restart = 1
                
              elif button_row == 4:
                # BRIGHTNESS
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'brightness':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    brightness = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) + pmin 
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    brightness = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin
                elif (mousey > preview_height * .75  and mousey < preview_height * .75+ int(bh/3)) and alt_dis == 2:
                    brightness = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin 
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        brightness -=1
                        brightness  = max(brightness ,pmin)
                    else:
                        brightness  +=1
                        brightness = min(brightness ,pmax)
                text(0,4,3,1,1,str(brightness/100),fv,10)
                draw_bar(0,4,lgrnColor,'brightness',brightness)
                time.sleep(0.025)
                restart = 1
                
              elif button_row == 5:
                # CONTRAST
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'contrast':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    contrast = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) 
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    contrast = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    contrast = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        contrast -=1
                        contrast  = max(contrast ,pmin)
                    else:
                        contrast  +=1
                        contrast = min(contrast ,pmax)
                text(0,5,3,1,1,str(contrast/100)[0:4],fv,10)
                draw_bar(0,5,lgrnColor,'contrast',contrast)
                time.sleep(0.025)
                restart = 1
                
              elif button_row == 6:
                # AWB
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'awb':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    awb = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    awb = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    awb = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        awb -=1
                        awb  = max(awb ,pmin)
                    else:
                        awb  +=1
                        awb = min(awb ,pmax)
                text(0,6,3,1,1,awbs[awb],fv,10)
                draw_bar(0,6,lgrnColor,'awb',awb)
                # Pas de restart nécessaire : le fast path applique instantanément via set_controls
                preview()  # Appel direct pour mise à jour immédiate
                time.sleep(.25)
                
              elif button_row == 7:
                # BLUE
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'blue':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    blue = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    blue = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    blue = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        blue -=1
                        blue  = max(blue ,pmin)
                    else:
                        blue  +=1
                        blue = min(blue ,pmax)
                text(0,7,3,1,1,str(blue/10)[0:3],fv,10)
                draw_bar(0,7,lgrnColor,'blue',blue)
                time.sleep(.25)
                restart = 1


              elif button_row == 8 :
                # RED
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'red':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    red = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    red = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    red = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        red -=1
                        red  = max(red ,pmin)
                    else:
                        red  +=1
                        red = min(red ,pmax)
                text(0,8,3,1,1,str(red/10)[0:3],fv,10)
                draw_bar(0,8,lgrnColor,'red',red)
                time.sleep(.25)
                restart = 1
                           
            # MENU 4
            elif menu == 4:
              if button_row == 1:
                  # Page 1 - retour au menu 3
                  menu = 3
                  Menu()

              elif button_row == 2:
                # METER
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'meter':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    meter = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    meter = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    meter = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        meter -=1
                        meter  = max(meter ,pmin)
                    else:
                        meter  +=1
                        meter = min(meter ,pmax)
                text(0,2,3,1,1,meters[meter],fv,10)
                draw_bar(0,2,lgrnColor,'meter',meter)
                time.sleep(.25)
                restart = 1
              
              elif button_row == 3:
                # QUALITY
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'quality':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    quality = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    quality = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    quality = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        quality -=1
                        quality  = max(quality ,pmin)
                    else:
                        quality  +=1
                        quality = min(quality ,pmax)
                text(0,3,3,1,1,str(quality)[0:3],fv,10)
                draw_bar(0,3,lgrnColor,'quality',quality)
                time.sleep(.25)
                restart = 1

              elif button_row == 4:
                # SATURATION
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'saturation':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    saturation = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    saturation = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    saturation = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        saturation -=1
                        saturation  = max(saturation ,pmin)
                    else:
                        saturation  +=1
                        saturation = min(saturation ,pmax)
                text(0,4,3,1,1,str(saturation/10),fv,10)
                draw_bar(0,4,lgrnColor,'saturation',saturation)
                time.sleep(.25)
                restart = 1

              elif button_row == 5:
                # DENOISE
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'denoise':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    denoise = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    denoise = int(((mousex-((button_row -1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    denoise = int(((mousex-((button_row -1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        denoise -=1
                        denoise = max(denoise,pmin)
                    else:
                        denoise +=1
                        denoise = min(denoise,pmax)
                text(0,5,3,1,1,denoises[denoise],fv,10)
                draw_bar(0,5,lgrnColor,'denoise',denoise)
                time.sleep(.25)
                restart = 1

              elif button_row == 6:
                # SHARPNESS
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'sharpness':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    sharpness = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh)  and mousey < preview_height + (bh) + int(bh/3)) and alt_dis == 1:
                    sharpness = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh)  and mousey < preview_height * .75 + (bh) + int(bh/3)) and alt_dis == 2:
                    sharpness = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        sharpness -=1
                        sharpness = max(sharpness,pmin)
                    else:
                        sharpness +=1
                        sharpness = min(sharpness,pmax)

                text(0,6,3,1,1,str(sharpness/10),fv,10)
                draw_bar(0,6,lgrnColor,'sharpness',sharpness)
                time.sleep(.25)
                restart = 1
                
              elif button_row == 7 and Pi_Cam == 9:
                # Waveshare imx290 IR Filter
                if mousex < preview_width + (bw/2):
                    IRF -=1
                    IRF = max(IRF ,0)
                else:
                    IRF  +=1
                    IRF = min(IRF ,1)
                if IRF == 0:
                    text(0,7,3,1,1,"Off",fv,10)
                    led_sw_ir.off()
                else:
                    text(0,7,3,1,1,"ON ",fv,10)
                    led_sw_ir.on()
                time.sleep(0.25)
                restart = 1
                   
              elif button_row == 7 and Pi_Cam == 4 and scientif == 1:
                # v4 (HQ) CAMERA Scientific.json
                if alt_dis == 0 and mousex < preview_width + (bw/2):
                    scientific -=1
                    scientific = max(scientific ,0)
                else:
                    scientific  +=1
                    scientific = min(scientific ,1)
                text(0,7,5,0,1,"Scientific",fv,10)
                if scientific == 0:
                    text(0,7,3,1,1,"Off",fv,10)
                else:
                    text(0,7,3,1,1,"ON ",fv,10)
                time.sleep(0.25)
                restart = 1

                            
              elif button_row == 7 and Pi_Cam == 3:
                # PI V3 CAMERA HDR
                if alt_dis == 0 and mousex < preview_width + (bw/2):
                    v3_hdr -=1
                    v3_hdr  = max(v3_hdr ,0)
                else:
                    v3_hdr  +=1
                    v3_hdr = min(v3_hdr, 4)  # Correction: permettre tous les modes HDR (0-4) incluant "Night" et "Clear HDR"

                text(0,7,5,0,1,"HDR",fv,10)
                text(0,7,3,1,1,v3_hdrs[v3_hdr],fv,10)
                draw_bar(0,7,lgrnColor,'v3_hdr',v3_hdr)
                time.sleep(0.25)
                restart = 1

              elif button_row == 7 and Pi_Cam != 3 and Pi == 5:
                # PI5 and NON V3 CAMERA HDR
                if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                    v3_hdr -=1
                    v3_hdr  = max(v3_hdr ,0)
                else:
                    v3_hdr  +=1
                    v3_hdr = min(v3_hdr, 4)  # Correction: permettre tous les modes HDR (0-4) incluant "Night" et "Clear HDR"

                text(0,7,5,0,1,"HDR",fv,10)
                text(0,7,3,1,1,v3_hdrs[v3_hdr],fv,10)
                draw_bar(0,7,lgrnColor,'v3_hdr',v3_hdr)
                time.sleep(0.25)
                restart = 1

              elif button_row == 8:
                # FILE FORMAT
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'extn':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    extn = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    extn = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    extn = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        extn -=1
                        extn  = max(extn ,pmin)
                    else:
                        extn  +=1
                        extn = min(extn ,pmax)
                text(0,8,3,1,1,extns[extn],fv,10)
                draw_bar(0,8,lgrnColor,'extn',extn)
                time.sleep(.25)
                restart = 1

              elif button_row == 9:
                   # SAVE CONFIG
                   text(0,9,3,0,1,"SAVE Config",fv,10)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = raw_format  # Remplace timet (fixé à 100ms)
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   config[42] = ghs_D
                   config[43] = ghs_b
                   config[44] = ghs_SP
                   config[45] = ghs_LP
                   config[46] = ghs_HP
                   config[47] = ghs_preset
                   config[48] = ls_preview_refresh
                   config[49] = ls_alignment_mode
                   config[50] = ls_enable_qc
                   config[51] = ls_max_fwhm
                   config[52] = ls_min_sharpness
                   config[53] = ls_max_drift
                   config[54] = ls_min_stars
                   config[55] = ls_stack_method
                   config[56] = ls_stack_kappa
                   config[57] = ls_stack_iterations
                   config[58] = ls_planetary_enable
                   config[59] = ls_planetary_mode
                   config[60] = ls_planetary_disk_min
                   config[61] = ls_planetary_disk_max
                   config[62] = ls_planetary_threshold
                   config[63] = ls_planetary_margin
                   config[64] = ls_planetary_ellipse
                   config[65] = ls_planetary_window
                   config[66] = ls_planetary_upsample
                   config[67] = ls_planetary_highpass
                   config[68] = ls_planetary_roi_center
                   config[69] = ls_planetary_corr
                   config[70] = ls_planetary_max_shift
                   config[71] = ls_lucky_buffer
                   config[72] = ls_lucky_keep
                   config[73] = ls_lucky_score
                   config[74] = ls_lucky_stack
                   config[75] = ls_lucky_align
                   config[76] = ls_lucky_roi
                   config[77] = use_native_sensor_mode
                   config[78] = focus_method
                   config[79] = star_metric
                   config[80] = snr_display
                   config[81] = metrics_interval
                   config[82] = ls_lucky_save_progress
                   config[83] = isp_enable
                   config[84] = allsky_mode
                   config[85] = allsky_mean_target
                   config[86] = allsky_mean_threshold
                   config[87] = allsky_video_fps
                   config[88] = allsky_max_gain
                   config[89] = allsky_apply_stretch
                   config[90] = allsky_cleanup_jpegs
                   config[91] = ls_save_progress
                   config[92] = ls_save_final
                   config[93] = ls_lucky_save_final
                   config[94] = fix_bad_pixels
                   config[95] = fix_bad_pixels_sigma
                   config[96] = fix_bad_pixels_min_adu
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,9,2,0,1,"SAVE CONFIG",fv,10)    
                                        
            # MENU 5
            elif menu == 5:   
              if button_row == 1:
                # VIDEO LENGTH
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'vlen':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    vlen = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    vlen = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    vlen = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if mousex < preview_width + (bw/2):
                        vlen -=1
                        vlen  = max(vlen ,pmin)
                    else:
                        vlen  +=1
                        vlen = min(vlen ,pmax)
                td = timedelta(seconds=vlen)
                text(0,1,3,1,1,str(td),fv,11)
                draw_Vbar(0,1,lpurColor,'vlen',vlen)
                time.sleep(.25)
 
              elif button_row == 2:
                # FPS
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'fps':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    fps = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    fps = min(fps,vfps)
                    fps = max(fps,pmin)
                elif (mousey > preview_height  + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    fps = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    fps = min(fps,vfps)
                    fps = max(fps,pmin)
                elif (mousey > preview_height * .75  + (bh*2) and mousey < preview_height  * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    fps = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    fps = min(fps,vfps)
                    fps = max(fps,pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        fps -=1
                        fps  = max(fps ,pmin)
                    else:
                        fps  +=1
                        fps = min(fps ,pmax)
                
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                time.sleep(.25)
                restart = 1
                   
              elif button_row == 3:
                # VFORMAT
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'vformat':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    # Utiliser UNIQUEMENT les résolutions natives (comme le zoom)
                    native_vformats = get_native_vformats()
                    if len(native_vformats) > 0:
                        native_idx = int(((mousex-preview_width) / bw) * len(native_vformats))
                        native_idx = min(native_idx, len(native_vformats) - 1)
                        vformat = native_vformats[native_idx]
                elif (mousey > preview_height  + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    # Utiliser UNIQUEMENT les résolutions natives (comme le zoom)
                    native_vformats = get_native_vformats()
                    if len(native_vformats) > 0:
                        native_idx = int(((mousex-((button_row - 1)*bw)) / bw) * len(native_vformats))
                        native_idx = min(native_idx, len(native_vformats) - 1)
                        vformat = native_vformats[native_idx]
                elif (mousey > preview_height * .75  + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    # Utiliser UNIQUEMENT les résolutions natives (comme le zoom)
                    native_vformats = get_native_vformats()
                    if len(native_vformats) > 0:
                        native_idx = int(((mousex-((button_row - 1)*bw)) / bw) * len(native_vformats))
                        native_idx = min(native_idx, len(native_vformats) - 1)
                        vformat = native_vformats[native_idx]
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        # Passer à la résolution native précédente (comme le zoom)
                        vformat = get_next_native_vformat(vformat, direction=-1)
                    else:
                        # Passer à la résolution native suivante (comme le zoom)
                        vformat = get_next_native_vformat(vformat, direction=1)
                    # set max video format
                    setmaxvformat()
                    vformat = min(vformat,max_vformat)
                draw_Vbar(0,3,lpurColor,'vformat',vformat)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                # Activer l'affichage temporaire du ROI (5 secondes pour compenser le restart)
                show_roi_until = time.time() + 5.0
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 10:
                    vfps = v10_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                # determine if camera native format
                vw = 0
                x = 0
                while x < len(vwidths2) and vw == 0:
                    if vwidth == vwidths2[x]:
                        if vheight == vheights2[x]:
                            vw = 1
                    x += 1
                if vw == 0:
                    text(0,3,3,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                if vw == 1:
                    text(0,3,1,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                time.sleep(.25)
                restart = 1

              elif button_row == 4:
                # CODEC
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'codec':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    codec = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    codec = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    codec = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        codec -=1
                        codec  = max(codec ,pmin)
                    else:
                        codec  +=1
                        codec = min(codec ,pmax)
                # set max video format
                setmaxvformat()
                vformat = min(vformat,max_vformat)
                # S'assurer que vformat est sur une résolution native (comme le zoom)
                vformat = get_next_native_vformat(vformat, direction=0 if vformat in get_native_vformats() else -1)
                text(0,4,3,1,1,codecs[codec],fv,11)
                draw_Vbar(0,4,lpurColor,'codec',codec)
                draw_Vbar(0,3,lpurColor,'vformat',vformat)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                # Activer l'affichage temporaire du ROI (5 secondes pour compenser le restart)
                show_roi_until = time.time() + 5.0
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 10:
                    vfps = v10_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                # determine if camera native format
                vw = 0
                x = 0
                while x < len(vwidths2) and vw == 0:
                    if vwidth == vwidths2[x]:
                        if vheight == vheights2[x]:
                            vw = 1
                    x += 1
                if vw == 0:
                    text(0,3,3,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                if vw == 1:
                    text(0,3,1,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                time.sleep(.25)

              elif button_row == 5:
                # H264 PROFILE
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'profile':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    profile = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    profile = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    profile = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        profile -=1
                        profile  = max(profile ,pmin)
                    else:
                        profile  +=1
                        profile = min(profile ,pmax)
                text(0,5,3,1,1,h264profiles[profile],fv,11)
                draw_Vbar(0,5,lpurColor,'profile',profile)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 10:
                    vfps = v10_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                time.sleep(.25)

              elif button_row == 6:
                # V_PREVIEW
                if (alt_dis == 0 and mousex > preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                    vpreview +=1
                    vpreview  = min(vpreview ,1)
                else:
                    vpreview  -=1
                    vpreview = max(vpreview ,0)

                if vpreview == 0:
                    text(0,7,3,1,1,"Off",fv,11)
                else:
                    text(0,7,3,1,1,"ON ",fv,11)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 10:
                    vfps = v10_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                time.sleep(0.25)
                
              elif button_row == 8:
                   # SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,11)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = raw_format  # Remplace timet (fixé à 100ms)
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   config[42] = ghs_D
                   config[43] = ghs_b
                   config[44] = ghs_SP
                   config[45] = ghs_LP
                   config[46] = ghs_HP
                   config[47] = ghs_preset
                   config[48] = ls_preview_refresh
                   config[49] = ls_alignment_mode
                   config[50] = ls_enable_qc
                   config[51] = ls_max_fwhm
                   config[52] = ls_min_sharpness
                   config[53] = ls_max_drift
                   config[54] = ls_min_stars
                   config[55] = ls_stack_method
                   config[56] = ls_stack_kappa
                   config[57] = ls_stack_iterations
                   config[58] = ls_planetary_enable
                   config[59] = ls_planetary_mode
                   config[60] = ls_planetary_disk_min
                   config[61] = ls_planetary_disk_max
                   config[62] = ls_planetary_threshold
                   config[63] = ls_planetary_margin
                   config[64] = ls_planetary_ellipse
                   config[65] = ls_planetary_window
                   config[66] = ls_planetary_upsample
                   config[67] = ls_planetary_highpass
                   config[68] = ls_planetary_roi_center
                   config[69] = ls_planetary_corr
                   config[70] = ls_planetary_max_shift
                   config[71] = ls_lucky_buffer
                   config[72] = ls_lucky_keep
                   config[73] = ls_lucky_score
                   config[74] = ls_lucky_stack
                   config[75] = ls_lucky_align
                   config[76] = ls_lucky_roi
                   config[77] = use_native_sensor_mode
                   config[78] = focus_method
                   config[79] = star_metric
                   config[80] = snr_display
                   config[81] = metrics_interval
                   config[82] = ls_lucky_save_progress
                   config[83] = isp_enable
                   config[84] = allsky_mode
                   config[85] = allsky_mean_target
                   config[86] = allsky_mean_threshold
                   config[87] = allsky_video_fps
                   config[88] = allsky_max_gain
                   config[89] = allsky_apply_stretch
                   config[90] = allsky_cleanup_jpegs
                   config[91] = ls_save_progress
                   config[92] = ls_save_final
                   config[93] = ls_lucky_save_final
                   config[94] = fix_bad_pixels
                   config[95] = fix_bad_pixels_sigma
                   config[96] = fix_bad_pixels_min_adu
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,11)

            # MENU 6 - TIMELAPSE (Multi-pages)
            elif menu == 6:
              # Get current page
              current_page = menu_page.get(6, 1)

              # ========== PAGE NAVIGATION ==========
              if button_row == 9:
                  if current_page == 1:
                      # Page 1 -> Page 2 (Allsky)
                      menu_page[6] = 2
                      Menu()
                  elif current_page == 2:
                      # Page 2 -> Save Config (button_row 9 on page 2 is SAVE CONFIG)
                      text(0,9,3,0,1,"SAVE Config",fv,10)
                      config[0] = mode
                      config[1] = speed
                      config[2] = gain
                      config[3] = int(brightness)
                      config[4] = int(contrast)
                      config[5] = frame
                      config[6] = int(red)
                      config[7] = int(blue)
                      config[8] = ev
                      config[9] = vlen
                      config[10] = fps
                      config[11] = vformat
                      config[12] = codec
                      config[13] = tinterval
                      config[14] = tshots
                      config[15] = extn
                      config[16] = zx
                      config[17] = zy
                      config[18] = zoom
                      config[19] = int(saturation)
                      config[20] = meter
                      config[21] = awb
                      config[22] = sharpness
                      config[23] = int(denoise)
                      config[24] = quality
                      config[25] = profile
                      config[26] = level
                      config[27] = histogram
                      config[28] = histarea
                      config[29] = v3_f_speed
                      config[30] = v3_f_range
                      config[31] = rotate
                      config[32] = IRF
                      config[33] = str_cap
                      config[34] = v3_hdr
                      config[35] = raw_format
                      config[36] = vflip
                      config[37] = hflip
                      config[38] = stretch_p_low
                      config[39] = stretch_p_high
                      config[40] = stretch_factor
                      config[41] = stretch_preset
                      config[42] = ghs_D
                      config[43] = ghs_b
                      config[44] = ghs_SP
                      config[45] = ghs_LP
                      config[46] = ghs_HP
                      config[47] = ghs_preset
                      config[48] = ls_preview_refresh
                      config[49] = ls_alignment_mode
                      config[50] = ls_enable_qc
                      config[51] = ls_max_fwhm
                      config[52] = ls_min_sharpness
                      config[53] = ls_max_drift
                      config[54] = ls_min_stars
                      config[55] = ls_stack_method
                      config[56] = ls_stack_kappa
                      config[57] = ls_stack_iterations
                      config[58] = ls_planetary_enable
                      config[59] = ls_planetary_mode
                      config[60] = ls_planetary_disk_min
                      config[61] = ls_planetary_disk_max
                      config[62] = ls_planetary_threshold
                      config[63] = ls_planetary_margin
                      config[64] = ls_planetary_ellipse
                      config[65] = ls_planetary_window
                      config[66] = ls_planetary_upsample
                      config[67] = ls_planetary_highpass
                      config[68] = ls_planetary_roi_center
                      config[69] = ls_planetary_corr
                      config[70] = ls_planetary_max_shift
                      config[71] = ls_lucky_buffer
                      config[72] = ls_lucky_keep
                      config[73] = ls_lucky_score
                      config[74] = ls_lucky_stack
                      config[75] = ls_lucky_align
                      config[76] = ls_lucky_roi
                      config[77] = use_native_sensor_mode
                      config[78] = focus_method
                      config[79] = star_metric
                      config[80] = snr_display
                      config[81] = metrics_interval
                      config[82] = ls_lucky_save_progress
                      config[83] = isp_enable
                      # Add allsky parameters
                      config[84] = allsky_mode
                      config[85] = allsky_mean_target
                      config[86] = allsky_mean_threshold
                      config[87] = allsky_video_fps
                      config[88] = allsky_max_gain
                      config[89] = allsky_apply_stretch
                      config[90] = allsky_cleanup_jpegs
                      config[91] = ls_save_progress
                      config[92] = ls_save_final
                      config[93] = ls_lucky_save_final
                      config[94] = fix_bad_pixels
                      config[95] = fix_bad_pixels_sigma
                      config[96] = fix_bad_pixels_min_adu

                      with open(config_file, 'w') as f:
                         for item in range(0,len(titles)):
                             f.write(titles[item] + " : " + str(config[item]) + "\n")
                      time.sleep(1)
                      text(0,9,2,0,1,"SAVE CONFIG",fv,10)

              # ========== PAGE 1 HANDLERS ==========
              if current_page == 1:
                  if button_row == 1:
                    # TIMELAPSE DURATION
                    for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'tduration':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        tduration = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                        tduration = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                    elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                        tduration = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            tduration -=1
                            tduration = max(tduration,pmin)
                        else:
                            tduration +=1
                            tduration = min(tduration,pmax)
                    td = timedelta(seconds=tduration)
                    text(0,1,3,1,1,str(td),fv,12)
                    draw_Vbar(0,1,lyelColor,'tduration',tduration)
                    if tinterval > 0:
                        tshots = int(tduration / tinterval)
                        text(0,3,3,1,1,str(tshots),fv,12)
                    else:
                        text(0,3,3,1,1," ",fv,12)
                    draw_Vbar(0,3,lyelColor,'tshots',tshots)
                    time.sleep(.25)

                  elif button_row == 2:
                    # TIMELAPSE INTERVAL
                    for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'tinterval':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        tinterval = round(((mousex-preview_width) / bw) * (pmax-pmin) + pmin, 2)
                    elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                        tinterval = round(((mousex-((button_row - 9)*bw)) / bw) * (pmax-pmin) + pmin, 2)
                    elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                        tinterval = round(((mousex-((button_row - 9)*bw)) / bw) * (pmax-pmin) + pmin, 2)
                    else:
                        # Pas intelligent: 0.01s si < 1s, sinon 0.1s
                        step = 0.01 if tinterval < 1 else 0.1
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            tinterval = round(tinterval - step, 2)
                            tinterval = max(tinterval, pmin)
                        else:
                            tinterval = round(tinterval + step, 2)
                            tinterval = min(tinterval, pmax)
                    td = timedelta(seconds=tinterval)
                    text(0,2,3,1,1,str(td),fv,12)
                    draw_Vbar(0,2,lyelColor,'tinterval',tinterval)
                    if tinterval != 0:
                        tduration = tinterval * tshots
                        td = timedelta(seconds=tduration)
                        text(0,1,3,1,1,str(td),fv,12)
                        draw_Vbar(0,1,lyelColor,'tduration',tduration)
                    if tinterval == 0:
                        text(0,3,3,1,1," ",fv,12)
                        if mode == 0:
                            speed = 15
                            custom_sspeed = 0  # Réinitialiser car speed a changé
                            shutter = shutters[speed]
                            if shutter < 0:
                                shutter = abs(1/shutter)
                            sspeed = int(shutter * 1000000)
                            if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
                                sspeed +=1
                            restart = 1
                    else:
                        text(0,3,3,1,1,str(tshots),fv,12)
                    time.sleep(.25)
                
                  elif button_row == 3 and tinterval > 0:
                    # TIMELAPSE SHOTS
                    for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'tshots':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        tshots = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                        tshots = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                        tshots = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            tshots -=1
                            tshots = max(tshots,pmin)
                        else:
                            tshots +=1
                            tshots = min(tshots,pmax)
                    text(0,3,3,1,1,str(tshots),fv,12)
                    draw_Vbar(0,3,lyelColor,'tshots',tshots)
                    if tduration > 0:
                        tduration = tinterval * tshots
                    if tduration == 0:
                        tduration = 1
                    td = timedelta(seconds=tduration)
                    text(0,1,3,1,1,str(td),fv,12)
                    draw_Vbar(0,1,lyelColor,'tduration',tduration)
                    time.sleep(.25)
              
                  elif button_row == 8:
                      # SAVE CONFIG (Page 1)
                      text(0,8,3,0,1,"SAVE Config",fv,12)
                      config[0] = mode
                      config[1] = speed
                      config[2] = gain
                      config[3] = int(brightness)
                      config[4] = int(contrast)
                      config[5] = frame
                      config[6] = int(red)
                      config[7] = int(blue)
                      config[8] = ev
                      config[9] = vlen
                      config[10] = fps
                      config[11] = vformat
                      config[12] = codec
                      config[13] = tinterval
                      config[14] = tshots
                      config[15] = extn
                      config[16] = zx
                      config[17] = zy
                      config[18] = zoom
                      config[19] = int(saturation)
                      config[20] = meter
                      config[21] = awb
                      config[22] = sharpness
                      config[23] = int(denoise)
                      config[24] = quality
                      config[25] = profile
                      config[26] = level
                      config[27] = histogram
                      config[28] = histarea
                      config[29] = v3_f_speed
                      config[30] = v3_f_range
                      config[31] = rotate
                      config[32] = IRF
                      config[33] = str_cap
                      config[34] = v3_hdr
                      config[35] = raw_format
                      config[36] = vflip
                      config[37] = hflip
                      config[38] = stretch_p_low
                      config[39] = stretch_p_high
                      config[40] = stretch_factor
                      config[41] = stretch_preset
                      config[42] = ghs_D
                      config[43] = ghs_b
                      config[44] = ghs_SP
                      config[45] = ghs_LP
                      config[46] = ghs_HP
                      config[47] = ghs_preset
                      config[48] = ls_preview_refresh
                      config[49] = ls_alignment_mode
                      config[50] = ls_enable_qc
                      config[51] = ls_max_fwhm
                      config[52] = ls_min_sharpness
                      config[53] = ls_max_drift
                      config[54] = ls_min_stars
                      config[55] = ls_stack_method
                      config[56] = ls_stack_kappa
                      config[57] = ls_stack_iterations
                      config[58] = ls_planetary_enable
                      config[59] = ls_planetary_mode
                      config[60] = ls_planetary_disk_min
                      config[61] = ls_planetary_disk_max
                      config[62] = ls_planetary_threshold
                      config[63] = ls_planetary_margin
                      config[64] = ls_planetary_ellipse
                      config[65] = ls_planetary_window
                      config[66] = ls_planetary_upsample
                      config[67] = ls_planetary_highpass
                      config[68] = ls_planetary_roi_center
                      config[69] = ls_planetary_corr
                      config[70] = ls_planetary_max_shift
                      config[71] = ls_lucky_buffer
                      config[72] = ls_lucky_keep
                      config[73] = ls_lucky_score
                      config[74] = ls_lucky_stack
                      config[75] = ls_lucky_align
                      config[76] = ls_lucky_roi
                      config[77] = use_native_sensor_mode
                      config[78] = focus_method
                      config[79] = star_metric
                      config[80] = snr_display
                      config[81] = metrics_interval
                      config[82] = ls_lucky_save_progress
                      config[83] = isp_enable
                      # Add allsky parameters
                      config[84] = allsky_mode
                      config[85] = allsky_mean_target
                      config[86] = allsky_mean_threshold
                      config[87] = allsky_video_fps
                      config[88] = allsky_max_gain
                      config[89] = allsky_apply_stretch
                      config[90] = allsky_cleanup_jpegs
                      config[91] = ls_save_progress
                      config[92] = ls_save_final
                      config[93] = ls_lucky_save_final
                      config[94] = fix_bad_pixels
                      config[95] = fix_bad_pixels_sigma
                      config[96] = fix_bad_pixels_min_adu
                      with open(config_file, 'w') as f:
                          for item in range(0,len(titles)):
                              f.write(titles[item] + " : " + str(config[item]) + "\n")
                      time.sleep(1)
                      text(0,8,2,0,1,"SAVE CONFIG",fv,12)

              # ========== PAGE 2 HANDLERS (ALLSKY) ==========
              elif current_page == 2:
                  if button_row == 1:
                      # Navigation back to Page 1
                      menu_page[6] = 1
                      Menu()

                  elif button_row == 2:
                      # ALLSKY MODE (0=OFF, 1=ON, 2=Auto-Gain)
                      for f in range(0,len(video_limits)-1,3):
                          if video_limits[f] == 'allsky_mode':
                              pmin = video_limits[f+1]
                              pmax = video_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          allsky_mode = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              allsky_mode = max(pmin, allsky_mode - 1)
                          else:
                              allsky_mode = min(pmax, allsky_mode + 1)
                      text(0,2,3,1,1,allsky_modes[allsky_mode],fv,10)
                      draw_Vbar(0,2,greyColor,'allsky_mode',allsky_mode)
                      Menu()  # Refresh to update N/A fields
                      time.sleep(.25)

                  elif button_row == 3 and allsky_mode == 2:
                      # MEAN TARGET (10-60 → 0.10-0.60)
                      for f in range(0,len(video_limits)-1,3):
                          if video_limits[f] == 'allsky_mean_target':
                              pmin = video_limits[f+1]
                              pmax = video_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          allsky_mean_target = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              allsky_mean_target = max(pmin, allsky_mean_target - 1)
                          else:
                              allsky_mean_target = min(pmax, allsky_mean_target + 1)
                      text(0,3,3,1,1,str(allsky_mean_target/100.0)[0:4],fv,10)
                      draw_Vbar(0,3,greyColor,'allsky_mean_target',allsky_mean_target)
                      time.sleep(.05)

                  elif button_row == 4 and allsky_mode == 2:
                      # MEAN THRESHOLD (2-15 → 0.02-0.15)
                      for f in range(0,len(video_limits)-1,3):
                          if video_limits[f] == 'allsky_mean_threshold':
                              pmin = video_limits[f+1]
                              pmax = video_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          allsky_mean_threshold = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              allsky_mean_threshold = max(pmin, allsky_mean_threshold - 1)
                          else:
                              allsky_mean_threshold = min(pmax, allsky_mean_threshold + 1)
                      text(0,4,3,1,1,str(allsky_mean_threshold/100.0)[0:4],fv,10)
                      draw_Vbar(0,4,greyColor,'allsky_mean_threshold',allsky_mean_threshold)
                      time.sleep(.05)

                  elif button_row == 5 and allsky_mode > 0:
                      # VIDEO FPS (15-60)
                      for f in range(0,len(video_limits)-1,3):
                          if video_limits[f] == 'allsky_video_fps':
                              pmin = video_limits[f+1]
                              pmax = video_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          allsky_video_fps = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              allsky_video_fps = max(pmin, allsky_video_fps - 1)
                          else:
                              allsky_video_fps = min(pmax, allsky_video_fps + 1)
                      text(0,5,3,1,1,str(allsky_video_fps),fv,10)
                      draw_Vbar(0,5,greyColor,'allsky_video_fps',allsky_video_fps)
                      time.sleep(.25)

                  elif button_row == 6 and allsky_mode == 2:
                      # MAX GAIN (50-500, step=10)
                      for f in range(0,len(video_limits)-1,3):
                          if video_limits[f] == 'allsky_max_gain':
                              pmin = video_limits[f+1]
                              pmax = video_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          allsky_max_gain = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          step = 10
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              allsky_max_gain = max(pmin, allsky_max_gain - step)
                          else:
                              allsky_max_gain = min(pmax, allsky_max_gain + step)
                      text(0,6,3,1,1,str(allsky_max_gain),fv,10)
                      draw_Vbar(0,6,greyColor,'allsky_max_gain',allsky_max_gain)
                      time.sleep(.25)

                  elif button_row == 7 and allsky_mode > 0:
                      # APPLY STRETCH (toggle)
                      allsky_apply_stretch = 1 - allsky_apply_stretch
                      stretch_text = "ON" if allsky_apply_stretch == 1 else "OFF"
                      text(0,7,3,1,1,stretch_text,fv,10)
                      draw_Vbar(0,7,greyColor,'allsky_apply_stretch',allsky_apply_stretch)
                      time.sleep(.25)

                  elif button_row == 8 and allsky_mode > 0:
                      # CLEANUP JPEGs (toggle)
                      allsky_cleanup_jpegs = 1 - allsky_cleanup_jpegs
                      cleanup_text = "YES" if allsky_cleanup_jpegs == 1 else "NO"
                      text(0,8,3,1,1,cleanup_text,fv,10)
                      draw_Vbar(0,8,greyColor,'allsky_cleanup_jpegs',allsky_cleanup_jpegs)
                      time.sleep(.25)

            # MENU 7 - STRETCH Settings
            elif menu == 7:
              # Gestion multi-pages - Phase 2
              current_page = menu_page.get(7, 1)

              if button_row == 9:
                  if current_page == 1 and stretch_preset == 1:
                      # Page 1 + GHS sélectionné : aller vers Page 2
                      menu_page[7] = 2
                      Menu()
                  elif current_page == 2:
                      # Page 2 : retour vers Page 1
                      menu_page[7] = 1
                      Menu()
                  else:
                      # Page 1 + pas GHS : retour CAMERA Settings
                      menu = 1
                      Menu()

              elif button_row == 1:
                if current_page == 1:
                    # PAGE 1 - STRETCH LOW PERCENTILE (0% à 0.2%, stocké x10)
                    pmin = 0    # 0%
                    pmax = 2    # 0.2%
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        stretch_p_low = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            stretch_p_low = max(0, stretch_p_low - 1)
                        else:
                            stretch_p_low = min(pmax, stretch_p_low + 1)
                    text(0,1,3,1,1,str(stretch_p_low/10)[0:4],fv,7)
                    draw_Vbar(0,1,greyColor,'stretch_p_low',stretch_p_low)
                    time.sleep(.05)
                elif current_page == 2:
                    # PAGE 2 - Navigation retour (déjà géré par button_row == 9)
                    # Clic sur ligne 1 = retour Page 1
                    menu_page[7] = 1
                    Menu()

              elif button_row == 2:
                if current_page == 1:
                    # PAGE 1 - STRETCH HIGH PERCENTILE (99.95% à 100%, stocké x100)
                    pmin = 9995  # 99.95%
                    pmax = 10000 # 100%
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        stretch_p_high = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            stretch_p_high = max(pmin, stretch_p_high - 1)
                        else:
                            stretch_p_high = min(pmax, stretch_p_high + 1)
                    text(0,2,3,1,1,str(stretch_p_high/100)[0:6],fv,7)
                    draw_Vbar(0,2,greyColor,'stretch_p_high',stretch_p_high)
                    time.sleep(.05)
                elif current_page == 2:
                    # PAGE 2 - GHS D (Force) : -10 à 100 -> -1.0 à 10.0
                    for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'ghs_D':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        ghs_D = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            ghs_D -= 1
                            ghs_D = max(ghs_D, pmin)
                        else:
                            ghs_D += 1
                            ghs_D = min(ghs_D, pmax)

                    # Passage automatique en mode Manual
                    ghs_preset = 0

                    text(0,2,3,1,1,str(ghs_D/10.0)[0:5],fv,7)
                    draw_Vbar(0,2,greyColor,'ghs_D',ghs_D)
                    text(0,7,3,1,1,ghs_presets[ghs_preset],fv,7)
                    draw_Vbar(0,7,greyColor,'ghs_preset',ghs_preset)
                    # Mettre à jour livestack/luckystack si actif
                    if livestack is not None:
                        livestack.configure(ghs_D=ghs_D / 10.0)
                    if luckystack is not None:
                        luckystack.configure(ghs_D=ghs_D / 10.0)
                    time.sleep(.05)

              elif button_row == 3:
                # Ligne 3: Page 1 = STRETCH FACTOR, Page 2 = GHS b (Local intensity)
                current_page = menu_page.get(7, 1)

                if current_page == 1:
                    # PAGE 1: STRETCH FACTOR (0 à 5, stocké x10)
                    pmin = 0    # 0.0
                    pmax = 80   # 8.0 (augmenté pour plus de flexibilité)
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        stretch_factor = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            stretch_factor = max(pmin, stretch_factor - 1)
                        else:
                            stretch_factor = min(pmax, stretch_factor + 1)
                    # Note: modification manuelle des paramètres - le preset affiché reste inchangé
                    text(0,3,3,1,1,str(stretch_factor/10)[0:4],fv,7)
                    draw_Vbar(0,3,greyColor,'stretch_factor',stretch_factor)
                    text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
                    draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)

                elif current_page == 2:
                    # PAGE 2: GHS b - Local intensity (-5.0 à 15.0, stocké x10)
                    pmin = -50   # -5.0
                    pmax = 150   # 15.0
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        ghs_b = int(((mousex-preview_width) / bw) * (pmax-pmin+1) + pmin)
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            ghs_b = max(pmin, ghs_b - 1)
                        else:
                            ghs_b = min(pmax, ghs_b + 1)

                    # Passage automatique en mode Manual
                    ghs_preset = 0

                    text(0,3,3,1,1,str(ghs_b/10.0)[0:5],fv,7)
                    draw_Vbar(0,3,greyColor,'ghs_b',ghs_b)
                    text(0,7,3,1,1,ghs_presets[ghs_preset],fv,7)
                    draw_Vbar(0,7,greyColor,'ghs_preset',ghs_preset)
                    # Mettre à jour livestack/luckystack si actif
                    if livestack is not None:
                        livestack.configure(ghs_b=ghs_b / 10.0)
                    if luckystack is not None:
                        luckystack.configure(ghs_b=ghs_b / 10.0)

                time.sleep(.05)

              elif button_row == 4:
                # Ligne 4: Page 1 = STRETCH PRESET, Page 2 = GHS SP (Symmetry point)
                current_page = menu_page.get(7, 1)

                if current_page == 1:
                    # PAGE 1: STRETCH PRESET
                    pmin = 0
                    pmax = 2  # OFF/GHS/Arcsinh
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        stretch_preset = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            stretch_preset -=1
                            stretch_preset = max(stretch_preset,pmin)
                        else:
                            stretch_preset +=1
                            stretch_preset = min(stretch_preset,pmax)

                    # Charger les valeurs du preset
                    if stretch_preset == 0:
                        # OFF - Pas de stretch
                        pass
                    elif stretch_preset == 1:
                        # GHS - Paramètres optimisés
                        ghs_D = 31   # 3.1
                        ghs_b = 1    # 0.1
                        ghs_SP = 19  # 0.19
                    elif stretch_preset == 2:
                        # Arcsinh - Paramètres par défaut
                        stretch_p_low = 0      # 0%
                        stretch_p_high = 9998  # 99.98%
                        stretch_factor = 25    # 2.5

                    # Mettre à jour l'affichage
                    text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
                    draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)

                    # Mettre à jour l'affichage des paramètres selon le preset
                    if stretch_preset == 1:  # GHS
                        text(0,5,3,1,1,str(ghs_D/10.0)[0:5],fv,7)
                        draw_Vbar(0,5,greyColor,'ghs_D',ghs_D)
                        text(0,6,3,1,1,str(ghs_b/10.0)[0:6],fv,7)
                        draw_Vbar(0,6,greyColor,'ghs_b',ghs_b)
                        text(0,7,3,1,1,str(ghs_SP/100.0)[0:5],fv,7)
                        draw_Vbar(0,7,greyColor,'ghs_SP',ghs_SP)
                    elif stretch_preset == 2:  # Arcsinh
                        text(0,1,3,1,1,str(stretch_p_low/10)[0:4],fv,7)
                        draw_Vbar(0,1,greyColor,'stretch_p_low',stretch_p_low)
                        text(0,2,3,1,1,str(stretch_p_high/100)[0:6],fv,7)
                        draw_Vbar(0,2,greyColor,'stretch_p_high',stretch_p_high)
                        text(0,3,3,1,1,str(stretch_factor/10)[0:4],fv,7)
                        draw_Vbar(0,3,greyColor,'stretch_factor',stretch_factor)

                    Menu()  # Redessiner le menu pour afficher/cacher les sliders GHS

                    # Mettre à jour livestack/luckystack avec le nouveau mode de stretch
                    if livestack is not None:
                        livestack.configure(
                            png_stretch=['off', 'ghs', 'asinh'][stretch_preset],
                            png_clip_low=0.0 if stretch_preset == 1 else stretch_p_low / 10.0,
                            png_clip_high=100.0 if stretch_preset == 1 else stretch_p_high / 100.0
                        )
                    if luckystack is not None:
                        luckystack.configure(
                            png_stretch=['off', 'ghs', 'asinh'][stretch_preset],
                            png_clip_low=0.0 if stretch_preset == 1 else stretch_p_low / 10.0,
                            png_clip_high=100.0 if stretch_preset == 1 else stretch_p_high / 100.0
                        )

                elif current_page == 2:
                    # PAGE 2: GHS SP - Symmetry point (0.0 à 1.0, stocké x100)
                    pmin = 0    # 0.0
                    pmax = 100  # 1.0
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        ghs_SP = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            ghs_SP = max(pmin, ghs_SP - 1)
                        else:
                            ghs_SP = min(pmax, ghs_SP + 1)

                    # Contraintes: LP <= SP et HP >= SP
                    if ghs_LP > ghs_SP:
                        ghs_LP = ghs_SP
                    if ghs_HP < ghs_SP:
                        ghs_HP = ghs_SP

                    # Passage automatique en mode Manual
                    ghs_preset = 0

                    text(0,4,3,1,1,str(ghs_SP/100.0)[0:5],fv,7)
                    draw_Vbar(0,4,greyColor,'ghs_SP',ghs_SP)
                    text(0,7,3,1,1,ghs_presets[ghs_preset],fv,7)
                    draw_Vbar(0,7,greyColor,'ghs_preset',ghs_preset)
                    # Mettre à jour livestack/luckystack si actif (SP peut modifier LP/HP via contraintes)
                    if livestack is not None:
                        livestack.configure(ghs_SP=ghs_SP / 100.0, ghs_LP=ghs_LP / 100.0, ghs_HP=ghs_HP / 100.0)
                    if luckystack is not None:
                        luckystack.configure(ghs_SP=ghs_SP / 100.0, ghs_LP=ghs_LP / 100.0, ghs_HP=ghs_HP / 100.0)

                time.sleep(.05)

              elif button_row == 5:
                # Ligne 5: Page 1 = (vide), Page 2 = GHS LP (Protect shadows)
                current_page = menu_page.get(7, 1)

                if current_page == 2:
                    # PAGE 2: GHS LP - Protect shadows (0.0 à 1.0, stocké x100)
                    # Contrainte: LP <= SP
                    pmin = 0          # 0.0
                    pmax = ghs_SP     # Limité par SP
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        ghs_LP = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            ghs_LP = max(pmin, ghs_LP - 1)
                        else:
                            ghs_LP = min(pmax, ghs_LP + 1)

                    # Forcer LP <= SP
                    if ghs_LP > ghs_SP:
                        ghs_LP = ghs_SP

                    # Passage automatique en mode Manual
                    ghs_preset = 0

                    text(0,5,3,1,1,str(ghs_LP/100.0)[0:5],fv,7)
                    draw_Vbar(0,5,greyColor,'ghs_LP',ghs_LP)
                    text(0,7,3,1,1,ghs_presets[ghs_preset],fv,7)
                    draw_Vbar(0,7,greyColor,'ghs_preset',ghs_preset)
                    # Mettre à jour livestack/luckystack si actif
                    if livestack is not None:
                        livestack.configure(ghs_LP=ghs_LP / 100.0)
                    if luckystack is not None:
                        luckystack.configure(ghs_LP=ghs_LP / 100.0)
                    time.sleep(.05)

              elif button_row == 6:
                # Ligne 6: Page 1 = (vide), Page 2 = GHS HP (Protect highlights)
                current_page = menu_page.get(7, 1)

                if current_page == 2:
                    # PAGE 2: GHS HP - Protect highlights (0.0 à 1.0, stocké x100)
                    # Note: HP = 0 désactive la protection (sera contraint à SP dans ghs_stretch)
                    pmin = 0          # Permet 0 pour désactiver la protection
                    pmax = 100        # 1.0
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        ghs_HP = int(((mousex-preview_width) / bw) * (pmax-pmin+1) + pmin)
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            ghs_HP = max(pmin, ghs_HP - 1)
                        else:
                            ghs_HP = min(pmax, ghs_HP + 1)

                    # Note: On autorise HP < SP (sera géré par ghs_stretch qui applique max(SP, HP))
                    # HP = 0 signifie "pas de protection highlights"

                    # Passage automatique en mode Manual
                    ghs_preset = 0

                    text(0,6,3,1,1,str(ghs_HP/100.0)[0:5],fv,7)
                    draw_Vbar(0,6,greyColor,'ghs_HP',ghs_HP)
                    text(0,7,3,1,1,ghs_presets[ghs_preset],fv,7)
                    draw_Vbar(0,7,greyColor,'ghs_preset',ghs_preset)
                    # Mettre à jour livestack/luckystack si actif
                    if livestack is not None:
                        livestack.configure(ghs_HP=ghs_HP / 100.0)
                    if luckystack is not None:
                        luckystack.configure(ghs_HP=ghs_HP / 100.0)
                    time.sleep(.05)

              elif button_row == 7:
                # Ligne 7: Page 1 = (vide), Page 2 = GHS Preset
                current_page = menu_page.get(7, 1)

                if current_page == 2:
                    # PAGE 2: GHS Preset - Charger valeurs prédéfinies
                    pmin = 0
                    pmax = 3  # 0=Manual, 1=Galaxies, 2=Nébuleuses, 3=Étirement initial
                    if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                        ghs_preset = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            ghs_preset -=1
                            ghs_preset = max(ghs_preset,pmin)
                        else:
                            ghs_preset +=1
                            ghs_preset = min(ghs_preset,pmax)

                    # Charger les valeurs du preset GHS
                    if ghs_preset == 1:
                        # Galaxies (ciel profond) - Paramètres optimisés tests M81
                        ghs_D = 31    # 3.1
                        ghs_b = 1     # 0.1
                        ghs_SP = 19   # 0.19
                        ghs_LP = 0    # 0.0
                        ghs_HP = 0    # 0.0
                    elif ghs_preset == 2:
                        # Nébuleuses (émission) - Extrapolé depuis optimisation Galaxies
                        ghs_D = 39    # 3.9 (2.5 * 1.55)
                        ghs_b = 1     # 0.1 (même réduction drastique que Galaxies)
                        ghs_SP = 13   # 0.13 (0.10 * 1.27)
                        ghs_LP = 0    # 0.0 (suppression comme Galaxies)
                        ghs_HP = 0    # 0.0 (suppression comme Galaxies)
                    elif ghs_preset == 3:
                        # Étirement initial (linéaire -> non-linéaire) - Extrapolé depuis optimisation Galaxies
                        ghs_D = 54    # 5.4 (3.5 * 1.55, mais limité à 5.0 max pratique)
                        ghs_b = 2     # 0.2 (même réduction drastique que Galaxies)
                        ghs_SP = 10   # 0.10 (0.08 * 1.27)
                        ghs_LP = 0    # 0.0
                        ghs_HP = 0    # 0.0 (suppression comme Galaxies)
                    # Si ghs_preset == 0 (Manual), ne rien changer

                    # Mettre à jour l'affichage
                    text(0,7,3,1,1,ghs_presets[ghs_preset],fv,7)
                    draw_Vbar(0,7,greyColor,'ghs_preset',ghs_preset)

                    # Mettre à jour l'affichage de tous les paramètres GHS (UNIQUEMENT si Page 2)
                    if current_page == 2 and ghs_preset > 0:  # Si Page 2 ET preset non-Manual
                        text(0,2,3,1,1,str(ghs_D/10.0)[0:5],fv,7)
                        draw_Vbar(0,2,greyColor,'ghs_D',ghs_D)
                        text(0,3,3,1,1,str(ghs_b/10.0)[0:5],fv,7)
                        draw_Vbar(0,3,greyColor,'ghs_b',ghs_b)
                        text(0,4,3,1,1,str(ghs_SP/100.0)[0:5],fv,7)
                        draw_Vbar(0,4,greyColor,'ghs_SP',ghs_SP)
                        text(0,5,3,1,1,str(ghs_LP/100.0)[0:5],fv,7)
                        draw_Vbar(0,5,greyColor,'ghs_LP',ghs_LP)
                        text(0,6,3,1,1,str(ghs_HP/100.0)[0:5],fv,7)
                        draw_Vbar(0,6,greyColor,'ghs_HP',ghs_HP)
                        # Mettre à jour livestack/luckystack avec tous les paramètres GHS
                        if livestack is not None:
                            livestack.configure(ghs_D=ghs_D/10.0, ghs_b=ghs_b/10.0, ghs_SP=ghs_SP/100.0, ghs_LP=ghs_LP/100.0, ghs_HP=ghs_HP/100.0)
                        if luckystack is not None:
                            luckystack.configure(ghs_D=ghs_D/10.0, ghs_b=ghs_b/10.0, ghs_SP=ghs_SP/100.0, ghs_LP=ghs_LP/100.0, ghs_HP=ghs_HP/100.0)

                    time.sleep(.05)

              elif button_row == 8:
                   # SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,7)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = raw_format  # Remplace timet (fixé à 100ms)
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   config[42] = ghs_D
                   config[43] = ghs_b
                   config[44] = ghs_SP
                   config[45] = ghs_LP
                   config[46] = ghs_HP
                   config[47] = ghs_preset
                   config[48] = ls_preview_refresh
                   config[49] = ls_alignment_mode
                   config[50] = ls_enable_qc
                   config[51] = ls_max_fwhm
                   config[52] = ls_min_sharpness
                   config[53] = ls_max_drift
                   config[54] = ls_min_stars
                   config[55] = ls_stack_method
                   config[56] = ls_stack_kappa
                   config[57] = ls_stack_iterations
                   config[58] = ls_planetary_enable
                   config[59] = ls_planetary_mode
                   config[60] = ls_planetary_disk_min
                   config[61] = ls_planetary_disk_max
                   config[62] = ls_planetary_threshold
                   config[63] = ls_planetary_margin
                   config[64] = ls_planetary_ellipse
                   config[65] = ls_planetary_window
                   config[66] = ls_planetary_upsample
                   config[67] = ls_planetary_highpass
                   config[68] = ls_planetary_roi_center
                   config[69] = ls_planetary_corr
                   config[70] = ls_planetary_max_shift
                   config[71] = ls_lucky_buffer
                   config[72] = ls_lucky_keep
                   config[73] = ls_lucky_score
                   config[74] = ls_lucky_stack
                   config[75] = ls_lucky_align
                   config[76] = ls_lucky_roi
                   config[77] = use_native_sensor_mode
                   config[78] = focus_method
                   config[79] = star_metric
                   config[80] = snr_display
                   config[81] = metrics_interval
                   config[82] = ls_lucky_save_progress
                   config[83] = isp_enable
                   config[84] = allsky_mode
                   config[85] = allsky_mean_target
                   config[86] = allsky_mean_threshold
                   config[87] = allsky_video_fps
                   config[88] = allsky_max_gain
                   config[89] = allsky_apply_stretch
                   config[90] = allsky_cleanup_jpegs
                   config[91] = ls_save_progress
                   config[92] = ls_save_final
                   config[93] = ls_lucky_save_final
                   config[94] = fix_bad_pixels
                   config[95] = fix_bad_pixels_sigma
                   config[96] = fix_bad_pixels_min_adu
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,7)

            elif menu == 8:
              # LIVE STACK Settings - Gestion des clics
              current_page = menu_page.get(8, 1)

              # Page 2 : Gestion des paramètres Stacker + Planetary
              if current_page == 2 and button_row >= 1 and button_row <= 8:
                  if button_row == 1:
                      # STACK METHOD (0-4: mean/median/kappa/winsorized/weighted)
                      for f in range(0,len(livestack_limits)-1,3):
                          if livestack_limits[f] == 'ls_stack_method':
                              pmin = livestack_limits[f+1]
                              pmax = livestack_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          ls_stack_method = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              ls_stack_method -=1
                              ls_stack_method = max(ls_stack_method,pmin)
                          else:
                              ls_stack_method +=1
                              ls_stack_method = min(ls_stack_method,pmax)
                      text(0,1,3,1,1,stack_methods[ls_stack_method],fv,7)
                      draw_Vbar(0,1,greyColor,'ls_stack_method',ls_stack_method)
                      # Mettre à jour la config livestack en temps réel
                      if livestack is not None:
                          method_names = ['mean', 'median', 'kappa_sigma', 'winsorized', 'weighted']
                          livestack.configure(stacking_method=method_names[ls_stack_method])
                      time.sleep(.05)

                  elif button_row == 2:
                      # STACK KAPPA (10-40 affiché 1.0-4.0)
                      for f in range(0,len(livestack_limits)-1,3):
                          if livestack_limits[f] == 'ls_stack_kappa':
                              pmin = livestack_limits[f+1]
                              pmax = livestack_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          ls_stack_kappa = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              ls_stack_kappa -=5
                              ls_stack_kappa = max(ls_stack_kappa,pmin)
                          else:
                              ls_stack_kappa +=5
                              ls_stack_kappa = min(ls_stack_kappa,pmax)
                      text(0,2,3,1,1,str(ls_stack_kappa/10.0)[0:4],fv,7)
                      draw_Vbar(0,2,greyColor,'ls_stack_kappa',ls_stack_kappa)
                      # Mettre à jour la config livestack en temps réel
                      if livestack is not None:
                          livestack.configure(kappa=ls_stack_kappa / 10.0)
                      time.sleep(.05)

                  elif button_row == 3:
                      # STACK ITERATIONS (1-10)
                      for f in range(0,len(livestack_limits)-1,3):
                          if livestack_limits[f] == 'ls_stack_iterations':
                              pmin = livestack_limits[f+1]
                              pmax = livestack_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          ls_stack_iterations = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              ls_stack_iterations -=1
                              ls_stack_iterations = max(ls_stack_iterations,pmin)
                          else:
                              ls_stack_iterations +=1
                              ls_stack_iterations = min(ls_stack_iterations,pmax)
                      text(0,3,3,1,1,str(ls_stack_iterations),fv,7)
                      draw_Vbar(0,3,greyColor,'ls_stack_iterations',ls_stack_iterations)
                      # Mettre à jour la config livestack en temps réel
                      if livestack is not None:
                          livestack.configure(iterations=ls_stack_iterations)
                      time.sleep(.05)

                  elif button_row == 4:
                      # PLANETARY ENABLE (toggle)
                      ls_planetary_enable = 1 - ls_planetary_enable
                      if ls_planetary_enable == 0:
                          text(0,4,3,1,1,"OFF",fv,7)
                      else:
                          text(0,4,3,1,1,"ON",fv,7)
                      draw_Vbar(0,4,greyColor,'ls_planetary_enable',ls_planetary_enable)
                      # Mettre à jour la config livestack en temps réel
                      if livestack is not None:
                          livestack.configure(planetary_enable=bool(ls_planetary_enable))
                      time.sleep(.05)

                  elif button_row == 5:
                      # PLANETARY MODE (0-2: disk/surface/hybrid)
                      for f in range(0,len(livestack_limits)-1,3):
                          if livestack_limits[f] == 'ls_planetary_mode':
                              pmin = livestack_limits[f+1]
                              pmax = livestack_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          ls_planetary_mode = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              ls_planetary_mode -=1
                              ls_planetary_mode = max(ls_planetary_mode,pmin)
                          else:
                              ls_planetary_mode +=1
                              ls_planetary_mode = min(ls_planetary_mode,pmax)
                      text(0,5,3,1,1,planetary_modes[ls_planetary_mode],fv,7)
                      draw_Vbar(0,5,greyColor,'ls_planetary_mode',ls_planetary_mode)
                      # Mettre à jour la config livestack en temps réel
                      if livestack is not None:
                          mode_names = ['disk', 'surface', 'hybrid']
                          livestack.configure(planetary_mode=mode_names[ls_planetary_mode])
                      time.sleep(.05)

                  elif button_row == 6:
                      # PLANETARY DISK MIN (20-500)
                      for f in range(0,len(livestack_limits)-1,3):
                          if livestack_limits[f] == 'ls_planetary_disk_min':
                              pmin = livestack_limits[f+1]
                              pmax = livestack_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          ls_planetary_disk_min = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              ls_planetary_disk_min -=10
                              ls_planetary_disk_min = max(ls_planetary_disk_min,pmin)
                          else:
                              ls_planetary_disk_min +=10
                              ls_planetary_disk_min = min(ls_planetary_disk_min,pmax)
                      text(0,6,3,1,1,str(ls_planetary_disk_min),fv,7)
                      draw_Vbar(0,6,greyColor,'ls_planetary_disk_min',ls_planetary_disk_min)
                      # Mettre à jour la config livestack en temps réel
                      if livestack is not None:
                          livestack.configure(planetary_disk_min=ls_planetary_disk_min)
                      time.sleep(.05)

                  elif button_row == 7:
                      # PLANETARY DISK MAX (100-2000)
                      for f in range(0,len(livestack_limits)-1,3):
                          if livestack_limits[f] == 'ls_planetary_disk_max':
                              pmin = livestack_limits[f+1]
                              pmax = livestack_limits[f+2]
                      if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                          ls_planetary_disk_max = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                      else:
                          if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                              ls_planetary_disk_max -=50
                              ls_planetary_disk_max = max(ls_planetary_disk_max,pmin)
                          else:
                              ls_planetary_disk_max +=50
                              ls_planetary_disk_max = min(ls_planetary_disk_max,pmax)
                      text(0,7,3,1,1,str(ls_planetary_disk_max),fv,7)
                      draw_Vbar(0,7,greyColor,'ls_planetary_disk_max',ls_planetary_disk_max)
                      # Mettre à jour la config livestack en temps réel
                      if livestack is not None:
                          livestack.configure(planetary_disk_max=ls_planetary_disk_max)
                      time.sleep(.05)

                  elif button_row == 8:
                      # SAVE CONFIG (Page 2 du menu 8)
                      config[0] = mode
                      config[1] = speed
                      config[2] = gain
                      config[3] = brightness
                      config[4] = contrast
                      config[5] = frame
                      config[6] = red
                      config[7] = blue
                      config[8] = ev
                      config[9] = vlen
                      config[10] = fps
                      config[11] = vformat
                      config[12] = codec
                      config[13] = tinterval
                      config[14] = tshots
                      config[15] = extn
                      config[16] = zx
                      config[17] = zy
                      config[18] = zoom
                      config[19] = saturation
                      config[20] = meter
                      config[21] = awb
                      config[22] = sharpness
                      config[23] = denoise
                      config[24] = quality
                      config[25] = profile
                      config[26] = level
                      config[27] = histogram
                      config[28] = histarea
                      config[29] = v3_f_speed
                      config[30] = v3_f_range
                      config[31] = rotate
                      config[32] = IRF
                      config[33] = str_cap
                      config[34] = v3_hdr
                      config[35] = raw_format  # Remplace timet (fixé à 100ms)
                      config[36] = vflip
                      config[37] = hflip
                      config[38] = stretch_p_low
                      config[39] = stretch_p_high
                      config[40] = stretch_factor
                      config[41] = stretch_preset
                      config[42] = ghs_D
                      config[43] = ghs_b
                      config[44] = ghs_SP
                      config[45] = ghs_LP
                      config[46] = ghs_HP
                      config[47] = ghs_preset
                      config[48] = ls_preview_refresh
                      config[49] = ls_alignment_mode
                      config[50] = ls_enable_qc
                      config[51] = ls_max_fwhm
                      config[52] = ls_min_sharpness
                      config[53] = ls_max_drift
                      config[54] = ls_min_stars
                      config[55] = ls_stack_method
                      config[56] = ls_stack_kappa
                      config[57] = ls_stack_iterations
                      config[58] = ls_planetary_enable
                      config[59] = ls_planetary_mode
                      config[60] = ls_planetary_disk_min
                      config[61] = ls_planetary_disk_max
                      config[62] = ls_planetary_threshold
                      config[63] = ls_planetary_margin
                      config[64] = ls_planetary_ellipse
                      config[65] = ls_planetary_window
                      config[66] = ls_planetary_upsample
                      config[67] = ls_planetary_highpass
                      config[68] = ls_planetary_roi_center
                      config[69] = ls_planetary_corr
                      config[70] = ls_planetary_max_shift
                      config[71] = ls_lucky_buffer
                      config[72] = ls_lucky_keep
                      config[73] = ls_lucky_score
                      config[74] = ls_lucky_stack
                      config[75] = ls_lucky_align
                      config[76] = ls_lucky_roi
                      config[77] = use_native_sensor_mode
                      config[78] = focus_method
                      config[79] = star_metric
                      config[80] = snr_display
                      config[81] = metrics_interval
                      config[82] = ls_lucky_save_progress
                      config[83] = isp_enable
                      config[84] = allsky_mode
                      config[85] = allsky_mean_target
                      config[86] = allsky_mean_threshold
                      config[87] = allsky_video_fps
                      config[88] = allsky_max_gain
                      config[89] = allsky_apply_stretch
                      config[90] = allsky_cleanup_jpegs
                      config[91] = ls_save_progress
                      config[92] = ls_save_final
                      config[93] = ls_lucky_save_final
                      config[94] = fix_bad_pixels
                      config[95] = fix_bad_pixels_sigma
                      config[96] = fix_bad_pixels_min_adu
                      with open(config_file, 'w') as f:
                          for item in range(0,len(titles)):
                              f.write(titles[item] + " : " + str(config[item]) + "\n")
                      time.sleep(1)
                      text(0,8,2,0,1,"SAVE CONFIG",fv,7)

              # Page 1 : Gestion des paramètres originaux
              elif current_page == 1 and button_row == 1:
                # PREVIEW REFRESH (3-10)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_preview_refresh':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_preview_refresh = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_preview_refresh -=1
                        ls_preview_refresh = max(ls_preview_refresh,pmin)
                    else:
                        ls_preview_refresh +=1
                        ls_preview_refresh = min(ls_preview_refresh,pmax)
                text(0,1,3,1,1,str(ls_preview_refresh),fv,7)
                draw_Vbar(0,1,greyColor,'ls_preview_refresh',ls_preview_refresh)
                # Mettre à jour la config (même si arrêté)
                if livestack is not None:
                    livestack.configure(preview_refresh=ls_preview_refresh)
                time.sleep(.05)

              elif current_page == 1 and button_row == 2:
                # ALIGNMENT MODE (0-2: translation/rotation/affine)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_alignment_mode':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_alignment_mode = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_alignment_mode -=1
                        ls_alignment_mode = max(ls_alignment_mode,pmin)
                    else:
                        ls_alignment_mode +=1
                        ls_alignment_mode = min(ls_alignment_mode,pmax)
                text(0,2,3,1,1,ls_alignment_modes[ls_alignment_mode],fv,7)
                draw_Vbar(0,2,greyColor,'ls_alignment_mode',ls_alignment_mode)
                # Mettre à jour la config (même si arrêté)
                if livestack is not None:
                    livestack.configure(alignment_mode=ls_alignment_modes[ls_alignment_mode])
                time.sleep(.05)

              elif current_page == 1 and button_row == 3:
                # MAX FWHM (0=OFF, 100-250 affiché 10.0-25.0)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_max_fwhm':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_max_fwhm = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_max_fwhm -=10
                        ls_max_fwhm = max(ls_max_fwhm,pmin)
                    else:
                        ls_max_fwhm +=10
                        ls_max_fwhm = min(ls_max_fwhm,pmax)
                if ls_max_fwhm == 0:
                    text(0,3,3,1,1,"OFF",fv,7)
                else:
                    text(0,3,3,1,1,str(ls_max_fwhm/10)[0:4],fv,7)
                draw_Vbar(0,3,greyColor,'ls_max_fwhm',ls_max_fwhm)
                # Mettre à jour la config livestack en temps réel
                if livestack is not None:
                    livestack.configure(max_fwhm=ls_max_fwhm / 10.0 if ls_max_fwhm > 0 else 999.0)
                time.sleep(.05)

              elif current_page == 1 and button_row == 4:
                # MIN SHARPNESS (0=OFF, 30-150 affiché 0.030-0.150)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_min_sharpness':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_min_sharpness = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_min_sharpness -=5
                        ls_min_sharpness = max(ls_min_sharpness,pmin)
                    else:
                        ls_min_sharpness +=5
                        ls_min_sharpness = min(ls_min_sharpness,pmax)
                if ls_min_sharpness == 0:
                    text(0,4,3,1,1,"OFF",fv,7)
                else:
                    text(0,4,3,1,1,str(ls_min_sharpness/1000)[0:5],fv,7)
                draw_Vbar(0,4,greyColor,'ls_min_sharpness',ls_min_sharpness)
                # Mettre à jour la config livestack en temps réel
                if livestack is not None:
                    livestack.configure(min_sharpness=ls_min_sharpness / 1000.0 if ls_min_sharpness > 0 else 0.0)
                time.sleep(.05)

              elif current_page == 1 and button_row == 5:
                # MAX DRIFT (0=OFF, 500-5000 pixels)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_max_drift':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_max_drift = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_max_drift -=100
                        ls_max_drift = max(ls_max_drift,pmin)
                    else:
                        ls_max_drift +=100
                        ls_max_drift = min(ls_max_drift,pmax)
                if ls_max_drift == 0:
                    text(0,5,3,1,1,"OFF",fv,7)
                else:
                    text(0,5,3,1,1,str(ls_max_drift),fv,7)
                draw_Vbar(0,5,greyColor,'ls_max_drift',ls_max_drift)
                # Mettre à jour la config livestack en temps réel
                if livestack is not None:
                    livestack.configure(max_drift=float(ls_max_drift) if ls_max_drift > 0 else 999999.0)
                time.sleep(.05)

              elif current_page == 1 and button_row == 6:
                # MIN STARS (0=OFF, 1-20 étoiles)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_min_stars':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_min_stars = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_min_stars -=1
                        ls_min_stars = max(ls_min_stars,pmin)
                    else:
                        ls_min_stars +=1
                        ls_min_stars = min(ls_min_stars,pmax)
                if ls_min_stars == 0:
                    text(0,6,3,1,1,"OFF",fv,7)
                else:
                    text(0,6,3,1,1,str(ls_min_stars),fv,7)
                draw_Vbar(0,6,greyColor,'ls_min_stars',ls_min_stars)
                # Mettre à jour la config livestack en temps réel
                if livestack is not None:
                    livestack.configure(min_stars=int(ls_min_stars))
                time.sleep(.05)

              elif current_page == 1 and button_row == 7:
                # QUALITY CONTROL (0=OFF, 1=ON)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_enable_qc':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_enable_qc = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    # Toggle entre 0 et 1
                    ls_enable_qc = 1 - ls_enable_qc
                if ls_enable_qc == 0:
                    text(0,7,3,1,1,"OFF",fv,7)
                else:
                    text(0,7,3,1,1,"ON",fv,7)
                draw_Vbar(0,7,greyColor,'ls_enable_qc',ls_enable_qc)
                # Mettre à jour la config livestack en temps réel
                if livestack is not None:
                    livestack.configure(enable_qc=bool(ls_enable_qc))
                time.sleep(.05)

              elif current_page == 2 and button_row == 8:
                   # PAGE 2 : SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,7)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = saturation
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = denoise
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = raw_format  # Remplace timet (fixé à 100ms)
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   config[42] = ghs_D
                   config[43] = ghs_b
                   config[44] = ghs_SP
                   config[45] = ghs_LP
                   config[46] = ghs_HP
                   config[47] = ghs_preset
                   config[48] = ls_preview_refresh
                   config[49] = ls_alignment_mode
                   config[50] = ls_enable_qc
                   config[51] = ls_max_fwhm
                   config[52] = ls_min_sharpness
                   config[53] = ls_max_drift
                   config[54] = ls_min_stars
                   config[55] = ls_stack_method
                   config[56] = ls_stack_kappa
                   config[57] = ls_stack_iterations
                   config[58] = ls_planetary_enable
                   config[59] = ls_planetary_mode
                   config[60] = ls_planetary_disk_min
                   config[61] = ls_planetary_disk_max
                   config[62] = ls_planetary_threshold
                   config[63] = ls_planetary_margin
                   config[64] = ls_planetary_ellipse
                   config[65] = ls_planetary_window
                   config[66] = ls_planetary_upsample
                   config[67] = ls_planetary_highpass
                   config[68] = ls_planetary_roi_center
                   config[69] = ls_planetary_corr
                   config[70] = ls_planetary_max_shift
                   config[71] = ls_lucky_buffer
                   config[72] = ls_lucky_keep
                   config[73] = ls_lucky_score
                   config[74] = ls_lucky_stack
                   config[75] = ls_lucky_align
                   config[76] = ls_lucky_roi
                   config[77] = use_native_sensor_mode
                   config[78] = focus_method
                   config[79] = star_metric
                   config[80] = snr_display
                   config[81] = metrics_interval
                   config[82] = ls_lucky_save_progress
                   config[83] = isp_enable
                   config[84] = allsky_mode
                   config[85] = allsky_mean_target
                   config[86] = allsky_mean_threshold
                   config[87] = allsky_video_fps
                   config[88] = allsky_max_gain
                   config[89] = allsky_apply_stretch
                   config[90] = allsky_cleanup_jpegs
                   config[91] = ls_save_progress
                   config[92] = ls_save_final
                   config[93] = ls_lucky_save_final
                   config[94] = fix_bad_pixels
                   config[95] = fix_bad_pixels_sigma
                   config[96] = fix_bad_pixels_min_adu
                   with open(config_file, 'w') as f:
                       for item in range(0,len(titles)):
                           f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,7)

              elif current_page == 1 and button_row == 8:
                  # Page 1 - Bouton ligne 8 : Navigation vers Page 2
                  menu_page[8] = 2
                  Menu()

              elif button_row == 9:
                  # Navigation: Retour Page 2 -> Page 1
                  current_page = menu_page.get(8, 1)
                  if current_page == 2:
                      # Page 2 -> Page 1
                      menu_page[8] = 1
                      Menu()

            elif menu == 10:
              # METRICS Settings - Gestion des clics
              if button_row == 1:
                # FOCUS METHOD (0-4: OFF/Laplacian/Gradient/Sobel/Tenengrad)
                pmin = 0
                pmax = 4
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    focus_method = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    focus_method = min(focus_method, pmax)  # Limiter à pmax
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        focus_method -=1
                        focus_method = max(focus_method,pmin)
                    else:
                        focus_method +=1
                        focus_method = min(focus_method,pmax)
                text(0,1,3,1,1,focus_methods[focus_method],fv,10)
                draw_bar(0,1,lgrnColor,'focus_method',focus_method)
                time.sleep(.05)

              elif button_row == 2:
                # STAR METRIC (0-2: OFF/HFR/FWHM)
                pmin = 0
                pmax = 2
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    star_metric = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    star_metric = min(star_metric, pmax)  # Limiter à pmax
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        star_metric -=1
                        star_metric = max(star_metric,pmin)
                    else:
                        star_metric +=1
                        star_metric = min(star_metric,pmax)
                text(0,2,3,1,1,star_metrics[star_metric],fv,10)
                draw_bar(0,2,lgrnColor,'star_metric',star_metric)
                time.sleep(.05)

              elif button_row == 3:
                # SNR DISPLAY (0-1: OFF/ON)
                pmin = 0
                pmax = 1
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    snr_display = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                else:
                    # Toggle entre 0 et 1
                    snr_display = 1 - snr_display
                if snr_display == 0:
                    text(0,3,3,1,1,"OFF",fv,10)
                else:
                    text(0,3,3,1,1,"ON",fv,10)
                draw_bar(0,3,lgrnColor,'snr_display',snr_display)
                time.sleep(.05)

              elif button_row == 4:
                # CALC INTERVAL (1-10 frames)
                pmin = 1
                pmax = 10
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    metrics_interval = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        metrics_interval -=1
                        metrics_interval = max(metrics_interval,pmin)
                    else:
                        metrics_interval +=1
                        metrics_interval = min(metrics_interval,pmax)
                text(0,4,3,1,1,str(metrics_interval),fv,10)
                draw_bar(0,4,lgrnColor,'metrics_interval',metrics_interval)
                time.sleep(.05)

              elif button_row == 8:
                # SAVE CONFIG
                text(0,8,3,0,1,"SAVE Config",fv,10)
                config[0] = mode
                config[1] = speed
                config[2] = gain
                config[3] = int(brightness)
                config[4] = int(contrast)
                config[5] = frame
                config[6] = int(red)
                config[7] = int(blue)
                config[8] = ev
                config[9] = vlen
                config[10] = fps
                config[11] = vformat
                config[12] = codec
                config[13] = tinterval
                config[14] = tshots
                config[15] = extn
                config[16] = zx
                config[17] = zy
                config[18] = zoom
                config[19] = int(saturation)
                config[20] = meter
                config[21] = awb
                config[22] = sharpness
                config[23] = int(denoise)
                config[24] = quality
                config[25] = profile
                config[26] = level
                config[27] = histogram
                config[28] = histarea
                config[29] = v3_f_speed
                config[30] = v3_f_range
                config[31] = rotate
                config[32] = IRF
                config[33] = str_cap
                config[34] = v3_hdr
                config[35] = raw_format  # Remplace timet (fixé à 100ms)
                config[36] = vflip
                config[37] = hflip
                config[38] = stretch_p_low
                config[39] = stretch_p_high
                config[40] = stretch_factor
                config[41] = stretch_preset
                config[42] = ghs_D
                config[43] = ghs_b
                config[44] = ghs_SP
                config[45] = ghs_LP
                config[46] = ghs_HP
                config[47] = ghs_preset
                config[48] = ls_preview_refresh
                config[49] = ls_alignment_mode
                config[50] = ls_enable_qc
                config[51] = ls_max_fwhm
                config[52] = ls_min_sharpness
                config[53] = ls_max_drift
                config[54] = ls_min_stars
                config[55] = ls_stack_method
                config[56] = ls_stack_kappa
                config[57] = ls_stack_iterations
                config[58] = ls_planetary_enable
                config[59] = ls_planetary_mode
                config[60] = ls_planetary_disk_min
                config[61] = ls_planetary_disk_max
                config[62] = ls_planetary_threshold
                config[63] = ls_planetary_margin
                config[64] = ls_planetary_ellipse
                config[65] = ls_planetary_window
                config[66] = ls_planetary_upsample
                config[67] = ls_planetary_highpass
                config[68] = ls_planetary_roi_center
                config[69] = ls_planetary_corr
                config[70] = ls_planetary_max_shift
                config[71] = ls_lucky_buffer
                config[72] = ls_lucky_keep
                config[73] = ls_lucky_score
                config[74] = ls_lucky_stack
                config[75] = ls_lucky_align
                config[76] = ls_lucky_roi
                config[77] = use_native_sensor_mode
                config[78] = focus_method
                config[79] = star_metric
                config[80] = snr_display
                config[81] = metrics_interval
                config[82] = ls_lucky_save_progress
                config[83] = isp_enable
                config[84] = allsky_mode
                config[85] = allsky_mean_target
                config[86] = allsky_mean_threshold
                config[87] = allsky_video_fps
                config[88] = allsky_max_gain
                config[89] = allsky_apply_stretch
                config[90] = allsky_cleanup_jpegs
                config[91] = ls_save_progress
                config[92] = ls_save_final
                config[93] = ls_lucky_save_final
                config[94] = fix_bad_pixels
                config[95] = fix_bad_pixels_sigma
                config[96] = fix_bad_pixels_min_adu
                with open(config_file, 'w') as f:
                    for item in range(0,len(titles)):
                        f.write(titles[item] + " : " + str(config[item]) + "\n")
                time.sleep(1)
                text(0,8,2,0,1,"SAVE CONFIG",fv,10)

              elif button_row == 9:
                # Retour menu OTHER Settings (menu 2)
                menu = 2
                Menu()

            elif menu == 9:
              # LUCKY STACK Settings - Gestion complète
              if button_row == 1:
                # LUCKY BUFFER (50-500)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_lucky_buffer':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_lucky_buffer = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_lucky_buffer -=10
                        ls_lucky_buffer = max(ls_lucky_buffer,pmin)
                    else:
                        ls_lucky_buffer +=10
                        ls_lucky_buffer = min(ls_lucky_buffer,pmax)
                text(0,1,3,1,1,str(ls_lucky_buffer),fv,7)
                draw_Vbar(0,1,greyColor,'ls_lucky_buffer',ls_lucky_buffer)
                # Mettre à jour la config en temps réel (même si stacker arrêté)
                print(f"[DEBUG] ls_lucky_buffer changé à {ls_lucky_buffer}")
                print(f"[DEBUG] luckystack_active={luckystack_active}, luckystack={luckystack}")
                print(f"[DEBUG] livestack_active={livestack_active}, livestack={livestack}")
                # Toujours appeler configure() si l'objet existe (actif ou non)
                if luckystack is not None:
                    print(f"[DEBUG] Appel luckystack.configure(lucky_buffer_size={ls_lucky_buffer}) [actif={luckystack_active}]")
                    luckystack.configure(lucky_buffer_size=ls_lucky_buffer)
                elif livestack is not None:
                    print(f"[DEBUG] Appel livestack.configure(lucky_buffer_size={ls_lucky_buffer}) [actif={livestack_active}]")
                    livestack.configure(lucky_buffer_size=ls_lucky_buffer)
                else:
                    print(f"[DEBUG] AUCUN objet luckystack/livestack disponible")
                time.sleep(.05)

              elif button_row == 2:
                # LUCKY KEEP % (1-50)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_lucky_keep':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_lucky_keep = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_lucky_keep -=1
                        ls_lucky_keep = max(ls_lucky_keep,pmin)
                    else:
                        ls_lucky_keep +=1
                        ls_lucky_keep = min(ls_lucky_keep,pmax)
                text(0,2,3,1,1,str(ls_lucky_keep)+"%",fv,7)
                draw_Vbar(0,2,greyColor,'ls_lucky_keep',ls_lucky_keep)
                # Mettre à jour la config (même si arrêté)
                if luckystack is not None:
                    luckystack.configure(lucky_keep_percent=float(ls_lucky_keep))
                elif livestack is not None:
                    livestack.configure(lucky_keep_percent=float(ls_lucky_keep))
                time.sleep(.05)

              elif button_row == 3:
                # LUCKY SCORE METHOD (0-3)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_lucky_score':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_lucky_score = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_lucky_score -=1
                        ls_lucky_score = max(ls_lucky_score,pmin)
                    else:
                        ls_lucky_score +=1
                        ls_lucky_score = min(ls_lucky_score,pmax)
                text(0,3,3,1,1,lucky_score_methods[ls_lucky_score],fv,7)
                draw_Vbar(0,3,greyColor,'ls_lucky_score',ls_lucky_score)
                # Mettre à jour la config (même si arrêté)
                score_names = ['laplacian', 'gradient', 'sobel', 'tenengrad']
                if luckystack is not None:
                    luckystack.configure(lucky_score_method=score_names[ls_lucky_score])
                elif livestack is not None:
                    livestack.configure(lucky_score_method=score_names[ls_lucky_score])
                time.sleep(.05)

              elif button_row == 4:
                # LUCKY STACK METHOD (0-2)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_lucky_stack':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_lucky_stack = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_lucky_stack -=1
                        ls_lucky_stack = max(ls_lucky_stack,pmin)
                    else:
                        ls_lucky_stack +=1
                        ls_lucky_stack = min(ls_lucky_stack,pmax)
                text(0,4,3,1,1,lucky_stack_methods[ls_lucky_stack],fv,7)
                draw_Vbar(0,4,greyColor,'ls_lucky_stack',ls_lucky_stack)
                # Mettre à jour la config (même si arrêté)
                stack_names = ['mean', 'median', 'sigma_clip']
                if luckystack is not None:
                    luckystack.configure(lucky_stack_method=stack_names[ls_lucky_stack])
                elif livestack is not None:
                    livestack.configure(lucky_stack_method=stack_names[ls_lucky_stack])
                time.sleep(.05)

              elif button_row == 5:
                # LUCKY ALIGN (toggle)
                ls_lucky_align = 1 - ls_lucky_align
                if ls_lucky_align == 0:
                    text(0,5,3,1,1,"OFF",fv,7)
                else:
                    text(0,5,3,1,1,"ON",fv,7)
                draw_Vbar(0,5,greyColor,'ls_lucky_align',ls_lucky_align)
                # Mettre à jour la config (même si arrêté)
                if luckystack is not None:
                    luckystack.configure(lucky_align_enabled=bool(ls_lucky_align))
                elif livestack is not None:
                    livestack.configure(lucky_align_enabled=bool(ls_lucky_align))
                time.sleep(.05)

              elif button_row == 6:
                # LUCKY ROI % (20-100)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_lucky_roi':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey >= (button_row * bh) and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_lucky_roi = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_lucky_roi -=5
                        ls_lucky_roi = max(ls_lucky_roi,pmin)
                    else:
                        ls_lucky_roi +=5
                        ls_lucky_roi = min(ls_lucky_roi,pmax)
                text(0,6,3,1,1,str(ls_lucky_roi)+"%",fv,7)
                draw_Vbar(0,6,greyColor,'ls_lucky_roi',ls_lucky_roi)
                # Mettre à jour la config (même si arrêté)
                if luckystack is not None:
                    luckystack.configure(lucky_score_roi_percent=float(ls_lucky_roi))
                elif livestack is not None:
                    livestack.configure(lucky_score_roi_percent=float(ls_lucky_roi))
                time.sleep(.05)

              elif button_row == 8:
                   # SAVE CONFIG (même code que menu 8)
                   text(0,8,3,0,1,"SAVE Config",fv,7)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = saturation
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = denoise
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = raw_format  # Remplace timet (fixé à 100ms)
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   config[42] = ghs_D
                   config[43] = ghs_b
                   config[44] = ghs_SP
                   config[45] = ghs_LP
                   config[46] = ghs_HP
                   config[47] = ghs_preset
                   config[48] = ls_preview_refresh
                   config[49] = ls_alignment_mode
                   config[50] = ls_enable_qc
                   config[51] = ls_max_fwhm
                   config[52] = ls_min_sharpness
                   config[53] = ls_max_drift
                   config[54] = ls_min_stars
                   config[55] = ls_stack_method
                   config[56] = ls_stack_kappa
                   config[57] = ls_stack_iterations
                   config[58] = ls_planetary_enable
                   config[59] = ls_planetary_mode
                   config[60] = ls_planetary_disk_min
                   config[61] = ls_planetary_disk_max
                   config[62] = ls_planetary_threshold
                   config[63] = ls_planetary_margin
                   config[64] = ls_planetary_ellipse
                   config[65] = ls_planetary_window
                   config[66] = ls_planetary_upsample
                   config[67] = ls_planetary_highpass
                   config[68] = ls_planetary_roi_center
                   config[69] = ls_planetary_corr
                   config[70] = ls_planetary_max_shift
                   config[71] = ls_lucky_buffer
                   config[72] = ls_lucky_keep
                   config[73] = ls_lucky_score
                   config[74] = ls_lucky_stack
                   config[75] = ls_lucky_align
                   config[76] = ls_lucky_roi
                   config[77] = use_native_sensor_mode
                   config[78] = focus_method
                   config[79] = star_metric
                   config[80] = snr_display
                   config[81] = metrics_interval
                   config[82] = ls_lucky_save_progress
                   config[83] = isp_enable
                   config[84] = allsky_mode
                   config[85] = allsky_mean_target
                   config[86] = allsky_mean_threshold
                   config[87] = allsky_video_fps
                   config[88] = allsky_max_gain
                   config[89] = allsky_apply_stretch
                   config[90] = allsky_cleanup_jpegs
                   config[91] = ls_save_progress
                   config[92] = ls_save_final
                   config[93] = ls_lucky_save_final
                   config[94] = fix_bad_pixels
                   config[95] = fix_bad_pixels_sigma
                   config[96] = fix_bad_pixels_min_adu
                   with open(config_file, 'w') as f:
                       for item in range(0,len(titles)):
                           f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,7)

              elif button_row == 9:
                  # Retour CAMERA Settings
                  menu = 1
                  Menu()

        # RESTART
        if restart > 0:
            kill_preview_process()
            text(0,0,6,2,1,"Waiting for preview ...",int(fv*1.7),1)
            time.sleep(1)
            preview()
