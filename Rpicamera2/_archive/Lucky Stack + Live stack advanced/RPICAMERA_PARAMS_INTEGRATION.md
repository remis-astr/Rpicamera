# Intégration Paramètres libastrostack dans RPiCamera.py

## Résumé des nouveaux fichiers

| Fichier | Type | Description |
|---------|------|-------------|
| `lucky_imaging.py` | **NOUVEAU** | Module Lucky Imaging complet |
| `aligner_planetary.py` | **NOUVEAU** | Alignement planétaire (Disk/Surface/Hybrid) |
| `config_advanced.py` | **NOUVEAU** | Configuration étendue |
| `stacker_advanced.py` | **NOUVEAU** | Stacker avec méthodes avancées |
| `drizzle.py` | **NOUVEAU** | Super-résolution |
| `rpicamera_livestack_advanced.py` | **NOUVEAU** | Wrapper pour RPiCamera.py |

---

## Nouveaux Paramètres à Ajouter

### 1. Stacker Avancé

```python
# Limites
'ls_stack_method', 0, 4,           # 0=mean, 1=median, 2=kappa_sigma, 3=winsorized, 4=weighted
'ls_stack_kappa', 10, 40,          # Kappa × 10 (valeur réelle: 1.0-4.0)
'ls_stack_iterations', 1, 10,      # Itérations sigma-clip

# Défauts
ls_stack_method = 0           # Mean
ls_stack_kappa = 25           # 2.5
ls_stack_iterations = 3

# Labels
stack_methods = ['Mean', 'Median', 'Kappa-Sigma', 'Winsorized', 'Weighted']
```

| Paramètre | Fonction |
|-----------|----------|
| `ls_stack_method` | Méthode de combinaison. Mean=rapide, Kappa-Sigma=rejette satellites/cosmiques |
| `ls_stack_kappa` | Seuil de rejection σ. Plus bas = plus agressif |
| `ls_stack_iterations` | Passes de sigma-clipping |

---

### 2. Alignement Planétaire

```python
# Limites
'ls_planetary_enable', 0, 1,       # 0=off, 1=on
'ls_planetary_mode', 0, 2,         # 0=disk, 1=surface, 2=hybrid
'ls_planetary_disk_min', 20, 500,  # Rayon min (pixels)
'ls_planetary_disk_max', 100, 2000,# Rayon max (pixels)
'ls_planetary_threshold', 10, 100, # Seuil Canny
'ls_planetary_margin', 5, 50,      # Marge disque (pixels)
'ls_planetary_ellipse', 0, 1,      # 0=cercle, 1=ellipse
'ls_planetary_window', 0, 2,       # Index → 128/256/512
'ls_planetary_upsample', 1, 20,    # Précision sub-pixel
'ls_planetary_highpass', 0, 1,     # Filtre passe-haut
'ls_planetary_roi_center', 0, 1,   # ROI au centre
'ls_planetary_corr', 10, 90,       # Corrélation min (÷100)
'ls_planetary_max_shift', 10, 200, # Décalage max

# Défauts
ls_planetary_enable = 0
ls_planetary_mode = 1         # Surface (FFT)
ls_planetary_disk_min = 50
ls_planetary_disk_max = 500
ls_planetary_threshold = 30
ls_planetary_margin = 10
ls_planetary_ellipse = 0
ls_planetary_window = 1       # 256px
ls_planetary_upsample = 10
ls_planetary_highpass = 1
ls_planetary_roi_center = 1
ls_planetary_corr = 30        # 0.30
ls_planetary_max_shift = 100

# Labels
planetary_modes = ['Disk', 'Surface', 'Hybrid']
planetary_windows = [128, 256, 512]
```

| Mode | Algorithme | Usage recommandé |
|------|------------|------------------|
| **Disk (0)** | Hough Circle sur limbe | Soleil/Lune sans détails visibles |
| **Surface (1)** | Corrélation FFT | Jupiter, Saturne, Soleil avec taches |
| **Hybrid (2)** | Disk + Surface | Lune (limbe + cratères) |

---

### 3. Lucky Imaging

```python
# Limites
'ls_lucky_enable', 0, 1,           # 0=off, 1=on
'ls_lucky_buffer', 50, 500,        # Taille buffer
'ls_lucky_keep', 1, 50,            # % à garder
'ls_lucky_score', 0, 3,            # Méthode scoring
'ls_lucky_stack', 0, 2,            # Méthode stack
'ls_lucky_align', 0, 1,            # Alignement
'ls_lucky_roi', 20, 100,           # % ROI scoring

# Défauts
ls_lucky_enable = 0
ls_lucky_buffer = 100
ls_lucky_keep = 10            # 10%
ls_lucky_score = 0            # Laplacian
ls_lucky_stack = 0            # Mean
ls_lucky_align = 1            # On
ls_lucky_roi = 50             # 50%

# Labels
lucky_score_methods = ['Laplacian', 'Gradient', 'Sobel', 'Tenengrad']
lucky_stack_methods = ['Mean', 'Median', 'Sigma-Clip']
```

| Paramètre | Fonction |
|-----------|----------|
| `ls_lucky_enable` | Active le mode Lucky (buffer + sélection) |
| `ls_lucky_buffer` | Nombre d'images accumulées avant sélection |
| `ls_lucky_keep` | % des meilleures images à garder |
| `ls_lucky_score` | Algorithme de notation (Laplacian=rapide, Tenengrad=précis) |
| `ls_lucky_stack` | Comment combiner les images sélectionnées |
| `ls_lucky_align` | Aligner les images sélectionnées avant stack |
| `ls_lucky_roi` | Zone centrale utilisée pour le scoring |

---

## Logique d'Activation

```
┌─────────────────────────────────────────────────────────────────┐
│  ls_lucky_enable = 1 ?                                          │
│         │                                                       │
│         ├── OUI ──► LUCKY IMAGING                               │
│         │            ├─ Buffer circulaire                       │
│         │            ├─ Scoring rapide                          │
│         │            ├─ Sélection X%                            │
│         │            └─ Stack                                   │
│         │                                                       │
│         └── NON ──► ls_planetary_enable = 1 ?                   │
│                            │                                    │
│                            ├── OUI ──► PLANÉTAIRE               │
│                            │            │                       │
│                            │            └─ Mode selon           │
│                            │               ls_planetary_mode    │
│                            │                                    │
│                            └── NON ──► DSO (étoiles)            │
│                                         │                       │
│                                         └─ Stacker selon        │
│                                            ls_stack_method      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code d'Intégration

### Mapping vers configure()

```python
def update_livestack_config():
    """Appelé quand un paramètre change"""
    
    livestack.configure(
        # === Stacker ===
        stacking_method=['mean', 'median', 'kappa_sigma', 'winsorized', 'weighted'][ls_stack_method],
        kappa=ls_stack_kappa / 10.0,
        iterations=ls_stack_iterations,
        
        # === Planétaire ===
        planetary_enable=bool(ls_planetary_enable),
        planetary_mode=ls_planetary_mode,  # 0/1/2 accepté directement
        planetary_disk_min=ls_planetary_disk_min,
        planetary_disk_max=ls_planetary_disk_max,
        planetary_disk_threshold=ls_planetary_threshold,
        planetary_disk_margin=ls_planetary_margin,
        planetary_disk_ellipse=bool(ls_planetary_ellipse),
        planetary_window=[128, 256, 512][ls_planetary_window],
        planetary_upsample=ls_planetary_upsample,
        planetary_highpass=bool(ls_planetary_highpass),
        planetary_roi_center=bool(ls_planetary_roi_center),
        planetary_corr=ls_planetary_corr / 100.0,
        planetary_max_shift=float(ls_planetary_max_shift),
        
        # === Lucky Imaging ===
        lucky_enable=bool(ls_lucky_enable),
        lucky_buffer_size=ls_lucky_buffer,
        lucky_keep_percent=float(ls_lucky_keep),
        lucky_score_method=['laplacian', 'gradient', 'sobel', 'tenengrad'][ls_lucky_score],
        lucky_stack_method=['mean', 'median', 'sigma_clip'][ls_lucky_stack],
        lucky_align_enabled=bool(ls_lucky_align),
        lucky_score_roi_percent=float(ls_lucky_roi),
    )
```

### Presets Rapides (Optionnel)

```python
presets = [
    'Manuel',           # 0
    'DSO Rapide',       # 1
    'DSO Qualité',      # 2
    'Planétaire',       # 3
    'Solaire',          # 4
    'Lunaire',          # 5
    'Lucky Rapide',     # 6
    'Lucky Qualité',    # 7
]

preset_map = {
    1: 'fast',
    2: 'quality',
    3: 'planetary',
    4: 'solar',
    5: 'lunar',
    6: 'lucky_fast',
    7: 'lucky_quality',
}

def apply_preset(index):
    if index > 0:
        livestack.configure(preset=preset_map[index])
```

---

## Tableau Récapitulatif Complet

| Groupe | Paramètre | Plage | Défaut | Description |
|--------|-----------|-------|--------|-------------|
| **Stacker** | `ls_stack_method` | 0-4 | 0 | Méthode combinaison |
| | `ls_stack_kappa` | 10-40 | 25 | Kappa (÷10) |
| | `ls_stack_iterations` | 1-10 | 3 | Itérations clip |
| **Planétaire** | `ls_planetary_enable` | 0-1 | 0 | Activer |
| | `ls_planetary_mode` | 0-2 | 1 | Disk/Surface/Hybrid |
| | `ls_planetary_disk_min` | 20-500 | 50 | Rayon min |
| | `ls_planetary_disk_max` | 100-2000 | 500 | Rayon max |
| | `ls_planetary_threshold` | 10-100 | 30 | Seuil Canny |
| | `ls_planetary_margin` | 5-50 | 10 | Marge |
| | `ls_planetary_ellipse` | 0-1 | 0 | Ellipse |
| | `ls_planetary_window` | 0-2 | 1 | Fenêtre FFT |
| | `ls_planetary_upsample` | 1-20 | 10 | Upsampling |
| | `ls_planetary_highpass` | 0-1 | 1 | Passe-haut |
| | `ls_planetary_roi_center` | 0-1 | 1 | ROI centre |
| | `ls_planetary_corr` | 10-90 | 30 | Corrélation min |
| | `ls_planetary_max_shift` | 10-200 | 100 | Shift max |
| **Lucky** | `ls_lucky_enable` | 0-1 | 0 | Activer |
| | `ls_lucky_buffer` | 50-500 | 100 | Taille buffer |
| | `ls_lucky_keep` | 1-50 | 10 | % à garder |
| | `ls_lucky_score` | 0-3 | 0 | Méthode score |
| | `ls_lucky_stack` | 0-2 | 0 | Méthode stack |
| | `ls_lucky_align` | 0-1 | 1 | Alignement |
| | `ls_lucky_roi` | 20-100 | 50 | % ROI |
