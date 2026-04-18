# PadelVision

> **Objetivo:** API que, dado un clip de vídeo de pádel, devuelve un score de nivel tipo Playtomic (0–7), clasificación de golpes y análisis de trayectorias de pelota.
>
> **Stack:** Python 3.11 | YOLOv11 | TrackNet v2 | MediaPipe | FastAPI | PyTorch MPS (M4 Pro)
> **Timeline MVP:** 8–10 semanas | **Arquitectura:** modular, cada componente independiente y testeable.

---

## Índice

1. [Arquitectura General](#1-arquitectura-general)
2. [Stack Técnico](#2-stack-técnico)
3. [Especificación de Módulos](#3-especificación-de-módulos)
4. [API REST — Especificación](#4-api-rest--especificación)
5. [Plan de Implementación por Fases](#5-plan-de-implementación-por-fases)
6. [Riesgos y Mitigaciones](#6-riesgos-y-mitigaciones)
7. [Métricas de Éxito del MVP](#7-métricas-de-éxito-del-mvp)
8. [Guía para Usar IA en el Desarrollo](#8-guía-para-usar-ia-en-el-desarrollo)
9. [Setup Inicial — Primeros Pasos](#9-setup-inicial--primeros-pasos)
10. [Referencias y Recursos](#apéndice--referencias-y-recursos)

---

## 1. Arquitectura General

El sistema se compone de siete capas encadenadas en un pipeline de inferencia. Cada capa puede desarrollarse, testearse y mejorarse de forma aislada.

### Pipeline end-to-end

```
VIDEO INPUT (.mp4 / .mov / URL)
        │
        ▼
[1] PREPROCESSOR       OpenCV + decord: extracción de frames, resize, normalización FPS
        │
        ▼
[2] DETECTOR + TRACKER  YOLOv11 (personas) + ByteTrack (IDs estables entre frames)
        │                    \
        │                     [3] BALL TRACKER  TrackNet v2 (trayectoria de la pelota)
        │
        ▼
[4] POSE ESTIMATOR     MediaPipe Pose (33 keypoints por jugador por frame)
        │
        ▼
[5] FEATURE ENGINEERING  Ángulos, velocidades, posición en pista, fases de golpe
        │
        ▼
[6] SHOT CLASSIFIER    LSTM sobre secuencias de keypoints → tipo de golpe
        │
        ▼
[7] SCORING ENGINE     Métricas → Score 0–7 estilo Playtomic por dimensión
        │
        ▼
FASTAPI REST ENDPOINT  JSON con score, golpes clasificados y trayectorias
```

### Estructura de directorios

```
padelvision/
  core/
    preprocessor.py        # Ingesta de vídeo, extracción de frames
    detector.py            # YOLOv11 detección y tracking de jugadores
    ball_tracker.py        # TrackNet inferencia y post-proceso
    pose_estimator.py      # MediaPipe keypoints
    feature_extractor.py   # Feature engineering geométrico
    shot_classifier.py     # LSTM clasificación de golpes
    scoring_engine.py      # Score 0–7 por dimensión
  api/
    main.py                # FastAPI app
    models.py              # Pydantic schemas (request/response)
    tasks.py               # Celery async jobs
  models/                  # Pesos .pt descargados/entrenados
  tests/
  notebooks/               # Experimentación y análisis
  data/                    # Vídeos de ejemplo y anotaciones
  docker/
  pyproject.toml
```

---

## 2. Stack Técnico

| Componente | Librería / Herramienta | Razón |
|---|---|---|
| Lenguaje | Python 3.11+ | Ecosistema CV maduro, tipado con mypy |
| Vídeo I/O | OpenCV 4.9 + decord | decord usa GPU para decode en Apple MPS |
| Detección personas | Ultralytics YOLOv11n/s | COCO preentrenado, detecta personas OOB |
| Tracking | ByteTrack (integrado Ultralytics) | IDs estables entre frames, sin reentrenar |
| Pose estimation | MediaPipe Pose (Lite/Full) | 33 keypoints, corre en CPU/MPS sin GPU dedicada |
| Ball tracking | TrackNet v2 (PyTorch port) | Diseñado para pelotas de raqueta rápidas |
| Deep learning | PyTorch 2.3 + MPS backend | Aceleración nativa Apple Silicon M4 |
| Shot classifier | LSTM custom (PyTorch) | Ligero, entrenado con datos propios |
| API framework | FastAPI 0.111 | Async, OpenAPI auto-doc, Pydantic v2 |
| Async jobs | Celery + Redis | Procesamiento vídeo en background |
| Testing | pytest + pytest-asyncio | Unit + integration tests |
| Notebooks | Jupyter + marimo | Experimentación interactiva |
| Empaquetado | uv (Astral) | Gestor de deps moderno, reemplaza pip/poetry |
| Containerización | Docker + docker-compose | Deploy y reproducibilidad |
| Anotación datos | Label Studio (self-hosted) | Gratuito, soporta vídeo y keypoints |

---

## 3. Especificación de Módulos

### 3.1 Preprocessor

Responsable de la ingesta del vídeo, normalización de FPS, resize y extracción de frames en batches. Es el punto de entrada de todo el pipeline.

**Inputs / Outputs:**

| Campo | Detalle |
|---|---|
| Input | Path local o URL pública a archivo `.mp4` / `.mov` / `.avi` |
| Output | `Iterator[FrameBatch]` de batches de frames (numpy arrays BGR) + metadata |
| Metadata | `{ fps, total_frames, width, height, duration_sec }` |

**Lógica clave:**
- Normalizar a 25–30 FPS si el vídeo tiene FPS diferente (slow-mo a 60+ FPS se submuestrea)
- Resize a `640x360` para YOLO, `256x256` para TrackNet — cada modelo tiene su propio resize
- Detección automática de pista: encuadrar y recortar si la cámara captura zona exterior
- Validaciones: duración mínima 5s, máxima 120s, codec compatible

**Interfaz Python:**

```python
class VideoPreprocessor:
    def __init__(self, target_fps: int = 25, max_dimension: int = 1280): ...
    def load(self, source: str | Path) -> VideoMetadata: ...
    def frame_batches(self, batch_size: int = 16) -> Iterator[FrameBatch]: ...
    def get_court_roi(self) -> BoundingBox | None: ...  # detección automática pista
```

---

### 3.2 Player Detector & Tracker

Usa YOLOv11 para detectar personas en cada frame y ByteTrack para mantener IDs consistentes. No requiere fine-tuning para detección básica de jugadores.

**Configuración YOLOv11:**

```python
model = YOLO('yolo11s.pt')  # small: buen balance velocidad/precisión
results = model.track(
    source=frames,
    classes=[0],          # solo clase 'person'
    conf=0.45,
    iou=0.5,
    tracker='bytetrack.yaml',
    device='mps',         # Apple Silicon
    persist=True,         # mantiene estado del tracker
)
```

**Output por frame:**

```python
@dataclass
class PlayerDetection:
    track_id: int          # ID estable ByteTrack
    bbox: BBox             # x1, y1, x2, y2 en píxeles
    confidence: float
    team: int | None       # 0 o 1, asignado por posición en pista
```

**Consideraciones:**
- Filtrar detecciones fuera de la ROI de la pista para eliminar público/árbitros
- Lógica de asignación de equipo: jugadores lado izquierdo = equipo 0, derecho = equipo 1
- ByteTrack re-identifica hasta 30 frames de pérdida
- Velocidad esperada en M4 Pro: **~45–60 FPS** con yolo11s en MPS

---

### 3.3 Ball Tracker (TrackNet v2)

El tracking de pelota es el módulo técnicamente más complejo. La pelota es pequeña (~15px en 720p), se mueve rápido y sufre motion blur severo.

> **¿Por qué TrackNet y no YOLO para la pelota?**
> YOLO falla con pelotas pequeñas y rápidas. A 100–180 km/h la pelota produce motion blur severo y ocupa ~15×15px en 720p. TrackNet usa **3 frames consecutivos** como input (tensor 9×H×W), lo que le permite predecir la posición incluso cuando la pelota no es visualmente distinguible. Fue diseñado y validado con tenis y bádminton, dos deportes con dinámica muy similar al pádel.

**Arquitectura:**
- Input: 3 frames RGB concatenados en canal → tensor `(9, H, W)`
- Output: heatmap de probabilidad `(H, W)` → pico = posición de la pelota
- Post-proceso: umbralizar heatmap + encontrar centroide del blob
- Interpolación: cuando no hay detección, interpolar con spline cúbica

**Datos y fine-tuning:**

| Opción | Detalle |
|---|---|
| Pesos preentrenados | Disponibles en GitHub (entrenados en tenis). Funcionan como baseline sin datos propios. |
| Fine-tuning opcional | Grabar 10–20 puntos de pádel y anotar con Label Studio. Mejora precisión ~15–20%. |
| Dataset público | TTNet dataset (tenis de mesa) como datos adicionales. |

**Output:**

```python
@dataclass
class BallPosition:
    frame_idx: int
    x: float | None        # None si no detectada
    y: float | None
    confidence: float
    interpolated: bool

@dataclass
class BallTrajectory:
    positions: list[BallPosition]
    bounces: list[int]     # índices de frames donde bota
    speed_kmh: list[float] # velocidad entre frames consecutivos
```

---

### 3.4 Feature Engineering

Transforma las detecciones brutas (bboxes, keypoints, posición pelota) en features semánticas y geométricas que alimentan el clasificador de golpes y el scoring engine.

**Features por jugador por frame:**

| Feature | Cálculo | Uso |
|---|---|---|
| `elbow_angle` | Ángulo 2D: muñeca-codo-hombro (keypoints 15, 13, 11) | Clasificación golpe, calidad técnica |
| `shoulder_rotation` | Ángulo entre línea hombros y eje X | Técnica de giro |
| `hip_rotation` | Ángulo entre línea caderas y eje X | Transferencia de peso |
| `knee_bend` | Ángulo rodilla (cadera-rodilla-tobillo) | Posición de golpeo |
| `weight_transfer` | Desplazamiento CoM entre frames t-3 a t | Dinamismo, anticipación |
| `court_position` | Posición normalizada en pista (0–1, 0–1) | Táctica, posicionamiento |
| `player_speed` | Distancia CoM / delta_t (m/s) | Nivel atlético |
| `distance_to_ball` | Distancia jugador-pelota en momento de impacto | Timing de golpeo |
| `stroke_phase` | Pre-carga / impacto / follow-through (ventana temporal) | Input clasificador |

**Detección de eventos de impacto:**
- Impacto = frame donde distancia jugador-pelota es mínima Y velocidad pelota cambia bruscamente
- Ventana de golpe: `[-10, +15]` frames alrededor del impacto
- Filtros anti-ruido: velocidad pelota suavizada con Savitzky-Golay

---

### 3.5 Shot Classifier

Modelo de clasificación temporal que, dada la ventana de keypoints alrededor de un impacto, predice el tipo de golpe.

**Clases de golpes (MVP):**

| Clase | Descripción |
|---|---|
| `drive_forehand` | Golpe de derecha desde zona de fondo |
| `drive_backhand` | Golpe de revés desde zona de fondo |
| `volea_forehand` | Volea de derecha en la red |
| `volea_backhand` | Volea de revés en la red |
| `bandeja` | Golpe sobre la cabeza, trayectoria parábola |
| `smash` | Remate sobre la cabeza con máxima potencia |
| `lob` | Globo defensivo u ofensivo |
| `unknown` | Golpe no identificado con confianza suficiente |

**Arquitectura del modelo:**

```python
class ShotClassifierLSTM(nn.Module):
    # Input:  (batch, 25_frames, 66_features)  <- 33 keypoints × 2 coordenadas
    # Output: (batch, n_classes) logits
    def __init__(self):
        self.lstm = nn.LSTM(
            input_size=66, hidden_size=128,
            num_layers=2, dropout=0.3, batch_first=True
        )
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
```

**Estrategia de datos sin presupuesto:**

| Fase | Acción |
|---|---|
| Baseline (semana 1–2) | Reglas heurísticas: si `elbow_angle > 150` y `ball_above_head` → bandeja/smash |
| Proxy labels (semana 3–4) | Vídeos de YouTube de jugadores pro con cámara fija. Auto-etiquetado por posición y ángulos. |
| Fine-tuning (semana 5–6) | Anotar 100–200 golpes en Label Studio. Entrenar LSTM sobre estos datos. |
| Producción | Modelo híbrido: LSTM cuando confianza > 0.7, heurísticas como fallback. |

---

### 3.6 Scoring Engine — Score Playtomic 0–7

Agrega todas las métricas en un score global entre 0 y 7, equivalente a la escala Playtomic. El score es multidimensional: un sub-score por dimensión, agregados con pesos.

**Dimensiones del score:**

| Dimensión | Métricas que la componen | Peso MVP |
|---|---|---|
| Consistencia | % golpes clasificados correctamente, tasa de errores no forzados | 25% |
| Técnica de golpeo | Ángulos medios de codo/hombro, estabilidad de pose en impacto | 30% |
| Movilidad | Velocidad media de desplazamiento, cobertura de pista, anticipación | 20% |
| Potencia/Control | Velocidad media de pelota post-impacto, varianza de trayectorias | 15% |
| Posicionamiento | Distancia media a posición óptima según tipo de golpe | 10% |

**Fórmula de scoring:**

```python
def compute_score(metrics: PlayerMetrics) -> PlayerScore:
    # Cada sub-score se normaliza a [0, 7] con percentiles de referencia
    consistency  = normalize(metrics.shot_success_rate,       p=[0.1, 0.9], range=[0, 7])
    technique    = normalize(metrics.avg_elbow_angle_quality, p=[0.1, 0.9], range=[0, 7])
    mobility     = normalize(metrics.avg_speed_ms,            p=[0.1, 0.9], range=[0, 7])
    power        = normalize(metrics.avg_ball_speed_kmh,      p=[0.1, 0.9], range=[0, 7])
    positioning  = normalize(metrics.avg_position_optimality, p=[0.1, 0.9], range=[0, 7])

    global_score = (
        0.25 * consistency + 0.30 * technique +
        0.20 * mobility    + 0.15 * power + 0.10 * positioning
    )
    return PlayerScore(global=round(global_score, 1), breakdown={...})
```

**Calibración de percentiles:**
Los percentiles de referencia se estiman inicialmente con vídeos de jugadores P1/P2 Playtomic (nivel 5–7) como referencia alta y vídeos de jugadores amateur principiantes como referencia baja. Se refinan iterativamente con feedback de usuarios reales.

---

## 4. API REST — Especificación

### `POST /analyze` — Análisis de vídeo

```http
Content-Type: multipart/form-data

{
  "video": <binary>,       // max 500MB
  "options": {
    "players": [0, 1],     // índices de jugadores (vacío = todos)
    "include_ball": true,
    "include_poses": false  // poses brutas (payload grande)
  }
}
```

```json
// Response 202 Accepted
{
  "job_id": "abc123",
  "status": "queued",
  "estimated_seconds": 45
}
```

### `GET /jobs/{job_id}` — Estado del job

```json
{
  "job_id": "abc123",
  "status": "completed",   // queued | processing | completed | failed
  "progress": 1.0,
  "result_url": "/results/abc123"
}
```

### `GET /results/{job_id}` — Resultado completo

```json
{
  "video_duration_sec": 87.3,
  "fps_analyzed": 25,
  "players": [
    {
      "player_id": 0,
      "team": 0,
      "score": {
        "global": 4.2,
        "breakdown": {
          "consistency": 4.5,
          "technique": 4.0,
          "mobility": 3.8,
          "power": 4.6,
          "positioning": 4.1
        }
      },
      "shots": [
        {
          "type": "drive_forehand",
          "frame": 312,
          "timestamp_sec": 12.48,
          "confidence": 0.87,
          "quality_score": 5.1
        }
      ],
      "stats": {
        "total_shots": 34,
        "shot_distribution": { "drive_forehand": 12, "bandeja": 5 },
        "avg_ball_speed_kmh": 98.4,
        "court_coverage_pct": 0.68
      }
    }
  ],
  "ball_trajectory": {
    "total_detected_frames": 1823,
    "detection_rate": 0.83,
    "avg_speed_kmh": 94.2
  }
}
```

---

## 5. Plan de Implementación por Fases

### Fase 1 — Fundamentos del Pipeline `Semanas 1–2`

**Objetivo:** Pipeline end-to-end con modelos out-of-the-box. Al final se puede enviar un vídeo y obtener bboxes de jugadores + keypoints.

| Tarea | Descripción |
|---|---|
| 1.1 Setup del proyecto | `uv`, estructura de directorios, pre-commit, pytest, mypy. Docker compose con Redis. |
| 1.2 VideoPreprocessor | Clase completa con OpenCV + decord. Tests con vídeos de ejemplo. Validaciones. |
| 1.3 PlayerDetector baseline | Integrar YOLOv11s con MPS. Visualización con OpenCV. Benchmark en M4 Pro. |
| 1.4 PoseEstimator baseline | Integrar MediaPipe Pose. Visualización de esqueleto sobre vídeo. |
| 1.5 Pipeline script | CLI que encadena los 3 módulos y genera vídeo anotado de salida. |
| 1.6 Métricas baseline | Logging de FPS, memoria, tiempo por módulo con cProfile. |

**Entregable:**
```bash
python analyze.py --video clip.mp4 --output annotated.mp4
# Output: vídeo con bboxes de jugadores y esqueleto superpuesto
```

---

### Fase 2 — Ball Tracking + Feature Engineering `Semanas 3–4`

**Objetivo:** Integrar TrackNet y construir la capa de features que transforma detecciones brutas en features semánticas.

| Tarea | Descripción |
|---|---|
| 2.1 TrackNet setup | Portar/adaptar implementación PyTorch de TrackNet v2. Descargar pesos preentrenados. |
| 2.2 Ball post-procesado | Umbralizado de heatmap, detección de centroide, interpolación con spline cúbica. |
| 2.3 Detección de bounces | Algoritmo por cambio de dirección vertical de la pelota. |
| 2.4 Feature extractor | Implementar todas las features geométricas. Tests unitarios por feature. |
| 2.5 Detección de impactos | Algoritmo que detecta el momento exacto de impacto jugador-pelota. |
| 2.6 Visualizador de trayectoria | Overlay de trayectoria con heatmap de posiciones frecuentes. |

**Entregable:** Vídeo anotado con trayectoria de pelota + features en JSON. Métrica: % frames con pelota detectada.

---

### Fase 3 — Shot Classifier + Scoring Engine `Semanas 5–7`

**Objetivo:** Entrenar el clasificador de golpes y construir el scoring engine que produce el score 0–7.

| Tarea | Descripción |
|---|---|
| 3.1 Dataset de golpes | Descargar 20–30 vídeos de YouTube de nivel variado. Anotar 150–200 golpes en Label Studio. |
| 3.2 Heurísticas baseline | Clasificador por reglas como baseline. Objetivo: >70% accuracy en golpes claros. |
| 3.3 LSTM training | Pipeline PyTorch: DataLoader, training loop, validation, checkpoint. |
| 3.4 Evaluación | Confusion matrix por clase, precision/recall por tipo de golpe. |
| 3.5 Scoring engine | Implementar `normalize()`, pesos por dimensión, fórmula global. |
| 3.6 Calibración | Vídeos pro como referencia alta, principiantes como baja para fijar percentiles. |
| 3.7 Modelo híbrido | LSTM (cuando conf > 0.7) + heurísticas como fallback. |

**Entregable:** Score 0–7 producido para cualquier vídeo de entrada, con breakdown por dimensión y distribución de golpes.

---

### Fase 4 — FastAPI + Deploy `Semanas 8–10`

**Objetivo:** Empaquetar el pipeline en una API REST funcional con procesamiento asíncrono.

| Tarea | Descripción |
|---|---|
| 4.1 FastAPI app | Endpoints `/analyze`, `/jobs`, `/results` con Pydantic v2 schemas. |
| 4.2 Celery + Redis | Job queue para procesamiento asíncrono. Progreso en tiempo real via polling. |
| 4.3 Pipeline manager | Clase que orquesta todos los módulos. Manejo de errores con reintentos. |
| 4.4 Almacenamiento | JSON de resultados en disco, SQLite para metadata. TTL de 24h para limpiar archivos. |
| 4.5 Docker compose | Contenedor API + Celery worker + Redis. Health checks. |
| 4.6 Tests de integración | Tests end-to-end con vídeos reales. Benchmark de latencia total. |
| 4.7 Documentación OpenAPI | Revisar schemas, descripciones y ejemplos en Swagger UI. |

**Entregable — MVP Completo:**
```bash
# Enviar vídeo y obtener score
curl -X POST /analyze -F video=@clip.mp4
# → { "job_id": "abc123" }

curl /results/abc123
# → JSON completo con score 0-7, golpes, trayectoria

# Deploy
docker-compose up
# → API lista en puerto 8000, Swagger en /docs
```
Latencia objetivo: **< 3 minutos** para vídeo de 90s en M4 Pro.

---

## 6. Riesgos y Mitigaciones

| Riesgo | Severidad | Descripción | Mitigación |
|---|---|---|---|
| Ball tracking impreciso | 🔴 Alto | Motion blur + tamaño pequeño → detección <60% frames en rallys rápidos | Interpolar con spline. Relajar métricas que dependen de pelota. |
| Calibración del score | 🔴 Alto | Sin ground truth real, el score puede no correlacionar con nivel Playtomic | Validar con 10 jugadores de nivel conocido antes de lanzar. |
| Ángulo de cámara | 🟡 Medio | El pipeline asume cámara elevada y centrada. Ángulos bajos rompen la geometría | V1: documentar como requisito. V2: normalizar con homografía de la pista. |
| FPS en M4 Pro | 🟡 Medio | Pipeline completo puede ser más lento de lo esperado con vídeos largos | Benchmark por módulo en Fase 1. Paralelizar por jugador si necesario. |
| Overfitting del clasificador | 🟢 Bajo | Con ~150 golpes el LSTM puede no generalizar bien | Data augmentation (flip, jitter de keypoints). Dropout. Fallback a heurísticas. |
| Oclusiones en la red | 🟢 Bajo | Los 4 jugadores se solapan frecuentemente en la zona de red | ByteTrack maneja hasta 30 frames de pérdida. |

---

## 7. Métricas de Éxito del MVP

| Métrica | Objetivo MVP | Cómo medir |
|---|---|---|
| Detección de jugadores | >90% frames con los 4 jugadores detectados | Evaluar en 10 clips de 30s con anotación manual |
| Tracking de pelota | >65% frames con posición detectada o interpolada | Evaluar en 5 clips con anotación frame a frame |
| Clasificación de golpes | >75% accuracy en 8 clases | Test set de 50 golpes anotados manualmente |
| Score global | Correlación de Pearson > 0.75 con nivel Playtomic real | 10 jugadores con nivel conocido, comparar scores |
| Latencia | <3 min para vídeo de 90s en M4 Pro | Benchmark automático en CI/CD |
| Disponibilidad API | 99% uptime en pruebas locales, errores bien manejados | Test suite de integración end-to-end |

---

## 8. Guía para Usar IA en el Desarrollo

### 8.1 Prompts por módulo

**Módulo 1 — VideoPreprocessor:**
```
Implementa la clase VideoPreprocessor en Python 3.11 con las siguientes especificaciones:
- Usa OpenCV + decord para lectura eficiente en Apple Silicon (MPS)
- Método frame_batches() que devuelve Iterator[FrameBatch] con batch_size configurable
- Normalización de FPS a target_fps con interpolación bilineal entre frames
- Detección de ROI de pista via edge detection (opcional, si confidence < 0.7 devuelve None)
- Dataclass VideoMetadata con fps, total_frames, width, height, duration_sec
- Tests pytest con un vídeo de 10s de ejemplo
- Type hints completos, docstrings, manejo de excepciones específicas
```

**Módulo 4 — Feature Engineering:**
```
Implementa FeatureExtractor en Python que dado:
- Lista de PlayerDetection por frame (bbox + track_id)
- Lista de MediaPipe PoseLandmark por frame por jugador
- Lista de BallPosition por frame
Devuelva para cada impacto detectado un dict con:
- elbow_angle: float (ángulo 2D muñeca-codo-hombro en grados)
- shoulder_rotation: float (ángulo línea hombros vs eje X)
- court_position: tuple[float, float] (normalizado 0-1)
- distance_to_ball_at_impact: float (píxeles)
- stroke_phase: Literal["loading", "impact", "followthrough"]
Incluye tests unitarios con datos sintéticos para cada feature.
```

**Módulo 6 — Scoring Engine:**
```
Implementa ScoringEngine que convierte PlayerMetrics en PlayerScore (0.0–7.0):
- Función normalize(value, percentiles, range) que mapea una métrica a 0-7
- Los percentiles de referencia se cargan de calibration_data.json
- Pesos configurables por dimensión en scoring_config.yaml
- Score global = suma ponderada de sub-scores
- Método compare(score_a, score_b) que devuelve diferencias por dimensión
- Logging detallado de cada métrica y su contribución al score final
- Tests que verifican que un jugador con todas las métricas máximas → score ~6.5–7.0
```

### 8.2 Flujo de iteración recomendado

| Paso | Descripción |
|---|---|
| 1. Pedir implementación base | Prompt detallado con specs. Pedir type hints, tests y docstrings siempre. |
| 2. Ejecutar tests | Si algún test falla, pegar el traceback completo y pedir fix. |
| 3. Benchmark | Ejecutar con datos reales. Pedir optimización si es lento. |
| 4. Refactor | Una vez funciona: "refactoriza para mejor separación de responsabilidades". |
| 5. Edge cases | Pedir: "¿qué casos extremos no cubre esta implementación y cómo manejarlos?" |

---

## 9. Setup Inicial — Primeros Pasos

### 1. Inicializar el proyecto

```bash
# Instalar uv (gestor de paquetes moderno)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear proyecto
uv init padelvision
cd padelvision

# Añadir dependencias principales
uv add ultralytics opencv-python mediapipe torch torchvision
uv add fastapi celery redis pydantic
uv add decord numpy scipy

uv add --dev pytest pytest-asyncio mypy ruff jupyter
```

### 2. Verificar aceleración MPS en M4 Pro

```python
import torch
print(torch.backends.mps.is_available())  # debe ser True
print(torch.backends.mps.is_built())      # debe ser True

# Test rápido
device = torch.device('mps')
x = torch.randn(1000, 1000).to(device)
print('MPS OK:', x.shape)
```

### 3. Descargar modelos

```bash
# YOLOv11 (se descarga automáticamente en primer uso)
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"

# TrackNet v2 — clonar repo y descargar pesos preentrenados
git clone https://github.com/ChgygLin/TrackNetV2-pytorch
# Pesos disponibles en el README del repo
```

### 4. Label Studio para anotación

```bash
# Instalar y arrancar Label Studio (gratuito, self-hosted)
pip install label-studio
label-studio start
# Abrir http://localhost:8080
# Crear proyecto tipo 'Video Object Tracking'
# Importar clips de pádel y anotar tipos de golpe
```

---

## Apéndice — Referencias y Recursos

| Recurso | URL |
|---|---|
| Ultralytics YOLOv11 | https://docs.ultralytics.com |
| TrackNet v2 PyTorch | https://github.com/ChgygLin/TrackNetV2-pytorch |
| MediaPipe Pose | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker |
| Label Studio | https://labelstud.io |
| ByteTrack paper | arXiv:2110.06864 |
| uv (package manager) | https://docs.astral.sh/uv |
| PyTorch MPS Guide | https://pytorch.org/docs/stable/notes/mps.html |
| FastAPI | https://fastapi.tiangolo.com |
