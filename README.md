# ⚽ Football Vision AI

A computer vision pipeline for analyzing football (soccer) matches using object detection, tracking, and color-based team classification to annotate players, goalkeepers, referees, and the ball on the original broadcast video.

---

## 🚀 Features

* 🧍 Player, goalkeeper, referee, and ball detection
* 🎨 Team classification using HSV color histograms (no neural network required)
* 🎯 Multi-object tracking with ByteTrack
* 🥅 Automatic goalkeeper team assignment based on proximity
* 🗺️ Homography transformation (camera → pitch view)
* 📍 Real-time player & ball position mapping
* 🎥 Annotated output video with ellipses, labels, and ball marker

---

## 🧠 How It Works

1. **Object Detection**
   Detect players, goalkeepers, referees, and the ball using a fine-tuned YOLO model on every frame.

2. **Tracking**
   Assign consistent IDs across frames using ByteTrack so each player keeps the same number throughout the video.

3. **Team Classification**
   Sample frames from the video, extract player crops, and compute HSV color histograms focused on the jersey region (top 2/3 of each crop). KMeans clustering separates the two teams by kit color — no neural network needed, works cross-platform without threading issues.

4. **Goalkeeper Assignment**
   Each goalkeeper is assigned to the team whose centroid (average player position) they are closest to.

5. **Visualization**
   Render ellipses around players colored by team, tracker ID labels, and a triangle marker above the ball.

6. **Pitch Map (optional)**
   Detect field keypoints, compute a homography, and render a top-down tactical map showing player and ball positions on a 2D pitch.

---

## 📁 Project Structure

```
football_vision/
│
├── data/                  # Input and output videos
├── models/                # Trained YOLO model weights
│
├── pipeline/              # Main inference pipeline
│   ├── main.py            # Entry point — change video name here
│   ├── inference.py       # Model loading utilities
│   └── match_inference.py # Standalone match inference script
│
├── modules/               # Core reusable components
│   ├── crop_players.py    # Strided crop extraction
│   ├── goalkeepers.py     # Goalkeeper team assignment
│   ├── homography.py      # ViewTransformer + keypoint padding
│   └── data_setup.py      # Roboflow dataset download
│
├── embeddings/            # Team classification
│   ├── siglip_embeddings.py       # HSV histogram classifier
│   └── embeddings_visualization.py
│
├── training/              # Model training scripts
│   ├── train.py           # Train player detection model
│   ├── train_pitch_model.py
│   └── train_pose_model.py
│
├── visualization/         # Pitch map rendering
│   └── match_visualization.py
│
├── evaluation/            # Metrics and evaluation
│   ├── evaluate.py
│   └── evaluation_metrics.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/rene-54/Football-vision.git
cd Football-vision
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the full pipeline

Place your input video inside the `data/` folder, then open `pipeline/main.py` and change the last line to match your video filename (without extension):

```python
main("Football_A")   # will read data/Football_A.mp4
                     # and write data/Football_A_output.mp4
```

Then run:

```bash
# Windows
python pipeline\main.py

# Mac/Linux
python pipeline/main.py
```

### Input / Output

| | Path |
|---|---|
| Input | `data/<video_name>.mp4` |
| Output | `data/<video_name>_output.mp4` |

---

## 🏋️ Training Models

### Train the player detection model

```bash
python training/train.py
```

### Train the pitch keypoint model

```bash
python training/train_pitch_model.py
```

### Train the pose model

```bash
python training/train_pose_model.py
```

> Requires a `ROBOFLOW_API_KEY` environment variable set for dataset download.

---

## 🧩 Tech Stack

| Library | Purpose |
|---|---|
| YOLOv8 | Object detection |
| Supervision | Tracking, annotation utilities |
| ByteTrack | Multi-object tracking |
| OpenCV | Image processing, homography |
| scikit-learn | KMeans clustering, StandardScaler |
| NumPy | Numerical computations |
| PyTorch | Model inference |
| Roboflow | Dataset management |

---

## 📊 Example Output

* Players annotated with ellipses colored by team (blue / pink)
* Referees annotated in gold
* Tracker ID labels below each detection
* Triangle marker above the ball
* Optional: top-down tactical pitch map with player positions

---

## 🐛 Known Issues & Platform Notes

| Issue | Status |
|---|---|
| macOS mutex crash with SigLIP/HuggingFace | Resolved — replaced with HSV histogram classifier |
| ByteTrack deprecation warning (sv ≥ 0.28) | Harmless, will be removed in sv 0.30 |
| `sports` module not installed | Not required — team classification uses OpenCV only |

---

## 🔮 Future Improvements

* Real-time live match processing
* Pass detection & event recognition
* Player heatmaps & analytics
* Re-enable SigLIP embeddings on Linux/Windows for higher team classification accuracy
* Integration with live data feeds

---

## 🙏 Acknowledgment

This project was developed as part of a workshop led by Alex Sanchez and Khang Ho.

---

## 👤 Author

Rene Guerra

---

## 📄 License

This project is for educational and research purposes.