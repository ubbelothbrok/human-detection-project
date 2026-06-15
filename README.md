# 🧑‍🔍 Real-Time Human Detection

A beginner-friendly computer vision project that detects humans in real-time using your webcam. Built with Python and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **Real-time human detection** using two complementary methods
- **HOG + SVM** — detects full-body (standing/walking people)
- **Haar Cascade Upper Body** — detects seated/partial-body (great for webcam use)
- **Live bounding boxes** with color-coded labels
- **FPS counter** displayed in real-time
- **Screenshot capture** — save any frame with a keypress
- **Mode switching** — toggle between detection methods on the fly
- **Error handling** for webcam access issues

---

## 🧠 How It Works

This project uses two classical computer vision techniques:

### 1. HOG (Histogram of Oriented Gradients) + SVM

The image is divided into small cells, and gradient directions (edges) are computed for each cell. These gradients are grouped into histograms that capture the shape/silhouette of objects. A pre-trained SVM (Support Vector Machine) classifier then determines whether the gradient pattern matches a human body.

> **Best for:** Full-body detection — people standing or walking at 3–10 feet from the camera.

### 2. Haar Cascade (Upper Body)

Uses a cascade of trained classifiers that look for patterns of light and dark regions matching the shape of a human head and shoulders. The detection scans the image at multiple scales to find people of different sizes.

> **Best for:** Upper-body detection — people sitting at a desk or partially visible.

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- A working webcam

### Setup

```bash
# Clone or navigate to the project directory
cd human-detection-project

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install opencv-python
```

---

## 🚀 Usage

```bash
# Make sure the virtual environment is activated
source venv/bin/activate

# Run the detector
python3 human_detector.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the program |
| `s` | Save current frame as JPG to `saved_frames/` |
| `m` | Switch detection mode (Both → HOG only → Upper Body only) |

### Detection Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Both** (default) | Runs HOG + Upper Body together | General use |
| **HOG Full-Body Only** | Only the HOG detector | People standing at a distance |
| **Upper Body Only** | Only the Haar cascade | Sitting at a desk / video calls |

### Bounding Box Colors

| Color | Meaning |
|-------|---------|
| 🟢 Green | Full-body detection (HOG) |
| 🔵 Cyan | Upper-body detection (Haar Cascade) |

---

## 📂 Project Structure

```
human-detection-project/
├── human_detector.py     # Main detection script
├── README.md             # This file
├── venv/                 # Python virtual environment
└── saved_frames/         # Auto-created folder for screenshots
    ├── capture_1.jpg
    ├── capture_2.jpg
    └── ...
```

---

## 💡 Tips for Better Detection

| Factor | Recommendation |
|--------|---------------|
| **Lighting** | Use even, well-lit environments. Avoid bright backlights (e.g., window behind you). |
| **Distance** | HOG works best at 3–10 ft. Upper Body works best at 1–5 ft. |
| **Background** | A simple, uncluttered background reduces false positives. |
| **Camera** | Keep the webcam stable. Higher resolution = better accuracy but slower FPS. |
| **Posture** | Face the camera directly. The upper-body detector expects head + shoulders visible. |

---

## 🔧 Tuning Parameters

You can adjust these values inside `human_detector.py` to fine-tune detection:

### HOG Parameters

```python
hog.detectMultiScale(
    frame,
    winStride=(8, 8),   # Smaller = more accurate, slower (try 4,4)
    padding=(8, 8),      # Border padding around detection window
    scale=1.05           # Closer to 1.0 = more thorough, slower
)
```

### Upper Body Cascade Parameters

```python
upper_body_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,    # Closer to 1.0 = more thorough
    minNeighbors=4,     # Higher = fewer false positives
    minSize=(60, 60)    # Minimum object size to detect
)
```

---

## 🧪 Modifications to Try

Here are some beginner-friendly enhancements you can add:

### 1. Detection Counter
Track total detections over time and display on screen.

### 2. Confidence-Based Colors
Use the `weights` returned by HOG to color boxes by confidence level (green = high, yellow = medium, red = low).

### 3. Video Recording
Use `cv2.VideoWriter` to save the entire detection session as a video file.

### 4. Face Detection
Add `haarcascade_frontalface_default.xml` as a third detector to also highlight faces.

### 5. Motion Detection
Compare consecutive frames to detect and highlight areas of movement.

---

## 📚 Learning Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [HOG Descriptor — Wikipedia](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Dalal & Triggs HOG Paper (2005)](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) — The original research paper

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not access the webcam" | Close other apps using the camera. Check if `/dev/video0` exists on Linux. |
| No detections at all | Try switching to "Upper Body Only" mode with `m`. Ensure good lighting. |
| Too many false positives | Increase `minNeighbors` (try 5 or 6) in the cascade parameters. |
| Very low FPS | Increase `winStride` to `(16, 16)` and `scale` to `1.10` for HOG. |
| QFontDatabase warnings | These are harmless — just OpenCV's Qt backend. Detection is unaffected. |

---

## 📝 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
