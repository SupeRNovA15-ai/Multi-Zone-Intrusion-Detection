# Multi-Zone-Intrusion-Detection
Here is the remaining content formatted as a singular, pastable bash script.

This script uses a `cat << 'EOF'` (here-document) to encapsulate the documentation. You can save this as a `.sh` file, and when you run it, it will print the formatted README text to your console.

```bash
#!/bin/bash
#
# This file contains the project documentation for the
# Variphi Multi-Zone Intrusion Detection System.
# You can execute this script to print the README to the console.

cat << 'EOF'
# Variphi Multi-Zone Intrusion Detection System

A YOLOv8-powered multi-zone intrusion detection system with live tracking, polygonal zone definition, and automatic Tkinter report generation.

It allows users to:
* Define multiple custom polygon zones interactively on video frames.
* Detect and track human movement using YOLOv8 (ByteTrack/BotSort/Centroid tracking).
* Log entry/exit events per zone.
* View a Tkinter-based intrusion report GUI at the end of processing.

---

### 🖥️ Features

* ✅ Define unlimited polygonal zones interactively
* ✅ Uses YOLOv8 for accurate real-time detection
* ✅ Supports ByteTrack, BotSort, or Centroid tracking methods
* ✅ Saves structured logs (`intrusion_events.json`)
* ✅ Generates readable logs (`intrusion_events.log`)
* ✅ Auto-generates a Tkinter GUI summary report
* ✅ Allows saving results as `intrusion_summary.txt`
* ✅ Exports processed video with overlays

---

### 🗂️ Project Structure

```

<img width="772" height="521" alt="image" src="https://github.com/user-attachments/assets/7c1196cb-d0cc-4c50-9d88-32ded9f54cd6" />


````

---

### ⚙️ Setup Instructions

**1. Clone or Download the Repository**
```bash
git clone [https://github.com/](https://github.com/)<your-username>/multi-zone-intrusion.git
cd multi-zone-intrusion
````

Or simply copy all files to a local folder.

**2. Install Python (Recommended: 3.9 – 3.11)**
Make sure Python is installed and available in `PATH`.
Check your version:

```bash
python --version
```

**3. Install Required Dependencies**
Install all dependencies using `pip`:

```bash
pip install ultralytics==8.2.82 opencv-python numpy pillow tk
```

Optional (if YOLO requests additional tracking packages):

```bash
pip install lapx filterpy
```

**4. Download the YOLOv8 Model**
Download the small YOLOv8 model file (`yolov8n.pt`) from the official Ultralytics source:
👉 **[Download YOLOv8n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)**

Place it in:

```
models/yolov8n.pt
```

**5. Create Tracker Config Files**

`trackers/bytetrack.yaml`

```yaml
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.7
match_thresh: 0.8
lost_track_buffer: 30
frame_rate: 30
```

`trackers/botsort.yaml`

```yaml
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.7
match_thresh: 0.8
lost_track_buffer: 30
frame_rate: 30
gmc_method: sparseOptFlow
```

-----

### ▶️ How to Run Locally

1.  Open a terminal in your project directory.
2.  Run the main script:
    ```bash
    python main.py
    ```
3.  Choose your preferred tracking method:
    ```
    1. bytetrack - SORT-based (Recommended)
    2. botsort - Enhanced SORT
    3. centroid - Custom centroid-based
    ```
4.  A file dialog will appear — select your video (e.g., CCTV footage).
5.  If zones are not yet defined:
      * A window will open for you to draw zones.
      * Use these keys:
          * 🖱️ **Left-click** → Add point
          * **c** → Complete current zone
          * **n** → Start new zone
          * **r** → Reset drawing
          * **q** → Save & exit zone creation
6.  Once zones are saved, processing starts automatically.
7.  You’ll see detection boxes and logs printed live in the console.
8.  When processing finishes:
      * The output video is saved as `out-<video_name>.mp4`.
      * A Tkinter summary window pops up with event logs.

-----

### 🪟 Intrusion Report GUI

At the end of video processing:

1.  A window titled “Intrusion Report Summary” appears.
2.  It displays all recorded events (entered/exited) with timestamps, IDs, zones, confidence, and duration.
3.  Color-coded entries:
      * 🔴 **Red** → Intrusion
      * 🟢 **Green** → Exit
4.  You can click “Save Report” to export a text summary (`intrusion_summary.txt`).

-----

### 📊 Output Files Explained

| File | Description |
| :--- | :--- |
| **`zones.json`** | Stores zone polygon coordinates so you don’t need to redraw them |
| **`intrusion_events.json`** | Structured machine-readable event data |
| **`intrusion_events.log`** | Human-readable event summary |
| **`intrusion_summary.txt`** | Saved summary report from the Tkinter window |
| **`out-<video>.mp4`** | Video with drawn zones, tracked IDs, and intrusions |


```
```
