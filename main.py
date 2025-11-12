import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import json
import os
from datetime import datetime
from collections import defaultdict, deque
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter


class MultiZoneIntrusionDetector:
    def __init__(self, tracking_method="bytetrack"):
        self.model = YOLO("yolov8n.pt")
        self.zones = []
        self.zone_names = []
        self.tracking_method = tracking_method

        self.object_states = defaultdict(lambda: {
            'in_zone': False,
            'zone_name': None,
            'entry_time': None,
            'positions': deque(maxlen=10),
            'first_seen': None,
            'total_detections': 0
        })
        self.events_log = []
        self.colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        print(f"Initialized with {tracking_method} tracking")

    # ---------- Zone management ----------
    def save_zones(self, filename="zones.json"):
        zones_data = [{"name": name, "polygon": polygon.tolist()}
                      for name, polygon in zip(self.zone_names, self.zones)]
        with open(filename, 'w') as f:
            json.dump(zones_data, f, indent=2)
        print(f"Zones saved to {filename}")

    def load_zones(self, filename="zones.json"):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                zones_data = json.load(f)
            self.zones = [np.array(zone["polygon"]) for zone in zones_data]
            self.zone_names = [zone["name"] for zone in zones_data]
            print(f"Loaded {len(self.zones)} zones from {filename}")
            return True
        return False

    # ---------- Drawing zones ----------
    def mouse_callback(self, event, x, y, flags, param):
        frame, points = param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point recorded: ({x}, {y}) - Total points: {len(points)}")
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            if len(points) > 1:
                cv2.line(frame, points[-2], points[-1], (0, 255, 255), 2)
            cv2.imshow("Draw Zones", frame)

    def draw_zones_ui(self, frame):
        print("\n=== Zone Drawing Mode ===")
        self.zones = []
        self.zone_names = []
        zone_count = 0
        temp_frame = frame.copy()
        points = []

        cv2.namedWindow("Draw Zones")
        cv2.setMouseCallback("Draw Zones", self.mouse_callback, (temp_frame, points))

        print("\nInstructions:")
        print(" - Left-click to add points")
        print(" - Press 'c' to complete current zone")
        print(" - Press 'n' to start new zone")
        print(" - Press 'r' to reset current drawing")
        print(" - Press 'q' to save zones & exit\n")

        while True:
            display = temp_frame.copy()

            for i, (zone, name) in enumerate(zip(self.zones, self.zone_names)):
                color = self.colors[i % len(self.colors)]
                cv2.polylines(display, [zone], True, color, 2)
                centroid = np.mean(zone, axis=0).astype(int)
                cv2.putText(display, name, tuple(centroid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if len(points) > 0:
                for i, pt in enumerate(points):
                    cv2.circle(display, pt, 5, (0, 255, 255), -1)
                    if i > 0:
                        cv2.line(display, points[i - 1], points[i], (0, 255, 255), 2)

            cv2.putText(display, "Press 'c'=complete, 'n'=new, 'q'=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Draw Zones", display)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'):
                if len(points) >= 3:
                    zone_count += 1
                    zone_polygon = np.array(points)
                    zone_name = input(f"Enter name for Zone {zone_count}: ").strip() or f"Zone_{zone_count}"
                    self.zones.append(zone_polygon)
                    self.zone_names.append(zone_name)
                    print(f"✅ Zone '{zone_name}' created with {len(points)} points")
                    points.clear()
                    temp_frame = frame.copy()
                else:
                    print("⚠️ Need at least 3 points to create a zone")

            elif key == ord('r'):
                points.clear()
                print("↩️ Reset current drawing")

            elif key == ord('n'):
                points.clear()
                print("➕ Starting new zone")

            elif key == ord('q'):
                if len(self.zones) > 0:
                    self.save_zones()
                    print("💾 Zones saved. Exiting drawing mode.")
                    cv2.destroyWindow("Draw Zones")
                    return True
                else:
                    print("⚠️ Create at least one zone before quitting")

        cv2.destroyAllWindows()
        return len(self.zones) > 0

    # ---------- Tracking ----------
    def setup_tracking(self):
        tracker_configs = {
            "bytetrack": "bytetrack.yaml",
            "botsort": "botsort.yaml",
            "centroid": None
        }
        return tracker_configs.get(self.tracking_method, "bytetrack.yaml")

    def calculate_centroid(self, box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    # ---------- Event logging ----------
    def log_event(self, event_type, object_id, zone_name, confidence=1.0, duration=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        event_data = {
            'timestamp': timestamp,
            'event_type': event_type,
            'object_id': object_id,
            'zone_name': zone_name,
            'confidence': f"{confidence:.3f}",
            'duration_seconds': f"{duration:.2f}" if duration else "0.00"
        }
        self.events_log.append(event_data)
        with open("intrusion_events.json", "a") as f:
            f.write(json.dumps(event_data) + "\n")
        return event_data

    def check_zone_intrusion(self, object_id, point, confidence):
        current_zone_name = None
        for zone, zone_name in zip(self.zones, self.zone_names):
            if cv2.pointPolygonTest(zone, point, False) >= 0:
                current_zone_name = zone_name
                break

        previous_state = self.object_states[object_id]
        if previous_state['first_seen'] is None:
            previous_state['first_seen'] = datetime.now()
        previous_state['total_detections'] += 1

        if not previous_state['in_zone'] and current_zone_name is not None:
            self.object_states[object_id].update({'in_zone': True, 'zone_name': current_zone_name,
                                                  'entry_time': datetime.now()})
            self.log_event('entered', object_id, current_zone_name, confidence)

        elif previous_state['in_zone'] and current_zone_name is None:
            exited_zone = previous_state['zone_name']
            duration = (datetime.now() - previous_state['entry_time']).total_seconds()
            self.object_states[object_id].update({'in_zone': False, 'zone_name': None,
                                                  'entry_time': None})
            self.log_event('exited', object_id, exited_zone, confidence, duration)

    # ---------- Processing ----------
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video")
            return
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_name = f"out-{os.path.basename(video_path).split('.')[0]}.mp4"
        out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"\n=== Starting Video Processing ===\nVideo: {video_path}\nResolution: {w}x{h}, FPS: {fps}")

        tracker_cfg = self.setup_tracking()
        frame_count = 0
        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            results = self.model.track(frame, persist=True, tracker=tracker_cfg, verbose=False, classes=[0])
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                for box, conf, tid in zip(boxes, confs, track_ids):
                    tid = int(tid)
                    centroid = self.calculate_centroid(box)
                    self.object_states[tid]['positions'].append(centroid)
                    self.check_zone_intrusion(tid, centroid, conf)
                    self.draw_detection(frame, tid, box, centroid, conf)
            self.draw_zones(frame)
            self.draw_stats(frame, frame_count, total_frames)
            out.write(frame)
            cv2.imshow("Multi-Zone Intrusion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n=== Processing Complete ===")
        print(f"Total Events: {len(self.events_log)} | Output saved: {out_name}")

        time.sleep(0.5)
        threading.Thread(target=self.show_intrusion_summary, daemon=False).start()

    # ---------- Drawing ----------
    def draw_detection(self, frame, track_id, box, centroid, conf):
        x1, y1, x2, y2 = map(int, box)
        state = self.object_states[track_id]
        color = (0, 0, 255) if state['in_zone'] else (0, 255, 0)
        status = f"IN {state['zone_name']}" if state['in_zone'] else "Safe"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} {status} ({conf:.2f})"
        cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, centroid, 5, color, -1)

    def draw_zones(self, frame):
        for i, (zone, name) in enumerate(zip(self.zones, self.zone_names)):
            color = self.colors[i % len(self.colors)]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [zone], True, color, 2)
            centroid = np.mean(zone, axis=0).astype(int)
            cv2.putText(frame, name, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def draw_stats(self, frame, frame_count, total):
        stats = [f"Frame: {frame_count}/{total}", f"Events: {len(self.events_log)}"]
        for i, s in enumerate(stats):
            cv2.putText(frame, s, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---------- GUI summary ----------
    def show_intrusion_summary(self):
        win = tk.Tk()
        win.title("Intrusion Report Dashboard")
        win.geometry("1000x700")
        win.configure(bg="#f0f0f0")

        tk.Label(win, text="📊 Intrusion Detection Summary", font=("Arial", 18, "bold"), bg="#f0f0f0").pack(pady=10)

        intrusions = [e for e in self.events_log if e['event_type'] == 'entered']
        exits = [e for e in self.events_log if e['event_type'] == 'exited']
        total = len(self.events_log)
        summary_text = f"Total Intrusions: {len(intrusions)}   |   Total Exits: {len(exits)}   |   Total Events: {total}"
        tk.Label(win, text=summary_text, font=("Arial", 12), bg="#f0f0f0").pack()

        chart_frame = tk.Frame(win, bg="#f0f0f0")
        chart_frame.pack(fill="both", expand=False, pady=10)

        zone_counts = Counter([e['zone_name'] for e in intrusions])
        zones = list(zone_counts.keys()) or ["No Intrusions"]
        counts = list(zone_counts.values()) or [0]

        fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
        axs[0].bar(zones, counts)
        axs[0].set_title("Intrusions per Zone", fontsize=10)
        axs[0].tick_params(axis='x', rotation=30)

        if total > 0:
            axs[1].pie([len(intrusions), len(exits)], labels=["Intrusions", "Exits"],
                       autopct="%1.1f%%", colors=["red", "green"], startangle=140)
        else:
            axs[1].pie([1], labels=["No Events"], colors=["gray"])
        axs[1].set_title("Intrusions vs Exits", fontsize=10)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        text = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set, font=("Consolas", 10))
        text.pack(fill="both", expand=True)
        scrollbar.config(command=text.yview)

        if self.events_log:
            for e in self.events_log:
                color = "red" if e['event_type'] == 'entered' else "green"
                text.insert("end", f"[{e['timestamp']}] {e['event_type'].upper()} | "
                                   f"ID:{e['object_id']} | Zone:{e['zone_name']} | "
                                   f"Conf:{e['confidence']} | Dur:{e['duration_seconds']}s\n", color)
                text.tag_config(color, foreground=color)
        else:
            text.insert("end", "No intrusion events detected.\n")
        text.config(state="disabled")

        def save_report():
            with open("intrusion_summary.txt", "w") as f:
                for e in self.events_log:
                    f.write(json.dumps(e) + "\n")
            messagebox.showinfo("Saved", "Report saved as intrusion_summary.txt")

        btn_frame = tk.Frame(win, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="💾 Save Report", bg="#2e8b57", fg="white", width=15,
                  command=save_report).pack(side="left", padx=10)
        tk.Button(btn_frame, text="❌ Close", bg="#b22222", fg="white", width=15,
                  command=win.destroy).pack(side="left", padx=10)

        win.mainloop()


# ---------- Main ----------
def main():
    print("=== Variphi Multi-Zone Intrusion Detector ===")
    print("1. bytetrack  2. botsort  3. centroid")
    choice = input("Select tracking method (1/2/3, default=1): ").strip()
    tracking_method = {"1": "bytetrack", "2": "botsort"}.get(choice, "bytetrack")

    detector = MultiZoneIntrusionDetector(tracking_method)
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=(("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"), ("All files", "*.*"))
    )
    if not video_path:
        print("No video selected.")
        return
    print(f"Selected: {os.path.basename(video_path)}")

    if not detector.load_zones():
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            detector.draw_zones_ui(frame)
        else:
            print("Error reading video")
            return

    detector.process_video(video_path)


if __name__ == "__main__":
    main()
