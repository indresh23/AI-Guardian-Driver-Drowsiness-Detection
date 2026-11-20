# 🛡️ AI Guardian — Driver Drowsiness Detection System

A real-time AI-powered driver drowsiness and fatigue monitoring system built with **Python**, **Streamlit**, and **Google MediaPipe Face Landmarker**. It computes Eye Aspect Ratio (EAR) on normalized 640×480 video streams, tracks a live Drowsy Score, triggers background audio alerts, and suggests live nearby safe rest stops with one-tap emergency contact calling.

---

## 📸 Features

- 🔐 **Glassmorphism Driver Login Gate** — Quick operator & emergency contact setup
- 👁️ **Real-Time Eye Tracking** — 6-point EAR landmark contours rendered live via MediaPipe FaceMesh
- 📊 **5-Card Live Telemetry Dashboard** — Live FPS, Live EAR, Drowsy Score (0–100%), Blink Count, and Alert Count
- 🚨 **Multi-Modal Drowsiness Alerts** — Non-blocking background thread audio (`async_beep()`) + 1.5s `pulse-border` visual warning banner
- 🗺️ **Live Nearby Rest Stop Suggestions** — Automatically fetches real nearby fuel stations, cafes, and rest areas via OpenStreetMap after 3 seconds of sustained drowsiness
- 📞 **Emergency Contact Integration** — One-click `tel:` call button when critical fatigue persists
- 🎥 **Dual Input Support** — Live Webcam (index 0) or Video Upload (`temp_vid.mp4`) with full error handling
- ☁️ **Streamlit Cloud Ready** — Configured with `opencv-python-headless` and `packages.txt`

---

## 🧠 How It Works

The system evaluates the **Eye Aspect Ratio (EAR)** across 6 facial landmarks per eye:

```text
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)