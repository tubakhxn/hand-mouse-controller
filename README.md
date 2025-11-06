# Hand → Mouse — Virtual Hand-Controlled Cursor

An experimental Python app that uses your webcam and MediaPipe Hands to control a stylized on-screen cursor with hand gestures. The project shows a live webcam window plus a separate "Virtual Mouse" window that renders an animated multi-cursor visual driven by your index finger and pinch gestures.
## Features
- Real-time hand detection with MediaPipe
- Index-finger → virtual cursor mapping

## Requirements
- Windows (PowerShell instructions below). The app may work on macOS/Linux but was developed and tested on Windows.
- Python 3.8+ (3.10/3.11 recommended)
- Webcam (built-in or USB)
## Quick install
From PowerShell run:

```powershell
## Run the app

```powershell
## Usage
- Put your hand in front of the webcam. The top window shows the live video.
- The separate "Virtual Mouse" window displays the stylized cursor(s).
- Move your index finger to move the cursor. Pinch (thumb + index) to perform a click visual.
## Configuration notes
- Smoothing: default is reasonably high (smooth motion). You can tweak `VideoThread.smoothing` in `main.py`.
- MediaPipe rate: the app throttles processing (processes every Nth frame). `VideoThread.process_every_n` controls this. Increasing it lowers CPU at the cost of latency.

## Troubleshooting
- If the camera feed closes unexpectedly, run from a terminal to capture stderr and paste the traceback here.
- If hand detection is flaky, improve lighting and point the index finger clearly toward the camera.
- MediaPipe can print startup logs about TensorFlow/oneDNN; these are informational and not errors.

## Contributing & forks
- To fork this repository on GitHub: go to https://github.com/tubakhxn/hand-mouse-control and click the Fork button (top-right). Then clone your fork locally and make changes in a feature branch.
- Typical flow:
	1. Fork the repo on GitHub
## License & Code of Conduct
- This project is released under the MIT License — see `LICENSE`.
- Please follow the `CODE_OF_CONDUCT.md` in this repository when contributing.

## Contact
- GitHub: https://github.com/tubakhxn
- If you want, I can prepare a ready-to-run release or create a small demo video.

## Copyright
© 2025 Tuba Khan (GitHub: tubakhxn). All rights reserved — MIT License (see `LICENSE`).

If you'd like I can also add a `CONTRIBUTING.md`, a tested PyInstaller `.spec`, or produce an example demo recording of the webcam + virtual mouse. Tell me which and I'll add it.
Hand -> Mouse Control (OpenCV + MediaPipe + PyAutoGUI)

A small Python app that opens your webcam, detects hand gestures with MediaPipe, maps your index finger to the system mouse, and performs click actions using pinch gestures. The UI uses PyQt5 and shows the live camera with a stylized animated panel.

Requirements
- Windows (instructions below target PowerShell)
- Python 3.8+ recommended

Install
Open PowerShell and run:

```powershell
cd "c:\Users\Tuba Khan\Downloads\mouse\hand_mouse_control"
python -m pip install --upgrade pip; python -m pip install -r requirements.txt
```

Run

```powershell
python main.py
```

How it works (quick)
- Move your index finger in front of the camera to move the cursor.
- Pinch (bring thumb tip and index tip together) to perform a left click (debounced).
- Hold an open palm (or no pinch) to keep moving without clicking.

Notes & troubleshooting
- If the mouse jumps too violently, you can add smoothing in the code or increase the movement duration.
- MediaPipe installation on Windows may be heavy; allow a few minutes on first install.

Next improvements you can try
- Add settings for sensitivity, smoothing, and toggle gestures.
- Use multi-hand support and add right-click gesture.
- Add a system tray icon and auto-start option.

Enjoy! If you'd like, I can add config UI controls, smoothing, or alternative gestures next.

Packaging to a single executable (Windows)
---------------------------------------
I included a convenience PowerShell script `build_exe.ps1` that uses PyInstaller to produce a single-file, windowed executable. Packaging MediaPipe can be large and sometimes requires extra hidden imports; the script below is a simple starting point.

From PowerShell run:

```powershell
cd "c:\Users\Tuba Khan\Downloads\mouse\hand_mouse_control"
python -m pip install --upgrade pip; python -m pip install pyinstaller
.\build_exe.ps1
```

Notes:
- If PyInstaller misses some imports (you get errors about missing modules), re-run with the `--hidden-import` flags for the missing modules.
- The produced exe will be in the `dist` folder. Test it on the same machine first.
- Packaging MediaPipe may significantly increase the exe size and can be tricky; if you want, I can create a tested `.spec` file tuned for MediaPipe.