# build_exe.ps1
# Simple script to build a single-file executable for Windows using PyInstaller.
# Run this from PowerShell in the project folder.

pyinstaller --noconfirm --onefile --windowed main.py \
    --name HandMouseController

Write-Host "Build finished. Check the 'dist' folder for HandMouseController.exe"
