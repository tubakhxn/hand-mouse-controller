# Contributing

Thanks for your interest in contributing! Small, focused contributions are welcome.

Getting started

1. Fork the repository on GitHub: https://github.com/tubakhxn/hand-mouse-control
2. Clone your fork locally:

```powershell
git clone https://github.com/tubakhxn/hand-mouse-controller
cd hand-mouse-control
```

3. Create a feature branch:

```powershell
git checkout -b feat/your-feature
```

4. Make changes, run the app locally, and add tests if appropriate.
5. Commit and push, then open a Pull Request against `tubakhxn/hand-mouse-control`.

Coding style

- Keep changes small and focused.
- Use descriptive commit messages.
- If you modify `main.py` behavior, please update the README with instructions.

Testing

- There are no automated tests included yet. If you add tests, prefer pytest and put tests under `tests/`.

Packaging

- A helper script `build_exe.ps1` exists for packaging with PyInstaller. Packaging MediaPipe may need additional hidden imports; please test the resulting exe on the target Windows machine.

Questions

- Open an issue on GitHub and tag @tubakhxn so maintainers notice it promptly.

