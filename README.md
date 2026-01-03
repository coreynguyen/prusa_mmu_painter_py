# MMU Painter v4 (Python / Qt Prototype)

![Status](https://img.shields.io/badge/status-Legacy%20%2F%20Prototype-orange.svg)
![License](https://img.shields.io/badge/license-Unlicense-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)
![Dependencies](https://img.shields.io/badge/dependencies-PySide6%20%7C%20OpenGL-lightgrey.svg)

![Tool Preview](preview.jpg)

## ⚠️ Legacy Notice
**This is the Python/PySide6 prototype of the MMU Painter.**
While fully functional, it relies on CPU-bound recursion for mesh subdivision. It performs well on low-poly assets but is **significantly slower** on high-density meshes compared to the production [C# / WPF version](https://github.com/coreynguyen/prusa_mmu_painter). It is archived here for educational purposes and for cross-platform users (Linux/macOS).

---

## 🖌️ Overview

**MMU Painter v4** is a direct-to-mesh painting utility for multi-material 3D printing (Prusa MMU, Bambu AMS). It solves the problem of painting low-poly models by implementing **Dynamic Sub-Triangle Subdivision**.

Instead of painting existing vertices (which requires millions of polygons for sharp lines), this tool mathematically slices triangles in real-time as the brush passes over them, creating high-resolution paint boundaries on low-resolution geometry.

### ✨ Key Features

* **⚪ Sphere Brush Projection:** Paints using a volumetric sphere with surface-tangent projection logic.
* **📐 Dynamic Subdivision:** Automatically subdivides mesh faces up to a user-defined depth (e.g., $4^6$) only where paint is applied.
* **🎨 Texture Quantization:** Includes a K-Means clustering engine to reduce photo textures to printable filament palettes (e.g., 5 colors).
* **🛡️ Masking Engine:** Protect specific faces from being painted using a Mask/Unmask tool.
* **🔌 Formats:** Supports `.obj` and `.3mf` (handling both Y-up and Z-up coordinate systems automatically).

---

## 🛠️ Installation

### Prerequisites
You need **Python 3.8+** and a few math/rendering libraries.

```bash
pip install numpy PySide6 PyOpenGL Pillow
