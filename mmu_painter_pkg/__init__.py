"""
MMU Painter - 3D Painting Tool for Multi-Material 3D Printing

Modules:
- core: Data structures and constants
- math_utils: Math utilities (NumPy accelerated)
- codec: MMU segmentation encoder/decoder
- io: File loading/saving (OBJ, 3MF)
- texture: Texture management and quantization
- brush: Sphere brush painting system
- viewport: OpenGL 3D viewport
- ui: Main window and panels
"""

__version__ = "4.1.0"

from .core import *
from .math_utils import *
from .codec import MMUCodec
from .io import load_model, load_obj, load_3mf
from .texture import TextureManager
from .brush import SphereBrush, RayCaster
