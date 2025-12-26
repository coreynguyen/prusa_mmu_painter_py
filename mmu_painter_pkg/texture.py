"""
Texture management for MMU Painter - Optimized Projection
"""

from typing import List, Tuple, Optional, Any
from .core import DEFAULT_COLORS, SubTriangle, MAX_SUBDIVISION_DEPTH
import math

# Check for dependencies
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TextureManager:
    """
    Manages texture loading, color quantization, and UV lookups.
    
    Quantization Methods:
    - kmeans: K-Means clustering (default, good general purpose)
    - median_cut: Median Cut algorithm (better for images with distinct regions)
    - octree: Octree quantization (fast, good color preservation)
    - uniform: Uniform color space division (fastest, less accurate)
    """
    
    def __init__(self):
        self.original_image: Optional[Any] = None
        self.working_image: Optional[Any] = None
        self.segmented_image: Optional[Any] = None
        
        self.palette: List[Tuple[int, int, int]] = list(DEFAULT_COLORS)
        self.labels: Optional[Any] = None  # NumPy array (H, W)
        
        self.flip_h: bool = False
        self.flip_v: bool = False
        self.num_colors: int = 5
        
        # Advanced options
        self.quantize_method: str = 'uniform'  # uniform, kmeans, median_cut, octree
        self.dither: bool = False
        self.blur_radius: int = 0  # Pre-blur to reduce noise
        self.saturation_boost: float = 1.0  # 0.5-2.0
        self.ignore_threshold: int = 0  # Ignore colors with fewer than N pixels
        
        self.width: int = 0
        self.height: int = 0
    
    def load(self, filepath: str) -> bool:
        """Load texture from file"""
        if not HAS_PIL:
            print("PIL not available")
            return False
        
        try:
            self.original_image = Image.open(filepath).convert('RGB')
            self.width = self.original_image.width
            self.height = self.original_image.height
            self._apply_flips()
            return True
        except Exception as e:
            print(f"Failed to load texture: {e}")
            return False
    
    def _apply_flips(self):
        if self.original_image is None: return
        img = self.original_image.copy()
        if self.flip_h: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v: img = img.transpose(Image.FLIP_TOP_BOTTOM)
        self.working_image = img
    
    def set_flip_h(self, flip: bool):
        self.flip_h = flip
        self._apply_flips()
        if self.labels is not None: self.quantize(self.num_colors)
    
    def set_flip_v(self, flip: bool):
        self.flip_v = flip
        self._apply_flips()
        if self.labels is not None: self.quantize(self.num_colors)

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Apply preprocessing before quantization"""
        result = img.copy()
        
        # Apply blur if set (reduces noise, smoother regions)
        if self.blur_radius > 0:
            from PIL import ImageFilter
            result = result.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        # Apply saturation boost if not 1.0
        if abs(self.saturation_boost - 1.0) > 0.01:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(self.saturation_boost)
        
        return result

    def quantize(self, num_colors: int = None, method: str = None) -> List[Tuple[int, int, int]]:
        """
        Quantize image to reduced color palette.
        
        Args:
            num_colors: Number of colors (2-256)
            method: 'kmeans', 'median_cut', 'octree', or 'uniform'
        """
        if self.working_image is None or not HAS_NUMPY: 
            return self.palette
        
        if num_colors is not None: 
            self.num_colors = max(2, min(256, num_colors))
        if method is not None:
            self.quantize_method = method
        
        # Preprocess
        processed = self._preprocess_image(self.working_image)
        arr = np.array(processed)
        h, w, _ = arr.shape
        
        # Choose quantization method
        if self.quantize_method == 'median_cut':
            centroids, labels = self._quantize_median_cut(arr)
        elif self.quantize_method == 'octree':
            centroids, labels = self._quantize_octree(arr)
        elif self.quantize_method == 'uniform':
            centroids, labels = self._quantize_uniform(arr)
        else:  # Default: kmeans
            centroids, labels = self._quantize_kmeans(arr)
        
        self.labels = labels.reshape(h, w)
        
        # Apply dithering if enabled
        if self.dither:
            self._apply_dithering(arr, centroids)
        
        # Filter out colors below threshold
        if self.ignore_threshold > 0:
            centroids, self.labels = self._filter_small_regions(centroids, self.labels)
        
        # Create segmented image for display
        seg_flat = centroids[self.labels.flatten()].astype(np.uint8)
        self.segmented_image = Image.fromarray(seg_flat.reshape(h, w, 3))
        
        # Update palette (Index 0 is reserved for unpainted)
        self.palette = [(128, 128, 128)] + [tuple(c.astype(int)) for c in centroids]
        return self.palette

    def _quantize_kmeans(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """K-Means clustering quantization"""
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3).astype(np.float32)
        
        # Subsample for speed
        max_samples = 100000
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels
        
        # K-means++ initialization
        np.random.seed(42)
        centroids = self._kmeans_plusplus_init(sample, self.num_colors)
        
        # Iterate
        for _ in range(15):
            # Assign to nearest centroid
            dists = np.linalg.norm(sample[:, None] - centroids[None, :], axis=2)
            sample_labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                sample[sample_labels == k].mean(axis=0) if np.any(sample_labels == k) else centroids[k]
                for k in range(self.num_colors)
            ])
            
            if np.allclose(centroids, new_centroids, atol=0.5):
                break
            centroids = new_centroids
        
        # Assign all pixels
        dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
        return centroids, labels
    
    def _kmeans_plusplus_init(self, data: np.ndarray, k: int) -> np.ndarray:
        """K-means++ initialization for better starting centroids"""
        n = len(data)
        centroids = [data[np.random.randint(n)]]
        
        for _ in range(1, k):
            # Distance to nearest centroid
            dists = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0)
            # Probability proportional to distance squared
            probs = dists / dists.sum()
            # Choose next centroid
            idx = np.random.choice(n, p=probs)
            centroids.append(data[idx])
        
        return np.array(centroids)

    def _quantize_median_cut(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Median Cut quantization - better for distinct color regions"""
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3).astype(np.float32)
        
        def median_cut_recursive(px, depth):
            if depth == 0 or len(px) == 0:
                return [px.mean(axis=0)] if len(px) > 0 else []
            
            # Find channel with greatest range
            ranges = px.max(axis=0) - px.min(axis=0)
            channel = np.argmax(ranges)
            
            # Sort by that channel and split at median
            sorted_idx = np.argsort(px[:, channel])
            mid = len(px) // 2
            
            return (median_cut_recursive(px[sorted_idx[:mid]], depth - 1) +
                    median_cut_recursive(px[sorted_idx[mid:]], depth - 1))
        
        # Calculate depth needed for num_colors
        depth = int(np.ceil(np.log2(self.num_colors)))
        centroids = np.array(median_cut_recursive(pixels, depth))[:self.num_colors]
        
        # Pad if we got fewer colors
        while len(centroids) < self.num_colors:
            centroids = np.vstack([centroids, centroids[-1]])
        
        # Assign labels
        dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
        return centroids, labels

    def _quantize_octree(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Octree quantization - fast with good color preservation"""
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3)
        
        # Reduce to 5 bits per channel for octree binning
        reduced = (pixels >> 3).astype(np.uint16)
        # Create unique color key
        keys = reduced[:, 0] * 1024 + reduced[:, 1] * 32 + reduced[:, 2]
        
        # Count unique colors
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        
        # Get top N colors by count
        top_indices = np.argsort(-counts)[:self.num_colors]
        top_keys = unique_keys[top_indices]
        
        # Decode keys back to colors
        centroids = np.zeros((self.num_colors, 3), dtype=np.float32)
        for i, key in enumerate(top_keys):
            r = ((key // 1024) << 3) + 4  # Add 4 to center the bin
            g = (((key % 1024) // 32) << 3) + 4
            b = ((key % 32) << 3) + 4
            centroids[i] = [r, g, b]
        
        # Assign labels
        dists = np.linalg.norm(pixels[:, None].astype(np.float32) - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
        return centroids, labels

    def _quantize_uniform(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform quantization - fastest, divides color space evenly"""
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3).astype(np.float32)
        
        # Calculate levels per channel
        levels = int(np.ceil(self.num_colors ** (1/3)))
        step = 256.0 / levels
        
        # Create uniform centroids
        centroids = []
        for r in range(levels):
            for g in range(levels):
                for b in range(levels):
                    if len(centroids) < self.num_colors:
                        centroids.append([
                            r * step + step / 2,
                            g * step + step / 2,
                            b * step + step / 2
                        ])
        centroids = np.array(centroids, dtype=np.float32)
        
        # Assign labels
        dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
        return centroids, labels

    def _apply_dithering(self, arr: np.ndarray, centroids: np.ndarray):
        """Apply Floyd-Steinberg dithering to labels"""
        h, w, _ = arr.shape
        pixels = arr.astype(np.float32).copy()
        
        for y in range(h):
            for x in range(w):
                old_pixel = pixels[y, x]
                # Find nearest centroid
                dists = np.sum((centroids - old_pixel) ** 2, axis=1)
                new_idx = np.argmin(dists)
                self.labels[y, x] = new_idx
                
                # Calculate error
                error = old_pixel - centroids[new_idx]
                
                # Distribute error (Floyd-Steinberg)
                if x + 1 < w:
                    pixels[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        pixels[y + 1, x - 1] += error * 3 / 16
                    pixels[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        pixels[y + 1, x + 1] += error * 1 / 16

    def _filter_small_regions(self, centroids: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove colors with fewer than threshold pixels, reassign to nearest"""
        unique, counts = np.unique(labels, return_counts=True)
        
        # Find colors to keep
        keep_mask = counts >= self.ignore_threshold
        keep_indices = unique[keep_mask]
        
        if len(keep_indices) < 2:
            return centroids, labels  # Don't filter if too few would remain
        
        # Remap labels
        new_centroids = centroids[keep_indices]
        new_labels = labels.copy()
        
        for old_idx in unique[~keep_mask]:
            mask = labels == old_idx
            if mask.any():
                # Find nearest remaining centroid
                dists = np.sum((new_centroids - centroids[old_idx]) ** 2, axis=1)
                nearest = np.argmin(dists)
                new_labels[mask] = nearest
        
        return new_centroids, new_labels

    def get_image_as_bytes(self, which: str = 'working'):
        img = {'original': self.original_image, 'working': self.working_image, 'segmented': self.segmented_image}.get(which)
        return (img.tobytes(), img.width, img.height) if img else None

    # =========================================================================
    # OPTIMIZED PROJECTION LOGIC - NumPy Vectorized
    # =========================================================================

    def project_texture_to_mesh(self, mesh, max_depth: int, use_quantized: bool = True) -> bool:
        """
        Project texture colors onto mesh using optimized sampling.
        """
        if self.labels is None: 
            return False
        
        print(f"Projecting texture (Depth {max_depth})...")
        
        h, w = self.labels.shape
        total_tris = len(mesh.triangles)
        processed = 0
        skipped = 0
        
        for i, tri in enumerate(mesh.triangles):
            if not tri.uv:
                skipped += 1
                continue
            
            # Get UV bounds for quick uniform check
            us = [p[0] for p in tri.uv]
            vs = [p[1] for p in tri.uv]
            
            u_min, u_max = min(us), max(us)
            v_min, v_max = min(vs), max(vs)
            
            # Convert to pixel coords (clamp to valid range)
            x0 = max(0, min(w-1, int(u_min * (w - 1))))
            x1 = max(0, min(w-1, int(u_max * (w - 1))))
            y0 = max(0, min(h-1, int((1.0 - v_max) * (h - 1))))
            y1 = max(0, min(h-1, int((1.0 - v_min) * (h - 1))))
            
            # Ensure valid slice
            if x1 < x0: x0, x1 = x1, x0
            if y1 < y0: y0, y1 = y1, y0
            
            # Slice region
            region = self.labels[y0:y1+1, x0:x1+1]
            
            # Empty or single-pixel region - sample center
            if region.size <= 1:
                cu = (u_min + u_max) * 0.5
                cv = (v_min + v_max) * 0.5
                cx = max(0, min(w-1, int(cu * (w - 1))))
                cy = max(0, min(h-1, int((1.0 - cv) * (h - 1))))
                color_idx = int(self.labels[cy, cx]) + 1
                tri.paint_data = [SubTriangle([(1,0,0), (0,1,0), (0,0,1)], color_idx, 0)]
                processed += 1
                continue
            
            # Fast path: Check if bounding box is uniform
            first_color = region.flat[0]
            if np.all(region == first_color):
                tri.paint_data = [SubTriangle(
                    [(1,0,0), (0,1,0), (0,0,1)], 
                    int(first_color) + 1, 
                    0
                )]
                processed += 1
                continue
            
            # Need subdivision
            new_paint = []
            self._recursive_subdivide_fast(
                tri.uv, 
                [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], 
                0, 
                max_depth, 
                new_paint,
                w, h
            )
            tri.paint_data = new_paint
            processed += 1
            
            # Progress update every 2000 triangles
            if processed % 2000 == 0:
                print(f"  Processed {processed}/{total_tris} triangles...")
        
        print(f"Projection complete. {processed} projected, {skipped} skipped (no UVs).")
        return True

    def _recursive_subdivide_fast(self, tri_uvs, bary, depth, max_depth, result, w, h):
        """
        Optimized recursive subdivision with 7-point sampling to catch edge crossings.
        Samples: 3 corners + 3 edge midpoints + 1 center
        """
        b0, b1, b2 = bary
        
        # Precompute UV coordinates for all 7 sample points
        # Corner 0
        u0 = b0[0]*tri_uvs[0][0] + b0[1]*tri_uvs[1][0] + b0[2]*tri_uvs[2][0]
        v0 = b0[0]*tri_uvs[0][1] + b0[1]*tri_uvs[1][1] + b0[2]*tri_uvs[2][1]
        # Corner 1
        u1 = b1[0]*tri_uvs[0][0] + b1[1]*tri_uvs[1][0] + b1[2]*tri_uvs[2][0]
        v1 = b1[0]*tri_uvs[0][1] + b1[1]*tri_uvs[1][1] + b1[2]*tri_uvs[2][1]
        # Corner 2
        u2 = b2[0]*tri_uvs[0][0] + b2[1]*tri_uvs[1][0] + b2[2]*tri_uvs[2][0]
        v2 = b2[0]*tri_uvs[0][1] + b2[1]*tri_uvs[1][1] + b2[2]*tri_uvs[2][1]
        
        # Edge midpoints
        um01, vm01 = (u0 + u1) * 0.5, (v0 + v1) * 0.5
        um12, vm12 = (u1 + u2) * 0.5, (v1 + v2) * 0.5
        um02, vm02 = (u0 + u2) * 0.5, (v0 + v2) * 0.5
        
        # Center
        uc, vc = (u0 + u1 + u2) / 3.0, (v0 + v1 + v2) / 3.0
        
        # Sample all 7 points - inline for speed
        wm1, hm1 = w - 1, h - 1
        
        def sample(u, v):
            x = int(u * wm1)
            y = int((1.0 - v) * hm1)
            # Clamp instead of modulo to avoid wrapping artifacts
            x = max(0, min(wm1, x))
            y = max(0, min(hm1, y))
            return int(self.labels[y, x])
        
        c0 = sample(u0, v0)
        c1 = sample(u1, v1)
        c2 = sample(u2, v2)
        cm01 = sample(um01, vm01)
        cm12 = sample(um12, vm12)
        cm02 = sample(um02, vm02)
        cc = sample(uc, vc)
        
        # Check if all 7 samples are the same color
        first = c0
        is_uniform = (c1 == first and c2 == first and 
                      cm01 == first and cm12 == first and cm02 == first and 
                      cc == first)
        
        # Early termination: uniform color or max depth
        if is_uniform or depth >= max_depth:
            # Use center color (+1 for 1-indexed palette)
            result.append(SubTriangle(list(bary), cc + 1, depth))
            return
        
        # Subdivide into 4 sub-triangles
        m01 = ((b0[0]+b1[0])*0.5, (b0[1]+b1[1])*0.5, (b0[2]+b1[2])*0.5)
        m12 = ((b1[0]+b2[0])*0.5, (b1[1]+b2[1])*0.5, (b1[2]+b2[2])*0.5)
        m02 = ((b0[0]+b2[0])*0.5, (b0[1]+b2[1])*0.5, (b0[2]+b2[2])*0.5)
        
        # Center triangle
        self._recursive_subdivide_fast(tri_uvs, [m01, m12, m02], depth+1, max_depth, result, w, h)
        # Corner triangles
        self._recursive_subdivide_fast(tri_uvs, [b0, m01, m02], depth+1, max_depth, result, w, h)
        self._recursive_subdivide_fast(tri_uvs, [m01, b1, m12], depth+1, max_depth, result, w, h)
        self._recursive_subdivide_fast(tri_uvs, [m02, m12, b2], depth+1, max_depth, result, w, h)

    def get_color_at_uv(self, u, v):
        if self.labels is None: 
            return 0
        h, w = self.labels.shape
        x = int(u * (w - 1)) % w
        y = int((1.0 - v) * (h - 1)) % h
        return int(self.labels[y, x]) + 1