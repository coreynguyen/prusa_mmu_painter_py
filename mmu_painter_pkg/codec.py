"""
MMU Segmentation Codec - Encode/decode PrusaSlicer MMU painting format

The encoding format (from PrusaSlicer source):
- String is read BACKWARDS (from end to start)
- Each hex nibble encodes: lower 2 bits = split type, upper 2 bits = color/special
- Split types: 0=leaf, 1=2 children, 2=3 children, 3=4 children
- For leaves with color=3 in upper bits, read next nibble + 3 for colors >= 3
"""

from typing import List, Optional
from .core import SubTriangle


class MMUCodec:
    """Encode/decode MMU segmentation strings"""
    
    @staticmethod
    def decode(encoded: str) -> List[SubTriangle]:
        """Decode segmentation string to sub-triangles"""
        if not encoded:
            return [SubTriangle(
                bary_corners=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                extruder_id=0,
                depth=0
            )]
        
        encoded = encoded.upper().strip()
        pos = [len(encoded) - 1]  # Mutable for nested function
        
        def get_nibble():
            while pos[0] >= 0:
                c = encoded[pos[0]]
                pos[0] -= 1
                if c != ' ':
                    if '0' <= c <= '9':
                        return ord(c) - ord('0')
                    elif 'A' <= c <= 'F':
                        return ord(c) - ord('A') + 10
            raise EOFError()
        
        def parse_node():
            try:
                code = get_nibble()
            except EOFError:
                return None
            
            num_split = code & 0b11
            upper = code >> 2
            
            if num_split == 0:
                # Leaf node
                color = upper
                if color == 3:
                    # Extended color encoding
                    try:
                        color = get_nibble() + 3
                    except EOFError:
                        pass
                return {'color': color, 'children': [None]*4, 'special': 0}
            else:
                # Split node
                node = {'color': 0, 'children': [None]*4, 'special': upper}
                num_children = num_split + 1
                for i in range(num_children):
                    node['children'][i] = parse_node()
                return node
        
        def collect_leaves(node, corners, depth, result):
            """Recursively collect leaf triangles"""
            if all(c is None for c in node['children']):
                result.append(SubTriangle(
                    bary_corners=list(corners),
                    extruder_id=node['color'],
                    depth=depth
                ))
                return
            
            v0, v1, v2 = corners
            t01 = tuple((a+b)/2 for a, b in zip(v0, v1))
            t12 = tuple((a+b)/2 for a, b in zip(v1, v2))
            t20 = tuple((a+b)/2 for a, b in zip(v2, v0))
            
            children = node['children']
            num = sum(1 for c in children if c is not None)
            ss = node['special']
            
            # Child layouts based on split type and special side
            if num == 2:
                layouts = {
                    0: [[t12, v2, v0], [v0, v1, t12]],
                    1: [[t20, v0, v1], [v1, v2, t20]],
                    2: [[t01, v1, v2], [v2, v0, t01]]
                }
            elif num == 3:
                layouts = {
                    0: [[v1, v2, t20], [t01, v1, t20], [v0, t01, t20]],
                    1: [[v2, v0, t01], [t12, v2, t01], [v1, t12, t01]],
                    2: [[v0, v1, t12], [t20, v0, t12], [v2, t20, t12]]
                }
            else:  # 4 children (full subdivision)
                layouts = {0: [[t01, t12, t20], [t12, v2, t20], [t01, v1, t12], [v0, t01, t20]]}
                ss = 0
            
            layout = layouts.get(ss, layouts.get(0, []))
            for i, child in enumerate(children):
                if child is not None and i < len(layout):
                    collect_leaves(child, layout[i], depth + 1, result)
        
        try:
            root = parse_node()
            if root:
                result = []
                root_corners = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
                collect_leaves(root, root_corners, 0, result)
                if result:
                    return result
        except Exception as e:
            print(f"MMU decode error: {e}")
        
        # Return default on any error
        return [SubTriangle(
            bary_corners=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            extruder_id=0,
            depth=0
        )]
    
    @staticmethod
    def encode(paint_data: List[SubTriangle]) -> str:
        """
        Encode sub-triangles to segmentation string.
        
        Note: This is a simplified encoder that handles basic cases.
        Full implementation would rebuild the optimal tree structure.
        """
        if not paint_data:
            return ""
        
        # Single unpainted triangle - empty string
        if len(paint_data) == 1 and paint_data[0].extruder_id == 0:
            return ""
        
        # Single solid color
        if len(paint_data) == 1:
            color = paint_data[0].extruder_id
            if color < 3:
                return format(color << 2, 'x')
            else:
                # Extended color: 'c' followed by (color - 3)
                return format(color - 3, 'x') + 'c'
        
        # For complex paint data, we need to rebuild the tree
        # This is a simplified version - a full implementation would
        # analyze the paint data structure and build an optimal encoding
        
        # Group by depth to understand structure
        max_depth = max(sub.depth for sub in paint_data)
        
        if max_depth == 0:
            # All at root level - shouldn't happen with multiple subs
            return ""
        
        # TODO: Implement full tree reconstruction for complex cases
        # For now, return empty (will lose paint data on save)
        return ""
    
    @staticmethod
    def _build_tree_from_paint(paint_data: List[SubTriangle]) -> Optional[dict]:
        """
        Rebuild tree structure from flat paint data.
        
        This is complex because we need to:
        1. Group sub-triangles by their position in the hierarchy
        2. Determine which split types were used
        3. Reconstruct the tree bottom-up
        """
        # TODO: Implement full tree reconstruction
        # This requires tracking the path to each sub-triangle
        # through the barycentric subdivision hierarchy
        pass
