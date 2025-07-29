import torch
import numpy as np
from PIL import Image, ImageDraw
import json

try:
    from scipy.ndimage import distance_transform_edt, label
except ImportError:
    print("ComfyUI-BMZ-Nodes: SciPy is not installed. Please install it via 'pip install scipy'")

class SAM2MakePointsFromMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mask": ("MASK",),
                "max_points": ("INT", {"default": 3, "min": 1, "max": 1024}),
                "min_distance": ("INT", {"default": 20, "min": 1, "max": 1024}),
                "add_negative_points": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("coords_positive", "coords_negative")
    FUNCTION = "make_points_from_mask"
    CATEGORY = "sam2"

    def _get_random_max_point(self, distance_map):
        max_val = np.max(distance_map)
        if max_val == 0:
            return None
        
        candidates = np.argwhere(distance_map == max_val)
        if len(candidates) == 0:
            # Should not happen if max_val > 0, but as a safeguard
            return None
            
        # Select one candidate randomly
        pt_coords = candidates[np.random.randint(len(candidates))]
        return (pt_coords[0], pt_coords[1]) # (y, x)

    def _find_points(self, mask_np, max_points, min_distance):
        if not np.any(mask_np):
            return []

        # Label connected components
        labeled_mask, num_labels = label(mask_np)

        if num_labels == 0:
            return []
        
        # If there's only one region, use the simple iterative method
        if num_labels == 1:
            distance_map = distance_transform_edt(mask_np)
            points = []
            h, w = distance_map.shape
            y_indices, x_indices = np.indices((h, w))

            for _ in range(max_points):
                if np.max(distance_map) == 0:
                    break
                
                point_coords = self._get_random_max_point(distance_map)
                if point_coords is None:
                    break
                pt_y, pt_x = point_coords

                points.append({"x": int(pt_x), "y": int(pt_y)})
                dist_sq = (x_indices - pt_x)**2 + (y_indices - pt_y)**2
                distance_map[dist_sq < min_distance**2] = 0
            return points

        # --- Multi-region logic ---
        final_points = []
        
        # 1. Find the best point from each distinct region
        best_points_per_region = []
        for i in range(1, num_labels + 1):
            region_mask = (labeled_mask == i)
            dist_map_region = distance_transform_edt(region_mask)
            
            point_coords = self._get_random_max_point(dist_map_region)
            if point_coords is None:
                continue
            pt_y, pt_x = point_coords

            dist = dist_map_region[pt_y, pt_x]
            if dist > 0:
                best_points_per_region.append((dist, {"x": int(pt_x), "y": int(pt_y)}))
        
        if not best_points_per_region:
             return []

        # Sort by how "good" the best point is
        best_points_per_region.sort(key=lambda x: x[0], reverse=True)

        # 2. Add one point from each region, until max_points is met
        num_to_seed = min(max_points, len(best_points_per_region))
        for i in range(num_to_seed):
            final_points.append(best_points_per_region[i][1])

        # 3. If we still need more points, find them iteratively on the global mask
        remaining_points_to_find = max_points - len(final_points)
        if remaining_points_to_find > 0:
            # Create a global distance map and suppress areas around already found points
            global_distance_map = distance_transform_edt(mask_np)
            h, w = global_distance_map.shape
            y_indices, x_indices = np.indices((h, w))
            
            for p in final_points:
                dist_sq = (x_indices - p['x'])**2 + (y_indices - p['y'])**2
                global_distance_map[dist_sq < min_distance**2] = 0

            # Find the rest of the points
            for _ in range(remaining_points_to_find):
                if np.max(global_distance_map) == 0:
                    break

                point_coords = self._get_random_max_point(global_distance_map)
                if point_coords is None:
                    break
                pt_y, pt_x = point_coords

                final_points.append({'x': int(pt_x), 'y': int(pt_y)})
                dist_sq = (x_indices - pt_x)**2 + (y_indices - pt_y)**2
                global_distance_map[dist_sq < min_distance**2] = 0
        
        return final_points

    def _find_negative_points(self, inverted_mask, max_points, min_distance):
        if not np.any(inverted_mask):
            return []

        dist_from_object = distance_transform_edt(inverted_mask)

        h, w = inverted_mask.shape
        # Create a map of distances from the border
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        border_dist_map = np.min([x_coords, y_coords, w - 1 - x_coords, h - 1 - y_coords], axis=0)

        # Combine the two distances to create a score.
        # We want points to be far from the border, but we can be more lenient about
        # how far they are from the object itself.
        # We use sqrt on dist_from_object to reduce its impact, allowing points
        # to be selected closer to the masked area.
        score_map = (dist_from_object ** 0.5) * (border_dist_map ** 0.75)

        points = []
        y_indices, x_indices = np.indices((h, w))

        for _ in range(max_points):
            if np.max(score_map) == 0:
                break
            
            point_coords = self._get_random_max_point(score_map)
            if point_coords is None:
                break
            pt_y, pt_x = point_coords

            points.append({"x": int(pt_x), "y": int(pt_y)})
            dist_sq = (x_indices - pt_x)**2 + (y_indices - pt_y)**2
            score_map[dist_sq < min_distance**2] = 0
        return points

    def make_points_from_mask(self, input_mask, max_points, min_distance, add_negative_points):
        # Ensure SciPy is available
        if 'distance_transform_edt' not in globals() or 'label' not in globals():
            raise ImportError("SciPy is not installed. Please run 'pip install scipy' and restart ComfyUI.")

        mask_np = input_mask.squeeze(0).cpu().numpy()

        # Find positive points
        positive_points = self._find_points(mask_np, max_points, min_distance)
        coords_positive = json.dumps(positive_points)

        # Find negative points if requested
        coords_negative = json.dumps([])
        if add_negative_points:
            inverted_mask = 1.0 - mask_np
            negative_points = self._find_negative_points(inverted_mask, max_points, min_distance)
            coords_negative = json.dumps(negative_points)

        return (coords_positive, coords_negative)

class SAM2VisualizePpoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "coords_positive": ("STRING", {"forceInput": True}),
                "coords_negative": ("STRING", {"forceInput": True}),
                "point_radius": ("INT", {"default": 10, "min": 1, "max": 1024}),
                "point_color_positive": ("STRING", {"default": "green"}),
                "point_color_negative": ("STRING", {"default": "red"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_points"
    CATEGORY = "sam2"

    def visualize_points(self, image, coords_positive, coords_negative, point_radius, point_color_positive, point_color_negative):
        # Convert tensor to PIL images
        images_pil = []
        for i in image:
            i = 255. * i.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            images_pil.append(img)
        
        # Parse coordinates
        try:
            pos_points = json.loads(coords_positive) if coords_positive else []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode positive coordinates JSON: {coords_positive}")
            pos_points = []
        
        try:
            neg_points = json.loads(coords_negative) if coords_negative else []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode negative coordinates JSON: {coords_negative}")
            neg_points = []

        output_images = []
        for img_pil in images_pil:
            draw = ImageDraw.Draw(img_pil)
            
            # Draw positive points
            for point in pos_points:
                x, y = point['x'], point['y']
                bbox = [x - point_radius, y - point_radius, x + point_radius, y + point_radius]
                draw.ellipse(bbox, fill=point_color_positive, outline=point_color_positive)
                
            # Draw negative points
            for point in neg_points:
                x, y = point['x'], point['y']
                bbox = [x - point_radius, y - point_radius, x + point_radius, y + point_radius]
                draw.ellipse(bbox, fill=point_color_negative, outline=point_color_negative)

            # Convert back to tensor
            output_images.append(np.array(img_pil).astype(np.float32) / 255.0)
        
        output_tensor = torch.from_numpy(np.array(output_images))
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "SAM2MakePointsFromMask": SAM2MakePointsFromMask,
    "SAM2VisualizePpoints": SAM2VisualizePpoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM2MakePointsFromMask": "SAM2 Make Points From Mask //BMZ",
    "SAM2VisualizePpoints": "SAM2 Visualize Points //BMZ",
} 