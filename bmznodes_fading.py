import torch
import numpy as np
import math

class VideoCrossFade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_sequence1": ("IMAGE",),
                "image_sequence2": ("IMAGE",),
                "method": (["linear", "cosine", "exponential"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video", "debug_info")
    FUNCTION = "blend_sequences"
    CATEGORY = "BMZ/fading"

    def blend_sequences(self, image_sequence1, image_sequence2, method):
        # Determine the number of frames to blend
        num_frames = min(len(image_sequence1), len(image_sequence2))

        # Ensure frames have the same dimensions (height, width, channels)
        if image_sequence1.shape[1:] != image_sequence2.shape[1:]:
            raise ValueError("Image sequences must have the same dimensions (height, width, channels).")

        blended_frames = []
        debug_info_lines = []

        if num_frames == 0:
            return (torch.empty((0, *image_sequence1.shape[1:])), "No frames to blend. One or both inputs are empty.")

        for i in range(num_frames):
            # Get corresponding frames
            frame1 = image_sequence1[i]
            frame2 = image_sequence2[i]

            # Calculate the interpolation factor `t` to avoid 0 and 1
            t = (i + 1) / (num_frames + 1)

            # Calculate alpha based on the selected method
            if method == "linear":
                alpha = t
            elif method == "cosine":
                alpha = (1 - math.cos(t * math.pi)) / 2
            elif method == "exponential":
                alpha = t * t
            else:
                alpha = t

            # Blend the frames
            blended_frame = (1 - alpha) * frame1 + alpha * frame2
            blended_frames.append(blended_frame)
            
            debug_info_lines.append(f"Frame {i+1:03d}/{num_frames}: t={t:.4f}, alpha={alpha:.4f}")

        # Stack the blended frames into a single tensor
        output_video = torch.stack(blended_frames)
        debug_info = "\n".join(debug_info_lines)

        return (output_video, debug_info)

class VideoFadeSaturation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "start_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 8.0, "step": 0.05}),
                "end_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 8.0, "step": 0.05}),
                "fade_duration": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video", "debug_info")
    FUNCTION = "fade_saturation"
    CATEGORY = "BMZ/fading"

    def _adjust_saturation(self, image_tensor, saturation_value):
        saturation_factor = 1.0 + saturation_value
        
        # Standard weights for converting to grayscale (luminance)
        lum_weights = torch.tensor([0.299, 0.587, 0.114], device=image_tensor.device).view(1, 1, 3)
        
        # Ensure grayscale has same dimensions for broadcasting
        grayscale = (image_tensor * lum_weights).sum(dim=-1, keepdim=True).repeat(1, 1, 3)
        
        # Linearly interpolate between the grayscale and original image
        saturated_image = torch.lerp(grayscale, image_tensor, saturation_factor)
        
        return torch.clamp(saturated_image, 0, 1)

    def fade_saturation(self, video, start_value, end_value, fade_duration):
        total_frames = video.shape[0]
        
        if total_frames == 0:
            return (torch.empty_like(video), "")
            
        fade_duration = max(0.0, min(1.0, fade_duration))
        fade_frames_count = int(total_frames * fade_duration)
        
        output_frames = []
        debug_info_lines = []

        for i in range(total_frames):
            current_frame = video[i]
            
            if i < fade_frames_count and fade_frames_count > 0:
                # Interpolate saturation value during the fade
                if fade_frames_count > 1:
                    t = i / (fade_frames_count - 1)
                    current_sat_value = start_value + t * (end_value - start_value)
                else: # fade_frames_count is 1, so this is the only fade frame
                    current_sat_value = end_value
            else:
                # Use the end_value for all frames after the fade
                current_sat_value = end_value
            
            processed_frame = self._adjust_saturation(current_frame, current_sat_value)
            output_frames.append(processed_frame)
            debug_info_lines.append(f"Frame {i+1:03d}/{total_frames}: saturation={current_sat_value:.4f}")

        debug_info = "\n".join(debug_info_lines)
        return (torch.stack(output_frames), debug_info)

class VideoFadeBrightness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "start_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "end_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "fade_duration": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video", "debug_info")
    FUNCTION = "fade_brightness"
    CATEGORY = "BMZ/fading"

    def _adjust_brightness(self, image_tensor, brightness_value):
        return torch.clamp(image_tensor + brightness_value, 0, 1)

    def fade_brightness(self, video, start_value, end_value, fade_duration):
        total_frames = video.shape[0]
        if total_frames == 0:
            return (torch.empty_like(video), "")
        
        fade_duration = max(0.0, min(1.0, fade_duration))
        fade_frames_count = int(total_frames * fade_duration)
        
        output_frames = []
        debug_info_lines = []
        for i in range(total_frames):
            current_frame = video[i]
            if i < fade_frames_count and fade_frames_count > 0:
                if fade_frames_count > 1:
                    t = i / (fade_frames_count - 1)
                    current_val = start_value + t * (end_value - start_value)
                else:
                    current_val = end_value
            else:
                current_val = end_value
            
            processed_frame = self._adjust_brightness(current_frame, current_val)
            output_frames.append(processed_frame)
            debug_info_lines.append(f"Frame {i+1:03d}/{total_frames}: brightness={current_val:.4f}")

        debug_info = "\n".join(debug_info_lines)
        return (torch.stack(output_frames), debug_info)

class VideoFadeContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "start_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 8.0, "step": 0.05}),
                "end_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 8.0, "step": 0.05}),
                "fade_duration": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video", "debug_info")
    FUNCTION = "fade_contrast"
    CATEGORY = "BMZ/fading"

    def _adjust_contrast(self, image_tensor, contrast_value):
        contrast_factor = 1.0 + contrast_value
        mean = torch.full_like(image_tensor, 0.5)
        contrasted_image = torch.lerp(mean, image_tensor, contrast_factor)
        return torch.clamp(contrasted_image, 0, 1)

    def fade_contrast(self, video, start_value, end_value, fade_duration):
        total_frames = video.shape[0]
        if total_frames == 0:
            return (torch.empty_like(video), "")
        
        fade_duration = max(0.0, min(1.0, fade_duration))
        fade_frames_count = int(total_frames * fade_duration)
        
        output_frames = []
        debug_info_lines = []
        for i in range(total_frames):
            current_frame = video[i]
            if i < fade_frames_count and fade_frames_count > 0:
                if fade_frames_count > 1:
                    t = i / (fade_frames_count - 1)
                    current_val = start_value + t * (end_value - start_value)
                else:
                    current_val = end_value
            else:
                current_val = end_value
            
            processed_frame = self._adjust_contrast(current_frame, current_val)
            output_frames.append(processed_frame)
            debug_info_lines.append(f"Frame {i+1:03d}/{total_frames}: contrast={current_val:.4f}")

        debug_info = "\n".join(debug_info_lines)
        return (torch.stack(output_frames), debug_info)

class VideoFadeTemperature:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "start_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "end_value": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "fade_duration": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video", "debug_info")
    FUNCTION = "fade_temperature"
    CATEGORY = "BMZ/fading"

    def _adjust_temperature(self, image_tensor, temperature_value):
        # A value of 0 should do nothing. > 0 is warmer, < 0 is cooler.
        # This is a simplified approach.
        red_shift = temperature_value * 0.1
        blue_shift = -temperature_value * 0.1
        shift = torch.tensor([red_shift, 0.0, blue_shift], device=image_tensor.device).view(1, 1, 3)
        temp_image = image_tensor + shift
        return torch.clamp(temp_image, 0, 1)

    def fade_temperature(self, video, start_value, end_value, fade_duration):
        total_frames = video.shape[0]
        if total_frames == 0:
            return (torch.empty_like(video), "")
        
        fade_duration = max(0.0, min(1.0, fade_duration))
        fade_frames_count = int(total_frames * fade_duration)
        
        output_frames = []
        debug_info_lines = []
        for i in range(total_frames):
            current_frame = video[i]
            if i < fade_frames_count and fade_frames_count > 0:
                if fade_frames_count > 1:
                    t = i / (fade_frames_count - 1)
                    current_val = start_value + t * (end_value - start_value)
                else:
                    current_val = end_value
            else:
                current_val = end_value
            
            processed_frame = self._adjust_temperature(current_frame, current_val)
            output_frames.append(processed_frame)
            debug_info_lines.append(f"Frame {i+1:03d}/{total_frames}: temperature={current_val:.4f}")

        debug_info = "\n".join(debug_info_lines)
        return (torch.stack(output_frames), debug_info)

class VideoFadeColorBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "start_cyan_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_cyan_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "start_magenta_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_magenta_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "start_yellow_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_yellow_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "fade_duration": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "balance_type": (["all", "shadows", "midtones", "highlights"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video", "debug_info")
    FUNCTION = "fade_color_balance"
    CATEGORY = "BMZ/fading"

    def _adjust_color_balance(self, image_tensor, cyan_red, magenta_green, yellow_blue, balance_type):
        color_shift = torch.tensor([cyan_red, magenta_green, yellow_blue], device=image_tensor.device).view(1, 1, 3)

        if balance_type == "all":
            adjusted_image = image_tensor + color_shift
            return torch.clamp(adjusted_image, 0, 1)

        lum_weights = torch.tensor([0.299, 0.587, 0.114], device=image_tensor.device).view(1, 1, 3)
        luminance = (image_tensor * lum_weights).sum(dim=-1, keepdim=True)
        
        shadows_mask = torch.clamp(1.0 - luminance * 2.5, 0, 1)
        highlights_mask = torch.clamp((luminance - 0.75) * 4, 0, 1)
        midtones_mask = 1.0 - shadows_mask - highlights_mask

        if balance_type == "shadows":
            mask = shadows_mask
        elif balance_type == "midtones":
            mask = midtones_mask
        else: # highlights
            mask = highlights_mask
        
        adjusted_image = image_tensor + (color_shift * mask)
        return torch.clamp(adjusted_image, 0, 1)

    def fade_color_balance(self, video, start_cyan_red, end_cyan_red, start_magenta_green, end_magenta_green, start_yellow_blue, end_yellow_blue, fade_duration, balance_type):
        total_frames = video.shape[0]
        if total_frames == 0:
            return (torch.empty_like(video), "")
        
        fade_duration = max(0.0, min(1.0, fade_duration))
        fade_frames_count = int(total_frames * fade_duration)
        
        output_frames = []
        debug_info_lines = []
        for i in range(total_frames):
            current_frame = video[i]
            
            if i < fade_frames_count and fade_frames_count > 0:
                if fade_frames_count > 1:
                    t = i / (fade_frames_count - 1)
                    cr = start_cyan_red + t * (end_cyan_red - start_cyan_red)
                    mg = start_magenta_green + t * (end_magenta_green - start_magenta_green)
                    yb = start_yellow_blue + t * (end_yellow_blue - start_yellow_blue)
                else:
                    cr, mg, yb = end_cyan_red, end_magenta_green, end_yellow_blue
            else:
                cr, mg, yb = end_cyan_red, end_magenta_green, end_yellow_blue
            
            processed_frame = self._adjust_color_balance(current_frame, cr, mg, yb, balance_type)
            output_frames.append(processed_frame)
            debug_info_lines.append(f"Frame {i+1:03d}/{total_frames}: cr={cr:.4f}, mg={mg:.4f}, yb={yb:.4f}")

        debug_info = "\n".join(debug_info_lines)
        return (torch.stack(output_frames), debug_info)

NODE_CLASS_MAPPINGS = {
    "VideoCrossFade //BMZ": VideoCrossFade,
    "VideoFadeSaturation //BMZ": VideoFadeSaturation,
    "VideoFadeBrightness //BMZ": VideoFadeBrightness,
    "VideoFadeContrast //BMZ": VideoFadeContrast,
    "VideoFadeTemperature //BMZ": VideoFadeTemperature,
    "VideoFadeColorBalance //BMZ": VideoFadeColorBalance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCrossFade //BMZ": "Video Cross Fade //BMZ",
    "VideoFadeSaturation //BMZ": "Video Fade Saturation //BMZ",
    "VideoFadeBrightness //BMZ": "Video Fade Brightness //BMZ",
    "VideoFadeContrast //BMZ": "Video Fade Contrast //BMZ",
    "VideoFadeTemperature //BMZ": "Video Fade Temperature //BMZ",
    "VideoFadeColorBalance //BMZ": "Video Fade Color Balance //BMZ"
} 