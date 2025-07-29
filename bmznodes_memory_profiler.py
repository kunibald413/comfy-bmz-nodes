import torch
import gc
import sys
import threading
import time
from collections import OrderedDict

# Shared state between nodes
class MonitoringState:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.last_report = ""
        self.problematic_modules = set()
        self.min_model_size_mb = 100.0

# Create a singleton instance to share state
monitoring_state = MonitoringState()

class MemoryProfilerStart:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "interval_seconds": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 60.0, "step": 1.0}),
            "min_model_size_mb": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 10.0}),
        },
        "optional": {
            "trigger_input": ("*", {}),  # Generic input that accepts any type
        }}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "start_profiling"
    CATEGORY = "utils"
    
    def __init__(self):
        # Print an initial message to confirm the module loaded
        self._print_with_attention("MEMORY PROFILER START NODE LOADED")
    
    def _print_with_attention(self, message):
        """Print messages in a way that's hard to miss in the console"""
        border = "!" * 100
        print("\n" + border)
        print(message)
        print(border + "\n")
        
        # Also try to flush stdout to ensure immediate output
        sys.stdout.flush()
    
    def start_profiling(self, interval_seconds, min_model_size_mb=100.0, trigger_input=None):
        return (self.start_monitoring(interval_seconds, min_model_size_mb),)
    
    def start_monitoring(self, interval_seconds, min_model_size_mb=100.0):
        monitoring_state.min_model_size_mb = min_model_size_mb
        
        def monitor_loop():
            self._print_with_attention(f"MEMORY MONITORING STARTED - REPORTING EVERY {interval_seconds} SECONDS")
            self._print_with_attention(f"FILTERING OUT MODELS SMALLER THAN {monitoring_state.min_model_size_mb} MB")
            
            while monitoring_state.monitoring:
                try:
                    report = self.generate_memory_report()
                    
                    # Super visible report formatting
                    output = "\n" + "#" * 100
                    output += "\n" + " " * 30 + "MEMORY USAGE REPORT" + " " * 30
                    output += "\n" + "#" * 100 + "\n"
                    output += report
                    output += "\n" + "#" * 100 + "\n"
                    
                    print(output)
                    sys.stdout.flush()  # Force flush to ensure output appears
                    
                    monitoring_state.last_report = report
                except Exception as e:
                    self._print_with_attention(f"ERROR IN MEMORY MONITOR: {str(e)}")
                
                time.sleep(interval_seconds)
        
        # Stop any existing monitoring
        if monitoring_state.monitoring:
            self.stop_monitoring()
            
        monitoring_state.monitoring = True
        monitoring_state.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_state.monitor_thread.start()
        
        return "Memory monitoring started"
    
    def stop_monitoring(self):
        if monitoring_state.monitoring:
            monitoring_state.monitoring = False
            if monitoring_state.monitor_thread:
                monitoring_state.monitor_thread.join(timeout=1.0)
                monitoring_state.monitor_thread = None
            self._print_with_attention("MEMORY MONITORING STOPPED")
            return "Memory monitoring stopped"
        return "Memory monitoring was not active"
    
    def generate_memory_report(self):
        gpu_models = []
        cpu_models = []
        total = {"gpu": 0, "cpu": 0}
        model_count = {"gpu": {}, "cpu": {}}
        
        # Skip known problematic modules
        skip_module_names = [
            "openai", "sounddevice", "_sounddevice", "_proxy", 
            "proxied", "audio", "portaudio"
        ]
        
        # Safety: use a list to avoid "dictionary changed during iteration" errors
        objects_to_check = list(gc.get_objects())
        
        # First, scan through to find PyTorch models
        for obj in objects_to_check:
            try:
                # Skip objects from problematic modules
                module_name = getattr(obj.__class__, "__module__", "")
                if any(skip_name in module_name.lower() for skip_name in skip_module_names):
                    continue
                    
                if isinstance(obj, torch.nn.Module):
                    # Get meaningful name when possible
                    try:
                        name = getattr(obj, 'name', obj.__class__.__name__)
                        obj_type = obj.__class__.__name__
                        
                        mem_params, mem_buffers = self.calculate_memory(obj)
                        device = self.get_device(obj)
                        
                        total_mem = mem_params + mem_buffers
                        total_mem_mb = total_mem / 1024**2
                        
                        # Skip models smaller than the minimum size
                        if total_mem_mb < monitoring_state.min_model_size_mb:
                            continue
                        
                        # Create a key for model type detection
                        # Round memory to handle tiny differences
                        mem_key = f"{obj_type}_{name}_{round(total_mem_mb, 1)}_{device}"
                        
                        # For counting duplicates
                        if mem_key not in model_count[device]:
                            model_info = {
                                "name": name,
                                "type": obj_type,
                                "device": device,
                                "memory_params": mem_params,
                                "memory_buffers": mem_buffers,
                                "memory": total_mem,
                                "count": 1,
                                "key": mem_key
                            }
                            
                            # Add to appropriate list based on device
                            if device == "gpu":
                                gpu_models.append(model_info)
                            else:
                                cpu_models.append(model_info)
                            
                            model_count[device][mem_key] = 0
                        
                        # Count this instance
                        model_count[device][mem_key] += 1
                        total[device] += total_mem
                    except Exception as e:
                        # Skip this object if there's an error processing it
                        pass
            except Exception:
                # Skip any object that causes problems on inspection
                continue
        
        # Update counts in the model objects
        for model in gpu_models:
            model["count"] = model_count["gpu"][model["key"]]
            
        for model in cpu_models:
            model["count"] = model_count["cpu"][model["key"]]
        
        # Sort both lists by memory usage (largest first)
        gpu_models.sort(key=lambda x: x["memory"], reverse=True)
        cpu_models.sort(key=lambda x: x["memory"], reverse=True)
        
        # Generate report
        report = []
        total_models = sum(model_count["gpu"].values()) + sum(model_count["cpu"].values())
        unique_models = len(gpu_models) + len(cpu_models)
        report.append(f"TOTAL MODELS FOUND (>= {monitoring_state.min_model_size_mb}MB): {total_models} ({unique_models} unique types)")
        report.append("-" * 80)
        
        # GPU Models section
        report.append(f"GPU MODELS: {sum(model_count['gpu'].values())} instances of {len(gpu_models)} types")
        report.append("-" * 80)
        
        for info in gpu_models:
            count_str = f"[x{info['count']}]" if info["count"] > 1 else ""
            line = (f"{info['name']} ({info['type']}) {count_str}: "
                   f"{info['memory']/1024**2:.2f}MB "
                   f"(Params: {info['memory_params']/1024**2:.2f}MB, "
                   f"Buffers: {info['memory_buffers']/1024**2:.2f}MB) "
                   f"on GPU")
            report.append(line)
        
        # CPU Models section
        report.append("-" * 80)
        report.append(f"CPU MODELS: {sum(model_count['cpu'].values())} instances of {len(cpu_models)} types")
        report.append("-" * 80)
        
        for info in cpu_models:
            count_str = f"[x{info['count']}]" if info["count"] > 1 else ""
            line = (f"{info['name']} ({info['type']}) {count_str}: "
                   f"{info['memory']/1024**2:.2f}MB "
                   f"(Params: {info['memory_params']/1024**2:.2f}MB, "
                   f"Buffers: {info['memory_buffers']/1024**2:.2f}MB) "
                   f"on CPU")
            report.append(line)
        
        report.append("-" * 80)
        report.append(f"TOTAL GPU MEMORY: {total['gpu']/1024**2:.2f}MB")
        report.append(f"TOTAL CPU MEMORY: {total['cpu']/1024**2:.2f}MB")
        
        return "\n".join(report)

    def calculate_memory(self, model):
        # Calculate both parameters AND buffers
        try:
            mem_params = sum(
                p.numel() * p.element_size() 
                for p in model.parameters() 
                if p is not None
            )
        except:
            mem_params = 0
            
        try:
            mem_buffers = sum(
                b.numel() * b.element_size() 
                for b in model.buffers() 
                if b is not None
            )
        except:
            mem_buffers = 0
            
        return mem_params, mem_buffers

    def get_device(self, model):
        try:
            for param in model.parameters():
                if param is not None:
                    return "gpu" if param.is_cuda else "cpu"
        except:
            pass
            
        try:
            for buffer in model.buffers():
                if buffer is not None:
                    return "gpu" if buffer.is_cuda else "cpu"
        except:
            pass
            
        return "cpu"  # Default fallback

class MemoryProfilerStop:
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {
            "trigger_input": ("*", {}),  # Generic input that accepts any type
        }}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "stop_profiling"
    CATEGORY = "utils"
    
    def __init__(self):
        # Print an initial message to confirm the module loaded
        self._print_with_attention("MEMORY PROFILER STOP NODE LOADED")
    
    def _print_with_attention(self, message):
        """Print messages in a way that's hard to miss in the console"""
        border = "!" * 100
        print("\n" + border)
        print(message)
        print(border + "\n")
        
        # Also try to flush stdout to ensure immediate output
        sys.stdout.flush()
    
    def stop_profiling(self, trigger_input=None):
        return (self.stop_monitoring(),)
    
    def stop_monitoring(self):
        if monitoring_state.monitoring:
            monitoring_state.monitoring = False
            if monitoring_state.monitor_thread:
                monitoring_state.monitor_thread.join(timeout=1.0)
                monitoring_state.monitor_thread = None
            self._print_with_attention("MEMORY MONITORING STOPPED")
            return "Memory monitoring stopped"
        return "Memory monitoring was not active"

# For backwards compatibility
class MemoryProfilerEnhanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "interval_seconds": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 60.0, "step": 1.0}),
            "start_monitoring": ("BOOLEAN", {"default": True}),
            "min_model_size_mb": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 10.0}),
        },
        "optional": {
            "trigger_input": ("*", {}),  # Generic input that accepts any type
        }}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "profile_memory"
    CATEGORY = "utils"
    
    def __init__(self):
        self.starter = MemoryProfilerStart()
        self.stopper = MemoryProfilerStop()
        self._print_with_attention("MEMORY PROFILER LOADED - MONITORING WILL START WHEN NODE IS EXECUTED")
    
    def _print_with_attention(self, message):
        """Print messages in a way that's hard to miss in the console"""
        border = "!" * 100
        print("\n" + border)
        print(message)
        print(border + "\n")
        
        # Also try to flush stdout to ensure immediate output
        sys.stdout.flush()
    
    def profile_memory(self, interval_seconds, start_monitoring, min_model_size_mb=100.0, trigger_input=None):
        if start_monitoring:
            return (self.starter.start_monitoring(interval_seconds, min_model_size_mb),)
        else:
            return (self.stopper.stop_monitoring(),)

NODE_CLASS_MAPPINGS = {
    "MemoryProfilerEnhanced": MemoryProfilerEnhanced,
    "MemoryProfilerStart": MemoryProfilerStart,
    "MemoryProfilerStop": MemoryProfilerStop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryProfilerEnhanced": "Memory Profiler",
    "MemoryProfilerStart": "Memory Profiler Start",
    "MemoryProfilerStop": "Memory Profiler Stop"
}