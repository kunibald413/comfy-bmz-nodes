from .bmznodes import NODE_CLASS_MAPPINGS as bmznodes_mappings, NODE_DISPLAY_NAME_MAPPINGS as bmznodes_display_names
from .bmznodes_loaders import NODE_CLASS_MAPPINGS as loaders_mappings, NODE_DISPLAY_NAME_MAPPINGS as loaders_display_names
from .bmznodes_selectors import NODE_CLASS_MAPPINGS as selectors_mappings, NODE_DISPLAY_NAME_MAPPINGS as selectors_display_names
from .bmznodes_legacy import NODE_CLASS_MAPPINGS as legacy_mappings, NODE_DISPLAY_NAME_MAPPINGS as legacy_display_names
from .bmznodes_memory_profiler import NODE_CLASS_MAPPINGS as memory_profiler_mappings, NODE_DISPLAY_NAME_MAPPINGS as memory_profiler_display_names
from .bmznodes_sam2 import NODE_CLASS_MAPPINGS as sam2_mappings, NODE_DISPLAY_NAME_MAPPINGS as sam2_display_names
from .bmznodes_fading import NODE_CLASS_MAPPINGS as fading_mappings, NODE_DISPLAY_NAME_MAPPINGS as fading_display_names

# Combine dictionaries
NODE_CLASS_MAPPINGS = {**bmznodes_mappings, **loaders_mappings, **selectors_mappings, **legacy_mappings, **memory_profiler_mappings, **sam2_mappings, **fading_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**bmznodes_display_names, **loaders_display_names, **selectors_display_names, **legacy_display_names, **memory_profiler_display_names, **sam2_display_names, **fading_display_names}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']