import numpy as np

def numpy_to_python(obj):
    """Convert NumPy types to Python native types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj 