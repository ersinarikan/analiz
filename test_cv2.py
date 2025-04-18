try:
    import cv2
    print(f"OpenCV imported successfully. Version: {cv2.__version__}")
    print(f"OpenCV path: {cv2.__file__}")
except Exception as e:
    print(f"Error importing cv2: {e}")

import sys
print(f"Python executable: {sys.executable}")
print("Python path:")
for path in sys.path:
    print(f"  {path}") 