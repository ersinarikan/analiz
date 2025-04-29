import cv2
import numpy as np
from age_estimator import AgeEstimator

def test_age_estimation():
    # Initialize age estimator
    estimator = AgeEstimator()
    
    # Load a test image (you can replace this with your test image path)
    test_image = cv2.imread('test_image.jpg')
    if test_image is None:
        print("Test image could not be loaded")
        return
    
    # Perform age estimation
    result = estimator.estimate_age(test_image)
    
    if result:
        print("Age Estimation Result:", result)
        print("Estimated Age:", result.get('age'))
        print("Confidence:", result.get('confidence'))
        if 'models' in result:
            print("\nIndividual Model Results:")
            for model_name, model_result in result['models'].items():
                print(f"{model_name}: {model_result}")
    else:
        print("Age estimation failed")

if __name__ == "__main__":
    test_age_estimation() 