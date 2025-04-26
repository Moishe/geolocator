from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import os

import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForImageClassification, AutoImageProcessor

# Ensure TensorFlow can see the Metal plugin
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class GeoPredictor:
    """Predict image location using a pre-trained model."""
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """Initialize the geolocation predictor with a pre-trained model.
        
        Args:
            model_name: HuggingFace model name or path to local model
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        # Predefined regions (we'll expand this or use a more granular approach)
        self.regions = {
            0: {"name": "North America", "center": (40.7128, -74.0060)},
            1: {"name": "South America", "center": (-15.7801, -47.9292)},
            2: {"name": "Europe", "center": (48.8566, 2.3522)},
            3: {"name": "Africa", "center": (-1.2921, 36.8219)},
            4: {"name": "Asia", "center": (35.6762, 139.6503)},
            5: {"name": "Australia", "center": (-33.8688, 151.2093)},
        }
    
    def load_model(self) -> None:
        """Load model and processor."""
        try:
            # For full implementation, we should use a model specifically trained
            # for geolocation. For now, we'll use a standard image classifier
            # and later map its outputs to geographic regions
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = TFAutoModelForImageClassification.from_pretrained(self.model_name)
            self.is_loaded = True
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def predict(self, image_array: np.ndarray) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        """Predict location based on image features.
        
        Args:
            image_array: Preprocessed image as numpy array
            
        Returns:
            Tuple of (latitude, longitude) and additional prediction metadata
        """
        if not self.is_loaded:
            self.load_model()
            
        # For a full implementation, we would:
        # 1. Use a model specifically trained to predict geolocation
        # 2. Map the outputs to geographic coordinates
        
        # For now, we'll use a basic approach that predicts a region class
        # and returns the center of that region
        
        # Process image for the model
        inputs = self.processor(images=image_array, return_tensors="tf")
        
        # Get model prediction
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get predicted class
        predicted_class = tf.argmax(logits, axis=1).numpy()[0]
        
        # Map to one of our predefined regions (using modulo for now)
        region_idx = predicted_class % len(self.regions)
        region = self.regions[region_idx]
        
        # Extract confidence scores
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        confidence = float(probs[predicted_class])
        
        # Return predicted coordinates and metadata
        metadata = {
            "region": region["name"],
            "confidence": confidence,
            "predicted_class": int(predicted_class)
        }
        
        return region["center"], metadata
        
    def get_region_info(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get information about the region containing the coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with region information
        """
        # Simple implementation - find closest region center
        min_dist = float('inf')
        closest_region = None
        
        for region_id, region_data in self.regions.items():
            region_lat, region_lon = region_data["center"]
            
            # Simple Euclidean distance (not accurate for geographic coords but sufficient for demo)
            dist = ((lat - region_lat) ** 2 + (lon - region_lon) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                closest_region = region_data
                
        return closest_region