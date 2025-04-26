from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import io

from PIL import Image
import exifread
import numpy as np


class ImageProcessor:
    """Process images to extract features and EXIF data."""
    
    def __init__(self):
        self.image: Optional[Image.Image] = None
        self.exif_data: Dict[str, Any] = {}
    
    def load_image(self, image_path: Path) -> Image.Image:
        """Load an image from file path."""
        try:
            self.image = Image.open(image_path)
            return self.image
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def preprocess_image(self, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for model input."""
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Resize and convert to RGB if needed
        img = self.image.copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def extract_exif(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF data from image file."""
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                self.exif_data = {k: v for k, v in tags.items()}
                return self.exif_data
        except Exception as e:
            raise ValueError(f"Failed to extract EXIF data: {str(e)}")
    
    def get_gps_from_exif(self) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from EXIF data if available."""
        if not self.exif_data:
            return None
            
        gps_latitude = self.exif_data.get('GPS GPSLatitude')
        gps_latitude_ref = self.exif_data.get('GPS GPSLatitudeRef')
        gps_longitude = self.exif_data.get('GPS GPSLongitude')
        gps_longitude_ref = self.exif_data.get('GPS GPSLongitudeRef')
        
        if None in (gps_latitude, gps_latitude_ref, gps_longitude, gps_longitude_ref):
            return None
            
        def _convert_to_degrees(value):
            d, m, s = (float(v.num) / float(v.den) for v in value.values)
            return d + (m / 60.0) + (s / 3600.0)
            
        lat = _convert_to_degrees(gps_latitude)
        if gps_latitude_ref.values == 'S':
            lat = -lat
            
        lon = _convert_to_degrees(gps_longitude)
        if gps_longitude_ref.values == 'W':
            lon = -lon
            
        return (lat, lon)