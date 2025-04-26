"""Tests for the image processor module."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from geolocator.image_processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Test cases for the ImageProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
        
        # Create a small test image
        self.test_image = Image.new('RGB', (10, 10), color='red')
        self.test_path = Path("test_image.jpg")
        
    def tearDown(self):
        """Clean up after tests."""
        if self.test_path.exists():
            os.remove(self.test_path)
            
    def test_load_image(self):
        """Test loading an image."""
        # Save the test image temporarily
        self.test_image.save(self.test_path)
        
        # Test loading the image
        loaded_image = self.processor.load_image(self.test_path)
        
        # Verify the image was loaded correctly
        self.assertIsNotNone(loaded_image)
        self.assertEqual(loaded_image.size, (10, 10))
        self.assertEqual(self.processor.image, loaded_image)
    
    def test_load_image_error(self):
        """Test loading an image that doesn't exist."""
        with self.assertRaises(ValueError):
            self.processor.load_image(Path("non_existent_image.jpg"))
    
    def test_preprocess_image(self):
        """Test preprocessing an image."""
        # Load a test image first
        self.test_image.save(self.test_path)
        self.processor.load_image(self.test_path)
        
        # Test preprocessing
        preprocessed = self.processor.preprocess_image(target_size=(8, 8))
        
        # Verify the preprocessing results
        self.assertIsInstance(preprocessed, np.ndarray)
        self.assertEqual(preprocessed.shape, (1, 8, 8, 3))  # (batch, height, width, channels)
    
    def test_preprocess_image_no_image(self):
        """Test preprocessing when no image is loaded."""
        self.processor.image = None
        with self.assertRaises(ValueError):
            self.processor.preprocess_image()
    
    @patch('exifread.process_file')
    def test_extract_exif(self, mock_process_file):
        """Test extracting EXIF data."""
        # Mock EXIF data
        mock_process_file.return_value = {'EXIF Tag': 'EXIF Value'}
        
        # Save test image
        self.test_image.save(self.test_path)
        
        # Extract EXIF data
        exif_data = self.processor.extract_exif(self.test_path)
        
        # Verify extraction
        self.assertEqual(exif_data, {'EXIF Tag': 'EXIF Value'})
        self.assertEqual(self.processor.exif_data, exif_data)
    
    def test_get_gps_from_exif_none(self):
        """Test getting GPS when no EXIF data is available."""
        self.processor.exif_data = {}
        result = self.processor.get_gps_from_exif()
        self.assertIsNone(result)
    
    def test_get_gps_from_exif_incomplete(self):
        """Test getting GPS with incomplete EXIF data."""
        # Missing some GPS tags
        self.processor.exif_data = {
            'GPS GPSLatitude': MagicMock(),
            'GPS GPSLatitudeRef': MagicMock(),
            # Missing longitude data
        }
        result = self.processor.get_gps_from_exif()
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()