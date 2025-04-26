"""Tests for the model module."""
import unittest

import numpy as np

from geolocator.model import GeoPredictor


class TestGeoPredictor(unittest.TestCase):
    """Test cases for the GeoPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = GeoPredictor()
        
        # Create test image arrays
        # Green-dominant image (forests)
        self.green_image = np.zeros((224, 224, 3))
        self.green_image[:, :, 1] = 0.8  # High green
        self.green_image[:, :, 0] = 0.3  # Low red
        self.green_image[:, :, 2] = 0.3  # Low blue
        
        # Red-dominant image (desert/urban)
        self.red_image = np.zeros((224, 224, 3))
        self.red_image[:, :, 0] = 0.8  # High red
        self.red_image[:, :, 1] = 0.3  # Low green
        self.red_image[:, :, 2] = 0.3  # Low blue
        
        # Blue-dominant image (water/beaches)
        self.blue_image = np.zeros((224, 224, 3))
        self.blue_image[:, :, 2] = 0.8  # High blue
        self.blue_image[:, :, 0] = 0.3  # Low red
        self.blue_image[:, :, 1] = 0.3  # Low green
        
        # Balanced image
        self.balanced_image = np.ones((224, 224, 3)) * 0.5
    
    def test_initialization(self):
        """Test initializing the predictor."""
        self.assertTrue(self.predictor.is_loaded)
        self.assertGreater(len(self.predictor.regions), 0)
    
    def test_predict_green_image(self):
        """Test predicting a green-dominant image."""
        coords, metadata = self.predictor.predict(self.green_image)
        
        # Should predict North America for green-dominant
        self.assertEqual(metadata["region"], "North America")
        self.assertIsInstance(coords, tuple)
        self.assertEqual(len(coords), 2)  # (lat, lon)
        self.assertIsInstance(metadata["confidence"], float)
        self.assertIsInstance(metadata["predicted_class"], int)
    
    def test_predict_red_image(self):
        """Test predicting a red-dominant image."""
        coords, metadata = self.predictor.predict(self.red_image)
        
        # Should predict Asia for red-dominant
        self.assertEqual(metadata["region"], "Asia")
        self.assertIsInstance(coords, tuple)
        self.assertEqual(len(coords), 2)  # (lat, lon)
        self.assertIsInstance(metadata["confidence"], float)
        self.assertIsInstance(metadata["predicted_class"], int)
    
    def test_predict_blue_image(self):
        """Test predicting a blue-dominant image."""
        coords, metadata = self.predictor.predict(self.blue_image)
        
        # Should predict Australia for blue-dominant
        self.assertEqual(metadata["region"], "Australia")
        self.assertIsInstance(coords, tuple)
        self.assertEqual(len(coords), 2)  # (lat, lon)
        self.assertIsInstance(metadata["confidence"], float)
        self.assertIsInstance(metadata["predicted_class"], int)
    
    def test_get_region_info(self):
        """Test getting region information."""
        # Test a coordinate near New York
        region_info = self.predictor.get_region_info(40.7, -74.0)
        self.assertIsInstance(region_info, dict)
        self.assertIn("name", region_info)
        
        # Test a coordinate far from any region centers
        region_info = self.predictor.get_region_info(0, 0)
        self.assertIsInstance(region_info, dict)
        self.assertIn("name", region_info)


if __name__ == '__main__':
    unittest.main()