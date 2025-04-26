"""Tests for the main application."""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import typer
from PIL import Image

import main


class TestMainApp(unittest.TestCase):
    """Test cases for the main functionality directly testing the function."""

    @patch('geolocator.image_processor.ImageProcessor.load_image')
    @patch('geolocator.image_processor.ImageProcessor.preprocess_image')
    @patch('geolocator.model.GeoPredictor.predict')
    @patch('geolocator.geocoder.Geocoder.reverse_geocode')
    @patch('geolocator.geocoder.Geocoder.format_location')
    @patch('rich.print')
    @patch('rich.console.Console.print')
    @patch('rich.console.Console.status')
    def test_main_function_without_exif(self, mock_status, mock_console_print, mock_print, 
                                        mock_format_location, mock_reverse_geocode, mock_predict, 
                                        mock_preprocess, mock_load_image):
        """Test the main function without EXIF verification."""
        # Setup mocks
        mock_image = MagicMock(spec=Image.Image)
        mock_load_image.return_value = mock_image
        
        mock_array = MagicMock()
        mock_preprocess.return_value = mock_array
        
        mock_predict.return_value = ((40.7128, -74.0060), {
            "region": "North America",
            "confidence": 0.8,
            "predicted_class": 0
        })
        
        mock_reverse_geocode.return_value = {
            "address": {
                "city": "New York",
                "state": "New York",
                "country": "USA"
            }
        }
        mock_format_location.return_value = "New York, New York, USA"
        
        # Mock context manager
        mock_status.return_value.__enter__ = MagicMock()
        mock_status.return_value.__exit__ = MagicMock()
        
        # Create a temporary file path
        image_path = Path("test_image.jpg")
        
        # Test the main function
        main.main(image_path, False)
        
        # Verify the correct methods were called
        mock_load_image.assert_called_once_with(image_path)
        mock_preprocess.assert_called_once()
        mock_predict.assert_called_once_with(mock_array)
        mock_reverse_geocode.assert_called_once_with(40.7128, -74.0060)
        mock_format_location.assert_called_once()
    
    @patch('geolocator.image_processor.ImageProcessor.load_image')
    @patch('geolocator.image_processor.ImageProcessor.preprocess_image')
    @patch('geolocator.image_processor.ImageProcessor.extract_exif')
    @patch('geolocator.image_processor.ImageProcessor.get_gps_from_exif')
    @patch('geolocator.model.GeoPredictor.predict')
    @patch('geolocator.geocoder.Geocoder.reverse_geocode')
    @patch('geolocator.geocoder.Geocoder.format_location')
    @patch('geolocator.geocoder.Geocoder.calculate_distance')
    @patch('rich.print')
    @patch('rich.console.Console.print')
    @patch('rich.console.Console.status')
    def test_main_function_with_exif(self, mock_status, mock_console_print, mock_print,
                                     mock_calculate_distance, mock_format_location, 
                                     mock_reverse_geocode, mock_predict, mock_get_gps,
                                     mock_extract_exif, mock_preprocess, mock_load_image):
        """Test the main function with EXIF verification."""
        # Setup mocks
        mock_image = MagicMock(spec=Image.Image)
        mock_load_image.return_value = mock_image
        
        mock_array = MagicMock()
        mock_preprocess.return_value = mock_array
        
        mock_extract_exif.return_value = {}
        mock_get_gps.return_value = (34.0522, -118.2437)  # LA coords
        
        mock_predict.return_value = ((40.7128, -74.0060), {  # NY coords
            "region": "North America",
            "confidence": 0.8,
            "predicted_class": 0
        })
        
        # Need to handle multiple calls to reverse_geocode
        def reverse_geocode_side_effect(lat, lon):
            if lat == 40.7128 and lon == -74.0060:  # NY
                return {
                    "address": {
                        "city": "New York",
                        "state": "New York",
                        "country": "USA"
                    }
                }
            elif lat == 34.0522 and lon == -118.2437:  # LA
                return {
                    "address": {
                        "city": "Los Angeles",
                        "state": "California",
                        "country": "USA"
                    }
                }
            return {}
            
        mock_reverse_geocode.side_effect = reverse_geocode_side_effect
        
        # Need to handle multiple calls to format_location
        def format_location_side_effect(data):
            if "city" in data.get("address", {}) and data["address"]["city"] == "New York":
                return "New York, New York, USA"
            elif "city" in data.get("address", {}) and data["address"]["city"] == "Los Angeles":
                return "Los Angeles, California, USA"
            return "Unknown"
            
        mock_format_location.side_effect = format_location_side_effect
        
        mock_calculate_distance.return_value = 3944.0  # NY to LA distance
        
        # Mock context manager
        mock_status.return_value.__enter__ = MagicMock()
        mock_status.return_value.__exit__ = MagicMock()
        
        # Create a temporary file path
        image_path = Path("test_image.jpg")
        
        # Test the main function
        main.main(image_path, True)
        
        # Verify the correct methods were called
        mock_load_image.assert_called_once_with(image_path)
        mock_preprocess.assert_called_once()
        mock_extract_exif.assert_called_once_with(image_path)
        mock_get_gps.assert_called_once()
        mock_predict.assert_called_once_with(mock_array)
        self.assertEqual(mock_reverse_geocode.call_count, 2)
        self.assertEqual(mock_format_location.call_count, 2)
        mock_calculate_distance.assert_called_once_with((40.7128, -74.0060), (34.0522, -118.2437))


if __name__ == '__main__':
    unittest.main()