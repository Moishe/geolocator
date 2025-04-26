"""Tests for the geocoder module."""
import unittest
from unittest.mock import patch, MagicMock

from geolocator.geocoder import Geocoder


class TestGeocoder(unittest.TestCase):
    """Test cases for the Geocoder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.geocoder = Geocoder(user_agent="test-agent")
    
    @patch('requests.get')
    def test_reverse_geocode(self, mock_get):
        """Test reverse geocoding coordinates."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "display_name": "New York, USA",
            "address": {
                "city": "New York",
                "state": "New York",
                "country": "USA"
            }
        }
        mock_get.return_value = mock_response
        
        # Test reverse geocoding
        result = self.geocoder.reverse_geocode(40.7128, -74.0060)
        
        # Verify results
        self.assertEqual(result["display_name"], "New York, USA")
        self.assertEqual(result["address"]["city"], "New York")
        
        # Verify the request was made correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["lat"], 40.7128)
        self.assertEqual(kwargs["params"]["lon"], -74.0060)
        self.assertEqual(kwargs["headers"]["User-Agent"], "test-agent")
    
    @patch('requests.get')
    def test_reverse_geocode_error(self, mock_get):
        """Test reverse geocoding with an error response."""
        # Setup mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_get.return_value = mock_response
        
        # Test reverse geocoding with error
        with self.assertRaises(ValueError):
            self.geocoder.reverse_geocode(40.7128, -74.0060)
    
    def test_format_location(self):
        """Test formatting location data."""
        # Test with complete data
        geocode_data = {
            "address": {
                "city": "New York",
                "state": "New York",
                "country": "USA"
            }
        }
        result = self.geocoder.format_location(geocode_data)
        self.assertEqual(result, "New York, New York, USA")
        
        # Test with partial data
        geocode_data = {
            "address": {
                "country": "USA"
            }
        }
        result = self.geocoder.format_location(geocode_data)
        self.assertEqual(result, "USA")
        
        # Test with town instead of city
        geocode_data = {
            "address": {
                "town": "Small Town",
                "state": "Some State",
                "country": "USA"
            }
        }
        result = self.geocoder.format_location(geocode_data)
        self.assertEqual(result, "Small Town, Some State, USA")
        
        # Test with empty data
        result = self.geocoder.format_location({})
        self.assertEqual(result, "Unknown location")
    
    def test_calculate_distance(self):
        """Test calculating distance between coordinates."""
        # New York to Los Angeles (roughly 3,944 km)
        ny = (40.7128, -74.0060)
        la = (34.0522, -118.2437)
        distance = self.geocoder.calculate_distance(ny, la)
        
        # Check approximate distance (allowing for some variance due to the formula)
        self.assertGreater(distance, 3800)
        self.assertLess(distance, 4100)
        
        # Test zero distance
        self.assertAlmostEqual(
            self.geocoder.calculate_distance(ny, ny),
            0.0,
            places=10
        )


if __name__ == '__main__':
    unittest.main()