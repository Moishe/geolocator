from typing import Dict, Any, Tuple, Optional
import requests


class Geocoder:
    """Convert between coordinates and human-readable locations."""
    
    def __init__(self, user_agent: str = "geolocator/0.1.0"):
        """Initialize geocoder with user agent for Nominatim API."""
        self.user_agent = user_agent
        self.base_url = "https://nominatim.openstreetmap.org"
    
    def reverse_geocode(self, lat: float, lon: float) -> Dict[str, Any]:
        """Convert coordinates to human-readable location using Nominatim."""
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": 1,
            "zoom": 18
        }
        
        headers = {
            "User-Agent": self.user_agent
        }
        
        response = requests.get(
            f"{self.base_url}/reverse", 
            params=params,
            headers=headers
        )
        
        if response.status_code != 200:
            raise ValueError(f"Geocoding API error: {response.status_code} - {response.text}")
            
        return response.json()
    
    def format_location(self, geocode_data: Dict[str, Any]) -> str:
        """Format geocoded data into a readable location string."""
        if not geocode_data:
            return "Unknown location"
            
        address = geocode_data.get("address", {})
        
        # Build location string from most relevant parts
        parts = []
        
        # Add city/town/village
        for key in ["city", "town", "village", "hamlet"]:
            if key in address:
                parts.append(address[key])
                break
                
        # Add state/province/region
        for key in ["state", "province", "region"]:
            if key in address:
                parts.append(address[key])
                break
                
        # Add country
        if "country" in address:
            parts.append(address["country"])
            
        return ", ".join(parts) if parts else "Unknown location"
    
    def calculate_distance(self, coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in kilometers (Haversine formula)."""
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1 = coords1
        lat2, lon2 = coords2
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r