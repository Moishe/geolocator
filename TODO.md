# Geolocator TODO

## Steps to build a photo location guesser

1. Set up basic project structure
   - Command-line argument parsing
   - Image loading functionality

2. Extract visual features from images
   - Implement image preprocessing
   - Research and select appropriate feature extraction methods

3. Implement geolocation prediction
   - Research available models for image-based geolocation
   - Explore options for local models that run on M1 Mac
   - Integrate chosen model to predict location from image features

4. Extract and parse EXIF data
   - Implement EXIF data extraction
   - Parse geolocation coordinates from EXIF

5. Compare prediction with EXIF data
   - Calculate distance between predicted and actual locations
   - Provide accuracy metrics

6. Enhance output and visualization
   - Format location results in a user-friendly way
   - Consider adding map visualization

7. Testing and optimization
   - Test with various images
   - Optimize performance for M1 Mac

## Questions about APIs/Models

- Do you have a preference for any specific geolocation model?
- Should we consider using Google's Reverse Geocoding API to convert coordinates to readable locations?
- Would you prefer a lightweight model approach or higher accuracy at the cost of performance?
- Do you have API keys for any map/location services you'd like to use?