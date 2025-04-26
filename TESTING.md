# Geolocator Testing Guide

The geolocator package includes a comprehensive test suite to ensure all functionality works as expected.

## Running Tests

To run the full test suite:

```bash
python -m pytest
```

To run a specific test file:

```bash
python -m pytest tests/test_model.py
```

To run with increased verbosity:

```bash
python -m pytest -v
```

## Test Coverage

The following components are tested:

1. **Image Processor** - Tests for image loading, preprocessing, and EXIF data extraction
2. **Geocoder** - Tests for reverse geocoding, location formatting, and distance calculation
3. **Model** - Tests for the geolocation prediction model
4. **Main Application** - Tests for CLI functionality

## Adding New Tests

When adding new features, please also add corresponding tests. Follow the existing patterns:

1. Use unittest's `TestCase` class
2. Mock external dependencies
3. Test both normal and error conditions
4. Use descriptive test method names

## Continuous Integration

In the future, we should set up CI/CD to automatically run these tests on each push and pull request.