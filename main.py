import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from geolocator.image_processor import ImageProcessor
from geolocator.model import GeoPredictor
from geolocator.geocoder import Geocoder

app = typer.Typer()
console = Console()


@app.command()
def main(image_path: Path = typer.Argument(..., help="Path to the image file", exists=True),
        use_exif: bool = typer.Option(False, "--use-exif", "-e", help="Use EXIF data to verify prediction")):
    """Predict the geolocation of an image."""
    try:
        rprint(f"[bold green]Processing image:[/bold green] {image_path}")
        rprint(f"[bold]EXIF verification:[/bold] {'Enabled' if use_exif else 'Disabled'}")
        
        # Initialize components
        image_processor = ImageProcessor()
        geo_predictor = GeoPredictor()
        geocoder = Geocoder()
        
        # Process image
        with console.status("[bold green]Loading image...", spinner="dots"):
            image = image_processor.load_image(image_path)
            image_array = image_processor.preprocess_image()
        
        # Get prediction
        with console.status("[bold green]Predicting location...", spinner="dots"):
            predicted_coords, metadata = geo_predictor.predict(image_array)
            predicted_lat, predicted_lon = predicted_coords
            
            # Get human-readable location
            location_data = geocoder.reverse_geocode(predicted_lat, predicted_lon)
            location_str = geocoder.format_location(location_data)
        
        # Display prediction results
        rprint(f"[bold blue]Predicted location:[/bold blue] {location_str}")
        rprint(f"[bold blue]Coordinates:[/bold blue] {predicted_lat:.6f}, {predicted_lon:.6f}")
        rprint(f"[bold blue]Confidence:[/bold blue] {metadata['confidence']:.2f}")
        
        # Check against EXIF data if requested
        if use_exif:
            with console.status("[bold green]Extracting EXIF data...", spinner="dots"):
                image_processor.extract_exif(image_path)
                exif_coords = image_processor.get_gps_from_exif()
            
            if exif_coords:
                exif_lat, exif_lon = exif_coords
                
                # Get human-readable EXIF location
                exif_location_data = geocoder.reverse_geocode(exif_lat, exif_lon)
                exif_location_str = geocoder.format_location(exif_location_data)
                
                # Calculate distance
                distance_km = geocoder.calculate_distance(predicted_coords, exif_coords)
                
                # Create comparison table
                table = Table(title="Prediction vs. EXIF Data")
                table.add_column("", style="cyan")
                table.add_column("Predicted", style="green")
                table.add_column("EXIF (Actual)", style="yellow")
                
                table.add_row("Location", location_str, exif_location_str)
                table.add_row("Coordinates", f"{predicted_lat:.6f}, {predicted_lon:.6f}", 
                              f"{exif_lat:.6f}, {exif_lon:.6f}")
                
                console.print(table)
                
                # Display distance
                if distance_km < 1:
                    distance_m = distance_km * 1000
                    rprint(f"[bold]Distance:[/bold] {distance_m:.0f} meters")
                else:
                    rprint(f"[bold]Distance:[/bold] {distance_km:.1f} kilometers")
                
                # Evaluate accuracy
                if distance_km < 10:
                    rprint("[bold green]Prediction is very close![/bold green]")
                elif distance_km < 100:
                    rprint("[bold yellow]Prediction is in the same region.[/bold yellow]")
                else:
                    rprint("[bold red]Prediction is far from actual location.[/bold red]")
            else:
                rprint("[yellow]No GPS data found in EXIF metadata.[/yellow]")
                
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
