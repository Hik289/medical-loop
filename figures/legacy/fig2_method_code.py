"""Convert an existing, locally supplied method PNG to a PDF."""
import argparse
from pathlib import Path
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_png", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input_png) as source:
        source.convert("RGB").save(args.output_pdf, "PDF", resolution=300.0)
