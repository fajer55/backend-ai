#!/usr/bin/env python
"""
Script to check and remove corrupted images
"""
import os
from PIL import Image

def check_and_clean_images(directory):
    """Check all images and remove corrupted ones"""
    removed = 0
    checked = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            # Skip non-image files
            if file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                continue

            checked += 1
            try:
                # Try to open and load the image
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    img.load()

                    # Check dimensions
                    if img.size[0] < 10 or img.size[1] < 10:
                        print(f"❌ Too small: {file_path}")
                        os.remove(file_path)
                        removed += 1

            except Exception as e:
                print(f"❌ Corrupted: {file_path} - {e}")
                try:
                    os.remove(file_path)
                    removed += 1
                except Exception as remove_error:
                    print(f"   Error removing: {remove_error}")

    print(f"\n✅ Checked {checked} images")
    print(f"🗑️  Removed {removed} corrupted images")
    return removed

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "uploads"
    print(f"Checking images in: {directory}")
    check_and_clean_images(directory)
