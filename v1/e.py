import rasterio

def check_bands(image_path):
    try:
        with rasterio.open(image_path) as src:
            band_count = src.count  # Number of bands in the image
            print(f"✅ Image Loaded: {image_path}")
            print(f"📊 Number of Bands Found: {band_count}")
            return band_count, image_path  # Returning both band count and image path
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return None, image_path

# Example Usage
image_path = "temp_image.tif"  # Replace with the actual image path
bands, img_path = check_bands(image_path)

if bands is not None:
    print(f"📌 The image '{img_path}' has {bands} spectral bands.")
else:
    print(f"⚠️ Could not read the image '{img_path}'.")
