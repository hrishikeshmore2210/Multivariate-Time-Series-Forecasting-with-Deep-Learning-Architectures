import os
import zipfile
import urllib.request
import tarfile

def download_file(url, dest_path):
    """Download a file from a URL if it does not exist."""
    if not os.path.exists(dest_path):
        print(f"📥 Downloading {os.path.basename(dest_path)}...")
        urllib.request.urlretrieve(url, dest_path)
        print("✅ Download complete.")
    else:
        print(f"✅ File already exists: {dest_path}")

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"📦 Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")

def extract_tar_gz(tar_path, extract_to):
    """Extract a .tar.gz or .tgz file."""
    print(f"📦 Extracting {os.path.basename(tar_path)}...")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")

# Create base data folder
os.makedirs("data", exist_ok=True)

# --------------------------
# POWER DATASET
# --------------------------
power_dir = "data/power"
os.makedirs(power_dir, exist_ok=True)

power_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
power_zip = os.path.join(power_dir, "LD2011_2014.txt.zip")

download_file(power_url, power_zip)
extract_zip(power_zip, power_dir)

# --------------------------
# TRAFFIC DATASET
# --------------------------
traffic_dir = "data/traffic"
os.makedirs(traffic_dir, exist_ok=True)

traffic_url = "https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip?download=1"
traffic_zip = os.path.join(traffic_dir, "traffic_hourly_dataset.zip")

download_file(traffic_url, traffic_zip)
extract_zip(traffic_zip, traffic_dir)

# --------------------------
# WEATHER DATASET
# --------------------------
weather_dir = "data/weather"
os.makedirs(weather_dir, exist_ok=True)

weather_url = "https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv.zip"
weather_zip = os.path.join(weather_dir, "jena.zip")

download_file(weather_url, weather_zip)
extract_zip(weather_zip, weather_dir)

print("\n🎯 All datasets downloaded and extracted successfully!")
