import os
import requests
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_video(video_name, output_dir="downloads"):
    """
    Download a video from the GKE deployment
    
    Args:
        video_name (str): Name of the video file to download
        output_dir (str): Directory to save the downloaded video
    """
    # Get the service IP from environment or use a default
    service_ip = os.getenv("GKE_SERVICE_IP", "YOUR_GKE_SERVICE_IP")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the URL
    url = f"http://{service_ip}:8080/download/{video_name}"
    
    # Download the file
    print(f"Downloading {video_name} from {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        output_path = os.path.join(output_dir, video_name)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully to {output_path}")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos from GKE deployment")
    parser.add_argument("video_name", help="Name of the video file to download")
    parser.add_argument("--output-dir", default="downloads", help="Directory to save the downloaded video")
    
    args = parser.parse_args()
    download_video(args.video_name, args.output_dir) 