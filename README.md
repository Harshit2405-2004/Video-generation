# AI Video Generator

A powerful tool that converts text into engaging video content using AI. This application uses Gemini 2.0 Flash API for text generation, Edge TTS for voice-over, and CogVideoX for video generation.

## Features

- Text-to-video generation
- AI-powered voice-over creation
- Multiple video segments based on text descriptions
- Cloud deployment on Google Kubernetes Engine (GKE)
- Video download functionality

## Deployment on GKE

### Prerequisites

- Google Cloud Platform account
- Google Cloud SDK installed
- Docker installed
- kubectl configured

### Deployment Steps

1. Clone this repository
2. Update the following files with your information:
   - `deploy.sh`: Replace `YOUR_PROJECT_ID`, `YOUR_GEMINI_API_KEY`, and `YOUR_HUGGINGFACE_TOKEN`
   - `k8s/deployment.yaml`: Replace `YOUR_PROJECT_ID` with your GCP project ID

3. Make the deployment script executable:
   ```bash
   chmod +x deploy.sh
   ```

4. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

5. After deployment, note the LoadBalancer IPs for both services:
   - Main application IP (port 80)
   - Download service IP (port 8080)

## Using the Application

1. Access the application through the main LoadBalancer IP in your web browser
2. Enter your text and select the number of video segments to generate
3. Wait for the video generation process to complete
4. The final video will be displayed in the interface

## Downloading Generated Videos

To download a generated video:

1. Note the filename of the video (e.g., `final_output_123e4567-e89b-12d3-a456-426614174000.mp4`)
2. Use the download script:
   ```bash
   python download_video.py final_output_123e4567-e89b-12d3-a456-426614174000.mp4
   ```

3. Alternatively, you can download directly from the browser:
   ```
   http://<DOWNLOAD_SERVICE_IP>:8080/download/final_output_123e4567-e89b-12d3-a456-426614174000.mp4
   ```

## Architecture

The application consists of:

- A Gradio web interface for user interaction
- Gemini 2.0 Flash API for text generation
- Edge TTS for voice-over creation
- CogVideoX for video generation
- FastAPI endpoints for video downloads
- Kubernetes deployment with LoadBalancer services
- Persistent volume for video storage

## Resource Requirements

- CPU: 2-4 cores per pod
- Memory: 4-8GB per pod
- Storage: 10GB for video storage
- GPU: Not required (using CPU for inference)

## Troubleshooting

- If videos are not generating, check the pod logs:
  ```bash
  kubectl logs -l app=ai-video-generator
  ```

- If downloads are failing, verify the persistent volume is mounted correctly:
  ```bash
  kubectl describe pod -l app=ai-video-generator
  ```

- To restart the deployment:
  ```bash
  kubectl rollout restart deployment/ai-video-generator
  ``` 