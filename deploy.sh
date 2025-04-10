#!/bin/bash

# Set your GCP project ID
PROJECT_ID="YOUR_PROJECT_ID"

# Build the Docker image
docker build -t gcr.io/$PROJECT_ID/ai-video-generator:latest .

# Push the image to Google Container Registry
docker push gcr.io/$PROJECT_ID/ai-video-generator:latest

# Create Kubernetes secrets for API keys
kubectl create secret generic ai-video-secrets \
  --from-literal=gemini-api-key='YOUR_GEMINI_API_KEY' \
  --from-literal=huggingface-token='YOUR_HUGGINGFACE_TOKEN'

# Apply Kubernetes configurations
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/download-service.yaml

# Wait for the deployment to be ready
kubectl rollout status deployment/ai-video-generator

echo "Deployment completed. Getting the LoadBalancer IPs..."
echo "Main application IP:"
kubectl get service ai-video-generator-service
echo "Download service IP:"
kubectl get service video-download-service 