# IRIS API Deployment Guide
## Docker and Google Kubernetes Engine (GKE) Setup

This guide provides step-by-step instructions for containerizing and deploying the IRIS API to Google Cloud Platform using Docker and Kubernetes.

---

## Prerequisites
- Docker installed on your local machine
- Google Cloud SDK (`gcloud`) installed and configured
- Active GCP project with billing enabled
- `kubectl` command-line tool installed

---

## Phase 1: Local Docker Setup

### Step 1.1: Configure Docker Permissions (Linux/Unix only)
```bash
# Add the current user to the "docker" group to run Docker commands without sudo
# Note: You'll need to log out and back in for this to take effect
sudo usermod -aG docker $USER
```

### Step 1.2: Build and Test Docker Image Locally
```bash
# Build the Docker image from the Dockerfile in the current directory
# -t flag tags the image with the name "iris-api"
docker build -t iris-api .

# Test run the container locally
# Maps port 8200 on your host to port 8200 in the container
docker run -p 8200:8200 iris-api
```
**Verification:** Access `http://localhost:8200` to ensure the API is running correctly.

---

## Phase 2: Google Cloud Setup

### Step 2.1: Create Artifact Registry Repository
```bash
# Create a Docker repository in Google Artifact Registry
# This will store your Docker images in the cloud
gcloud artifacts repositories create iris-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repo for IRIS API"
```
**Note:** Replace `us-central1` with your preferred region if needed.

### Step 2.2: Configure Docker Authentication with Artifact Registry
```bash
# Configure Docker to authenticate with Google Artifact Registry
# This allows you to push images to your GCP repository
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Step 2.3: Tag Docker Image for Artifact Registry
```bash
# Tag your local image with the Artifact Registry path
# Format: [REGION]-docker.pkg.dev/[PROJECT_ID]/[REPO_NAME]/[IMAGE_NAME]:[TAG]
docker tag iris-api us-central1-docker.pkg.dev/buoyant-country-473106-a3/iris-repo/iris-api:latest
```
**Important:** Replace `buoyant-country-473106-a3` with your actual GCP project ID.

### Step 2.4: Push Image to Artifact Registry
```bash
# Push the tagged image to Google Artifact Registry
# This uploads your Docker image to the cloud repository
docker push us-central1-docker.pkg.dev/buoyant-country-473106-a3/iris-repo/iris-api:latest
```

---

## Phase 3: Kubernetes Cluster Setup

### Step 3.1: Create GKE Autopilot Cluster
```bash
# Create a Google Kubernetes Engine cluster with Autopilot mode
# Autopilot mode automatically manages the infrastructure for you
gcloud container clusters create-auto iris-cluster \
    --location=us-central1 \
    --project=buoyant-country-473106-a3
```
**Note:** This operation may take 5-10 minutes to complete.

### Step 3.2: Install GKE Authentication Plugin
```bash
# Install the authentication plugin required for kubectl to work with GKE
gcloud components install gke-gcloud-auth-plugin

# Alternative installation method (if using apt-get):
# sudo apt-get install google-cloud-cli-gke-gcloud-auth-plugin
```

### Step 3.3: Get Cluster Credentials
```bash
# Configure kubectl to connect to your GKE cluster
# This downloads credentials and configures kubectl context
gcloud container clusters get-credentials iris-cluster \
    --zone us-central1 \
    --project buoyant-country-473106-a3
```

---

## Phase 4: Deploy to Kubernetes

### Step 4.1: Apply Kubernetes Configurations
```bash
# Deploy the application using Kubernetes manifest files
# deployment.yaml defines how your app runs (pods, replicas, etc.)
kubectl apply -f k8s/deployment.yaml

# Create a service to expose your deployment
# service.yaml defines how to access your pods (load balancing, ports, etc.)
kubectl apply -f k8s/service.yaml
```

### Step 4.2: Verify Deployment
```bash
# Check if pods are running successfully
kubectl get pods

# Check if the service is created and has an external IP (for LoadBalancer type)
kubectl get services
```

---

## Phase 5: Maintenance Operations

### Update Deployment with New Image
```bash
# When you have a new version of your image, update the deployment
# First, build, tag, and push the new image with a different tag (e.g., v2, latest-new)
docker build -t iris-api .
docker tag iris-api us-central1-docker.pkg.dev/buoyant-country-473106-a3/iris-repo/iris-api:v2
docker push us-central1-docker.pkg.dev/buoyant-country-473106-a3/iris-repo/iris-api:v2

# Update the Kubernetes deployment to use the new image
kubectl set image deployment/iris-api \
    iris-api=us-central1-docker.pkg.dev/buoyant-country-473106-a3/iris-repo/iris-api:v2

# Monitor the rollout status
kubectl rollout status deployment/iris-api
```

### Common Kubectl Commands
```bash
# View pod logs for debugging
kubectl logs [POD_NAME]

# Get detailed information about a pod
kubectl describe pod [POD_NAME]

# Scale the deployment
kubectl scale deployment iris-api --replicas=3

# View deployment history
kubectl rollout history deployment/iris-api

# Rollback to previous version if needed
kubectl rollout undo deployment/iris-api
```

### Clean Up Resources (When Needed)
```bash
# Delete the Kubernetes deployment
kubectl delete deployment iris-api

# Delete the Kubernetes service
kubectl delete service iris-api-service

# Delete the GKE cluster (WARNING: This will delete all resources in the cluster)
gcloud container clusters delete iris-cluster --zone us-central1

# Delete the Artifact Registry repository
gcloud artifacts repositories delete iris-repo --location=us-central1
```

---

## Important Notes

1. **Project ID**: Remember to replace `buoyant-country-473106-a3` with your actual GCP project ID throughout these commands.

2. **Region**: The examples use `us-central1`. Choose a region closest to your users for better performance.

3. **Costs**: Running a GKE cluster and storing images in Artifact Registry incurs costs. Monitor your GCP billing.

4. **Security**: For production deployments, consider:
   - Using private GKE clusters
   - Implementing proper IAM roles and service accounts
   - Scanning images for vulnerabilities
   - Using secrets management for sensitive data

5. **Kubernetes Files**: Ensure you have the following files in your `k8s/` directory:
   - `deployment.yaml`: Defines your application deployment
   - `service.yaml`: Defines how to expose your application

6. **Monitoring**: Set up monitoring and logging using Google Cloud Operations (formerly Stackdriver) for production environments.
