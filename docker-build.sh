#!/bin/bash

# Build the Docker image and load it into Docker
docker build --load -t iseebetter:latest .

echo "Docker image 'iseebetter:latest' built successfully and loaded into Docker" 