#!/bin/bash
set -e

echo "Building Serverless App"
docker build -t python3action-mnist:dev ./code
docker image prune -f