#!/bin/bash
set -e

echo "Building DistributedWorker"
docker build -t python3action-dist-train-mnist:dev ../code
docker image prune -f
