#!/bin/bash
# ------- BEGIN CONFIGURATION ------- #
PKG_MANAGER=apt-get
# -------- END CONFIGURATION -------- #

sudo $PKG_MANAGER clean
sudo $PKG_MANAGER autoclean

sudo $PKG_MANAGER update
if [[ $? -ne $(expr 0) ]]; then
echo "apt-get encountered problem, exit."
exit
fi

# Setup docker
if [[ ! $(docker --version | grep version) ]]; then
echo "Installing docker"
curl -fsSL https://get.docker.com -o get-docker.sh
set +e
sudo sh get-docker.sh
rm get-docker.sh
set -e
fi

# Add current user to docker group
# if [[ ! $(docker ps | grep CONTAINER) ]]; then 
# echo "Adding current user to docker group"
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl restart docker

echo "Installation completed"

fi