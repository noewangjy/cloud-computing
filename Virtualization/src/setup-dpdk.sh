#!/bin/bash
# ------- BEGIN CONFIGURATION ------- #
SRC_PATH=~/Src/dpdk
DPDK_VERSION=20.11.1
PROFILE=~/.bashrc
# -------- END CONFIGURATION -------- #

CURR_PATH=$(pwd)


set -e

if [ ! -d "$SRC_PATH" ]; then
echo "making dir $SRC_PATH" && mkdir -p "$SRC_PATH"
fi

cd "$SRC_PATH"

# Download dpdk source code and extract
if [ ! -f "dpdk-$DPDK_VERSION.tar.xz" ]; then
wget https://fast.dpdk.org/rel/dpdk-$DPDK_VERSION.tar.xz
fi

if [ ! -d "dpdk-stable-$DPDK_VERSION" ]; then
tar xf dpdk-$DPDK_VERSION.tar.xz
fi

export DPDK_DIR=$(pwd)/dpdk-stable-$DPDK_VERSION
cd $DPDK_DIR
export DPDK_BUILD=$DPDK_DIR/build

# Build dpdk from source
export CC=clang
meson build
ninja -C build
sudo ninja -C build install
sudo ldconfig

# Verify installation of dpdk
pkg-config --modversion libdpdk

if [ $PROFILE ]; then
echo "export DPDK_DIR=$DPDK_DIR" >> $PROFILE
echo "export PATH=\$PATH:\$DPDK_DIR/usertools/" >> $PROFILE
echo "Installation completed, you should run 'source $PROFILE' for changes to take effect"
else
echo "Installation completed, you should manually add 'export DPDK_DIR=$DPDK_DIR' 'export PATH=\$PATH:\$DPDK_DIR/usertools/' to your profile"
fi

cd $CURR_PATH
