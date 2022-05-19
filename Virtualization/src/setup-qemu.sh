#!/bin/bash
# ------- BEGIN CONFIGURATION ------- #
SRC_PATH=~/Src/qemu
QEMU_VERSION=6.1.0
PROFILE=~/.bashrc
# -------- END CONFIGURATION -------- #

CURR_PATH=$(pwd)
N_PROC=$(expr $(cat /proc/cpuinfo |grep "processor"|wc -l) \* 2)
set -e

if [ ! -d "$SRC_PATH" ]; then
echo "making dir $SRC_PATH" && mkdir -p "$SRC_PATH"
fi

cd "$SRC_PATH"
echo $(pwd)
# Download qemu source code
if [ ! -f "qemu-$QEMU_VERSION.tar.xz" ]; then
wget https://download.qemu.org/qemu-$QEMU_VERSION.tar.xz
fi

if [ ! -f "qemu-$QEMU_VERSION" ]; then
tar xJf qemu-$QEMU_VERSION.tar.xz
fi

cd qemu-$QEMU_VERSION
./configure --enable-vhost-user --enable-vhost-net --enable-kvm  --enable-libusb --enable-gtk
make -j$N_PROC

if [ $PROFILE ]; then
echo "export PATH=$(pwd)/build:\$PATH" >> $PROFILE
echo "Installation completed, you should run 'source $PROFILE' to use qemu commands"
else
echo "Installation completed, you should manually add 'export PATH=$(pwd)/build:\$PATH' to your profile"
fi

cd $CURR_PATH
