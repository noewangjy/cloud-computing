#!/bin/bash
# TODO: Test this block
# if [[ $(lsb_release -i --short) -eq "Ubuntu" ||  $(lsb_release -i --short) -eq "Debian" ]]; then
# PKG_MANAGER=apt-get
# echo "Using apt-get"
# elif [[ $(lsb_release -i --short) -eq "Fedora" ||  $(lsb_release -i --short) -eq "CentOS" ]]; then
# PKG_MANAGER=yum
# echo "yum"
# fi
PKG_MANAGER=apt-get


sudo $PKG_MANAGER clean
sudo $PKG_MANAGER update
sudo $PKG_MANAGER install build-essential \
                     ninja-build \
					 meson \
					 libmount-dev \
					 libpixman-1-dev \
					 libusb-1.0 \
					 libglib2.0-dev \
					 openjdk-11-jdk \
					 nodejs \
					 npm \
					 git \
					 curl \
					 autoconf \
					 libtool \
					 libgtk-3-dev \
					 clang \
					 libsysfs-dev -y
sudo $PKG_MANAGER clean
sudo $PKG_MANAGER autoclean
