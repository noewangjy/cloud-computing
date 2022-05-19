#!/bin/bash
# ------- BEGIN CONFIGURATION ------- #
SRC_PATH=~/Src/ovs
OVS_VERSION=2.16.0
PROFILE=~/.bashrc
# -------- END CONFIGURATION -------- #

CURR_PATH=$(pwd)
N_PROC=$(expr $(cat /proc/cpuinfo |grep "processor"|wc -l) \* 2)

set -e

if [ ! -d "$SRC_PATH" ]; then
echo "making dir $SRC_PATH" && mkdir -p "$SRC_PATH"
fi

cd $SRC_PATH

# Clone ovs source code
if [ ! -d "ovs" ]; then
set +e
timeout 2m git clone https://github.com/openvswitch/ovs.git
if [[ $? -ne $(expr 0) ]]; then 
rm -rf ovs
set -e
# In case of bad internet connection, use mirror
git clone https://hub.fastgit.org/openvswitch/ovs.git
fi
fi

# Build ovs with dpdk support
cd ovs
git checkout v$OVS_VERSION
./boot.sh
export CC=clang
./configure --with-dpdk=yes
make -j$N_PROC

set +e
sudo make install
set -e

# Add openvswitch scripts to path
if [ $PROFILE ]; then
echo "export PATH=\$PATH:/usr/local/share/openvswitch/scripts" >> $PROFILE
echo "Installation completed, you should run 'source $PROFILE' to use ovs commands"
else
echo "Installation completed."
fi


cd $CURR_PATH
