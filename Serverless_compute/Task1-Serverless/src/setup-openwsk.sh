#!/bin/bash
# ------- BEGIN CONFIGURATION ------- #
SRC_PATH=~/Src/openwsk
OPENWSK_VERSION=1.0.0
OPENWSK_CLI_VERSION=1.2.0
ARCH=amd64
# -------- END CONFIGURATION -------- #

CURR_PATH=$(pwd)
set -e


if [ ! -d "$SRC_PATH" ]; then
echo "making dir $SRC_PATH" && mkdir -p "$SRC_PATH"
fi
cd $SRC_PATH

# Download openwhisk source code
if [ ! -d "openwhisk" ]; then
set +e
timeout 2m git clone --recursive https://github.com/apache/openwhisk.git
if [[ $? -ne $(expr 0) ]]; then
rm -rf openwhisk
set -e
# In case of bad internet connection, use mirror
git clone --recursive https://hub.fastgit.org/apache/openwhisk.git
fi
fi

cd openwhisk
# Do not checkout v1.0.0, use code from master branch
# git checkout $OPENWSK_VERSION
# Build OpenWhisk with gradlew
./gradlew core:standalone:build

if [ ! -f "OpenWhisk_CLI-$OPENWSK_CLI_VERSION-linux-$ARCH.tgz" ]; then
set +e
wget https://github.com/apache/openwhisk-cli/releases/download/$OPENWSK_CLI_VERSION/OpenWhisk_CLI-$OPENWSK_CLI_VERSION-linux-$ARCH.tgz
if [[ $? -ne $(expr 0) ]]; then
# In case of bad internet connection, use mirror
set -e
wget https://hub.fastgit.org/apache/openwhisk-cli/releases/download/$OPENWSK_CLI_VERSION/OpenWhisk_CLI-$OPENWSK_CLI_VERSION-linux-$ARCH.tgz
fi
fi

tar -xzvf OpenWhisk_CLI-$OPENWSK_CLI_VERSION-linux-$ARCH.tgz 
sudo mv wsk /usr/local/bin/
echo "alias start-openwsk=\"java -jar $SRC_PATH/openwhisk/bin/openwhisk-standalone.jar\"
" >> $PROFILE
echo "Installation completed, 'wsk' command should be available. Run OpenWhisk with 'java -jar $SRC_PATH/openwhisk/bin/openwhisk-standalone.jar' or the alias 'start-openwsk'"
cd $CURR_PATH
