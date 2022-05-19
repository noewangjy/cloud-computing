#!/bin/bash

set -e

bash src/setup-deps.sh
bash src/setup-qemu.sh
bash src/setup-dpdk.sh
bash src/setup-ovs.sh
