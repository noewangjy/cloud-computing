#!/bin/bash
# start-ovs.sh

set -e 

# ------- BEGIN CONFIGURATION ------- #
ETH_INTERFACE=ens33 # Name of ethernet interface
MEM_HUGEPAGE=4096 # Hugepage size
OVSDEV_PCIID=0000:06:00.0
DPDK_DIR=/home/noewang/Src/dpdk/dpdk-stable-20.11.1 # DPDK installation
OVS_RUN_DIR=/usr/local/var/run/openvswitch
OVS_SCRIPT_PATH=/usr/local/share/openvswitch/scripts # OVS script path
DB_SOCK=$OVS_RUN_DIR/db.sock # Place to create db sock
OVSDB_PID=$OVS_RUN_DIR/ovs-vswitchd.pid # Place to store OBSDB pid
# -------- END CONFIGURATION -------- #


if [ ! -d "$OVS_RUN_DIR" ]; then
echo "making dir $OVS_RUN_DIR" && sudo mkdir -p "$OVS_RUN_DIR"
fi

# Kill all ovs process
set +e
sudo killall -9 ovsdb-server
sudo killall -9 ovs-vswitchd
set -e
# Init service
# Dont need on Ubuntu 18.04
# sudo service openvswitch-switch start 

# Configure hugepage
echo ">>> Configuring hugepage"
sudo sysctl -w vm.nr_hugepages=$MEM_HUGEPAGE
echo $MEM_HUGEPAGE | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
grep HugePages_ /proc/meminfo
sudo mount -t hugetlbfs none /dev/hugepages

# Set vifo permission
echo ">>> Configuring vfio-pci"
dmesg | grep -e DMAR -e IOMMU
modprobe vfio-pci
sudo /bin/chmod a+x /dev/vfio
sudo /bin/chmod 0666 /dev/vfio/*

# Configure DPDK
echo ">>> Configuring DPDK"
sudo $DPDK_DIR/usertools/dpdk-devbind.py --status
sudo $DPDK_DIR/usertools/dpdk-devbind.py --bind=vfio-pci $ETH_INTERFACE
sudo $DPDK_DIR/usertools/dpdk-devbind.py --status

# Create ovsdb
if [ ! -f "/usr/local/etc/openvswitch/conf.db" ];then
echo ">>> Creating db"
sudo ovsdb-tool create /usr/local/etc/openvswitch/conf.db /usr/local/share/openvswitch/vswitch.ovsschema
fi

# Start ovsdb
set +e
echo ">>> Starting ovsdb"
sudo ovsdb-server --remote=punix:/usr/local/var/run/openvswitch/db.sock \
             --remote=db:Open_vSwitch,Open_vSwitch,manager_options \
             --private-key=db:Open_vSwitch,SSL,private_key \
             --certificate=db:Open_vSwitch,SSL,certificate \
             --bootstrap-ca-cert=db:Open_vSwitch,SSL,ca_cert \
             --pidfile --detach
set -e

# Configure ovs
echo ">>> Configuring ovs"
sudo ovs-vsctl --no-wait set Open_vSwitch . other_config:dpdk-init=true
#0x06 = 0b110 will use core 2 and core 1
sudo ovs-vsctl set Open_vSwitch . other_config:pmd-cpu-mask=0x6
sudo ovs-vsctl set Open_vSwitch . other_config:dpdk-lcore-mask=0x1
# sudo ovs-vsctl set Open_vSwitch . other_config:dpdk-socket-mem=512

# Start ovs
set +e
echo ">>> Starting ovs"
sudo $OVS_SCRIPT_PATH/ovs-ctl --no-ovsdb-server --db-sock="$DB_SOCK" start
set -e

# Valiating
sudo ovs-vsctl get Open_vSwitch . dpdk_initialized
sudo ovs-vswitchd --version
