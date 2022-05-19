sudo ovs-vsctl del-port vhost-user-0
sudo ovs-vsctl del-br br0

sudo ovs-vsctl add-br br0 -- set bridge br0 datapath_type=netdev
sudo ovs-vsctl add-port br0 vhost-user-0 -- set Interface vhost-user-0 type=dpdkvhostuserclient options:vhost-server-path="/tmp/sock0"
sudo ovs-vsctl add-port br0 vhost-user-1 -- set Interface vhost-user-1 type=dpdkvhostuserclient options:vhost-server-path="/tmp/sock1"
sudo ovs-vsctl show