export SHARED_PATH=../public
sudo -E $(which qemu-system-x86_64) \
   -smp 2,sockets=1,cores=2,threads=1 -m 2048 \
   -device virtio-gpu-pci \
   -display default,show-cursor=on \
   -device qemu-xhci -device usb-kbd \
   -device usb-tablet -device intel-hda \
   -device hda-duplex \
   -drive file=$SHARED_PATH/centos_disk_0.img,if=virtio,cache=writethrough \
   -nic user,model=virtio,hostfwd=tcp::10123-:22 \
   -enable-kvm \
   -object memory-backend-file,id=mem,size=2048M,mem-path=/dev/hugepages,share=on \
   -mem-prealloc -numa node,memdev=mem \
   -chardev socket,id=char1,path=/tmp/sock1,server=on \
   -netdev tap,type=vhost-user,id=mynet-1,chardev=char1,vhostforce=on,queues=4 \
   -device virtio-net-pci,netdev=mynet-1,mq=on,vectors=10,id=net1,mac=00:00:00:00:00:02 \
   -monitor stdio \
   -incoming tcp:0:16666