export SHARED_PATH=../public
sudo -E $(which qemu-system-x86_64) \
   -smp 2,sockets=1,cores=2,threads=1 -m 2048 \
   -device virtio-gpu-pci \
   -display default,show-cursor=on \
   -device qemu-xhci -device usb-kbd \
   -device usb-tablet -device intel-hda \
   -device hda-duplex \
   -drive file=$SHARED_PATH/centos_disk_0.img,if=virtio,cache=writethrough \
   -nic user,model=virtio,hostfwd=tcp::10122-:22 \
   -enable-kvm \
   -object memory-backend-file,id=mem,size=2048M,mem-path=/dev/hugepages,share=on \
   -mem-prealloc -numa node,memdev=mem \
   -chardev socket,id=char0,path=/tmp/sock0,server=on \
   -netdev tap,type=vhost-user,id=mynet-0,chardev=char0,vhostforce=on,queues=4 \
   -device virtio-net-pci,netdev=mynet-0,mq=on,vectors=10,id=net0,mac=00:00:00:00:00:01 \
   -monitor stdio