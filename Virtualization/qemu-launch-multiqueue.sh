# Must match -object,size= with -m and less than /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
export SHARED_PATH=../public
sudo -E $(which qemu-system-x86_64) -serial stdio \
   -smp 4,sockets=1,cores=4,threads=1 -m 4096 \
   -device virtio-gpu-pci \
   -display default,show-cursor=on \
   -device qemu-xhci -device usb-kbd \
   -device usb-tablet -device intel-hda \
   -device hda-duplex \
   -drive file=$SHARED_PATH/centos_disk_0.img,if=virtio,cache=writethrough \
   -nic user,model=virtio,hostfwd=tcp::10122-:22 \
   -enable-kvm \
   -object memory-backend-file,id=mem0,size=4096M,mem-path=/dev/hugepages,share=on \
   -mem-prealloc -numa node,memdev=mem0 \
   -chardev socket,id=char0,path=/tmp/sock0,server=on \
   -netdev tap,type=vhost-user,id=vhost-user-0,chardev=char0,vhostforce=on,queues=8 \
   -device virtio-net-pci,netdev=vhost-user-0,mq=on,vectors=18,id=net0,mac=00:00:00:00:00:01
