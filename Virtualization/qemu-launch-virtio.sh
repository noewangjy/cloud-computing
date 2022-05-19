export SHARED_PATH=../public
qemu-system-x86_64 -serial stdio \
	-smp 3,sockets=1,cores=3,threads=1 -m 4096 \
	-device virtio-gpu-pci \
	-display default,show-cursor=on \
	-device qemu-xhci -device usb-kbd \
	-device usb-tablet -device intel-hda \
	-device hda-duplex \
	-enable-kvm \
	-drive file=$SHARED_PATH/centos_disk_0.img,if=virtio,cache=writethrough \
	-nic user,model=virtio,hostfwd=tcp::10122-:22

