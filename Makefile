user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)
phism=/workspace
vhls=/jc9016/tools/Xilinx/2020.2
th=1

# Build Phism
build-docker: test-docker
	docker run -it -v $(shell pwd):/workspace -v $(vhls):$(vhls) phism8:latest /bin/bash \
	-c "make build-phism"
	echo "Phism has been installed successfully!"

# Clone submodule and build docker container
test-docker:
	(cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) . --tag phism8)

# Enter docker container
shell:
	docker run -it -v $(shell pwd):/workspace -v $(vhls):$(vhls) phism8:latest /bin/bash

# Evaluate polybench (baseline) - need to be used in environment
test-polybench:
	python3 scripts/pb-flow.py -c -j $(th) example/polybench

# Evaluate polybench (polymer) - need to be used in environment
test-polybench-polymer:
	python3 scripts/pb-flow.py -c -p -j $(th) example/polybench

# Build LLVM and Phism
build-phism:
	source scripts/setup-vitis-hls-llvm.sh
	./scripts/build-llvm.sh
	./scripts/build-phism.sh

# Sync and update submodules
sync:
	git submodule sync
	git submodule update --init --recursive

clean: clean-phism
	rm -rf $(phism)/llvm/build

clean-phism:
	rm -rf $(phism)/build
