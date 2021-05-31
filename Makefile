user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)
phism=/workspace
vhls=/tools/Xilinx/2020.2

# Build Phism
build-docker: test-docker
	docker run -it -v $(shell pwd):/workspace -v $(vhls):$(vhls) phism7:latest /bin/bash \
	-c "make build-phism"
	echo "Phism has been installed successfully!"

# Clone submodule and build docker container
test-docker:
	git submodule update --init --recursive
#	(cd Docker; docker build --no-cache --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) . --tag phism7)
	(cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) . --tag phism7)

# Enter docker container
shell:
	docker run -it -v $(shell pwd):/workspace -v $(vhls):$(vhls) phism7:latest /bin/bash

# Evaluate polybench (baseline) - need to be used in environment
test-polybench:
	./scripts/pb-flow ./example/polybench -c COSIM 2>&1 | tee phism-test.log

# Evaluate polybench (polymer) - need to be used in environment
test-polybench-polymer:
	./scripts/pb-flow ./example/polybench -p USE POLYMER -c COSIM  2>&1 | tee phism-test.log

# Build LLVM and Phism
build-phism:
	source scripts/setup-vitis-hls-llvm.sh
	./scripts/build-llvm.sh
	./scripts/build-phism.sh

clean: clean_phism
	rm -rf $(phism)/llvm/build

clean_phism:
	rm -rf $(phism)/build
