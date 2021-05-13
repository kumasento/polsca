user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)
phism=/workspace
vhls=~/tools/Xilinx/2020.2

# Build Phism
build-docker: test-docker
	docker run -it -v $(shell pwd):/workspace -v $(vhls):/tools phism20:latest /bin/bash \
	-c "make build_"
	echo "Phism has been installed successfully!"

# Clone submodule and build docker container
test-docker:
	git submodule update --init --recursive
	(cd Docker; docker build --no-cache --build-arg UID=$(user) --build-arg GID=$(group) . --tag phism20)

# Enter docker container
shell:
	docker run -it -v $(shell pwd):/workspace -v $(vhls):/tools phism20:latest /bin/bash

# Evaluate polybench (baseline) - need to be used in environment
test-polybench:
	./scripts/pb-flow ./example/polybench 2>&1 | tee phism-test.log

# Evaluate polybench (polymer) - need to be used in environment
test-polybench-polymer:
	./scripts/pb-flow ./example/polybench 1 2>&1 | tee phism-test.log

# Build LLVM and Phism
build_:
	set -e # Abort if one of the commands fail
	mkdir -p $(phism)/llvm/build
	(cd $(phism)/llvm/build; \
	  cmake ../llvm \
	  -DCMAKE_C_COMPILER=gcc \
  	  -DCMAKE_CXX_COMPILER=g++ \
  	  -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang;clang-extra-tools" \
  	  -DCMAKE_BUILD_TYPE=DEBUG \
  	  -DLLVM_BUILD_EXAMPLES=OFF \
  	  -DLLVM_TARGETS_TO_BUILD="host" \
  	  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  	  -DLLVM_ENABLE_OCAMLDOC=OFF \
  	  -DLLVM_ENABLE_BINDINGS=OFF \
  	  -DLLVM_INSTALL_UTILS=ON \
	  -DBUILD_POLYMER=ON \
	  -DPLUTO_LIBCLANG_PREFIX=$(shell llvm-config-9 --prefix) \
  	  -DLLVM_ENABLE_ASSERTIONS=ON; \
	  cmake --build . --target all -- -j 32)
	mkdir -p $(phism)/build
	(cd $(phism)/build; \
	  cmake .. \
	  -DCMAKE_BUILD_TYPE=Debug \
	  -DLLVM_ENABLE_ASSERTIONS=ON \
	  -DMLIR_DIR=$(phism)/llvm/build/lib/cmake/mlir/ \
	  -DLLVM_DIR=$(phism)/llvm/build/lib/cmake/llvm/ \
	  -DCMAKE_C_COMPILER=clang-9 \
	  -DCMAKE_CXX_COMPILER=clang++-9 \
	  -DLLVM_EXTERNAL_LIT=$(phism)/llvm/build/bin/llvm-lit \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON; \
	  cmake --build . --target check-phism -- -j 32)

clean: clean_phism
	rm -rf $(phism)/llvm/build

clean_phism:
	rm -rf $(phism)/build
