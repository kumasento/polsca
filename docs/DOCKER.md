---
Author: Jianyi Cheng
---

# Build with Docker

To build a Docker container with Phism installed:
```sh
make build-docker
```

To use Phism in the Docker container:
```
make shell vhls=${YOUR_VITIS_DIR}
```
PS: To check your `${YOUR_VITIS_DIR}`, you should see the following when run:
```
$ ls ${YOUR_VITIS_DIR}
DocNav  Model_Composer  Vitis  Vitis_HLS  Vivado  xic
```

For instance, to run Polybench:
```
make test-polybench
```
