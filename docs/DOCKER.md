---
Author: Jianyi Cheng
---

# Build with Docker

To build a Docker container with Phism installed:
```sh
make build-docker vhls=${YOUR_VITIS_DIR}
```
PS: To check your `${YOUR_VITIS_DIR}`, you should see the following when run:
```
$ ls ${YOUR_VITIS_DIR}
DocNav  Model_Composer  Vitis  Vitis_HLS  Vivado  xic
```

To use Phism in the Docker container:
```
make shell vhls=${YOUR_VITIS_DIR}
```

For instance, to run Polybench:
```
# In the docker container:
make test-polybench
```
