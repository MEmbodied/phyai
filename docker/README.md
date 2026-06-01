# Easy Setup with Docker and DevContainer for phyai

To simplify developers' experience with phyai, we provide ready-to-use Dockerfile and DevContainer configurations.

## 1. Using Dockerfile

```bash
git clone https://github.com/MEmbodied/phyai.git
cd phyai/docker

# NVIDIA GPU. Chose your CUDA version: Dockerfile.cuxxx
docker build -t phyai_torch211cu13 -f Dockerfile.torch211cu13 .
docker run -it --gpus all --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name phyai_torch211cu13_dev phyai_torch211cu13 bash
```

## 2. Using DevContainer

To set up with VS Code Dev Containers:

1. Install prerequisites:
    - Docker
    - VS Code
    - Dev Containers extension

2. Clone repository with submodules:

    ```shell
    git clone --recursive https://github.com/MEmbodied/phyai.git
    ```

3. Open project in VS Code:

    ```shell
    code phyai
    ```

4. When prompted:

    "Folder contains a Dev Container configuration file. Reopen in container?"
    Click Reopen in Container

    (Alternatively: Press F1 → "Dev Containers: Reopen in Container")

The container will automatically build and launch with:

* All dependencies pre-installed
* Correct environment configuration
* Shared memory and security settings applied
