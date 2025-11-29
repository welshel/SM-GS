

# 🛠️ SM-GS Full Installation Guide (WSL2)

这份文档详细记录了如何从零开始（Windows 10/11）搭建运行 SM-GS 所需的完整环境。本指南严格复现作者的开发环境：

  * **OS**: Ubuntu 24.04 LTS (via WSL2)
  * **Compiler**: GCC/G++ 11.5.0
  * **CUDA Toolkit**: 11.8
  * **PyTorch**: 2.1.2 + cu118

-----

## Part 1: 安装 WSL2 与 Ubuntu 24.04

1.  **开启 WSL 功能**
    以**管理员身份**打开 PowerShell，输入以下命令：

    ```powershell
    wsl --install
    ```

    *注意：如果你的 Windows 版本较新，这通常默认安装 Ubuntu。如果需要指定版本，请使用 `wsl --install -d Ubuntu-24.04`。*

2.  **重启电脑**
    安装完成后，重启计算机。系统会自动弹出终端窗口完成 Ubuntu 的初始化（设置用户名和密码）。

3.  **检查系统版本**
    进入 Ubuntu 终端，确认系统版本：

    ```bash
    cat /etc/os-release
    # 应输出 PRETTY_NAME="Ubuntu 24.04.x LTS"
    ```

-----

## Part 2: 配置基础编译环境 (GCC 11)

Ubuntu 24.04 默认携带的是 GCC 13/14，这与 CUDA 11.8 不兼容。我们需要手动安装 GCC 11 并设置为编译时的默认编译器。

1.  **更新源并安装 GCC-11**

    ```bash
    sudo apt update
    sudo apt install gcc-11 g++-11 build-essential -y
    ```

2.  **验证安装**

    ```bash
    gcc-11 --version
    # 应输出 gcc-11 (Ubuntu 11.5.0-...) 11.5.0
    ```

-----

## Part 3: 安装 CUDA Toolkit 11.8

**重要提示**：在 WSL2 中，**不要安装显卡驱动**！Windows 主机的显卡驱动会自动透传给 WSL2。你只需要安装 CUDA **Toolkit**。

1.  **下载并安装 CUDA 11.8 (Runfile 方式推荐)**

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```

2.  **安装选项配置**

      * 在安装界面，首先输入 `accept`。
      * **关键步骤**：按空格键**取消勾选 Driver**（因为 WSL 使用 Windows 驱动）。
      * 确保勾选 `CUDA Toolkit 11.8`。
      * 选择 `Install`。

3.  **配置环境变量**
    打开 `.bashrc` 文件：

    ```bash
    nano ~/.bashrc
    ```

    在文件末尾添加：

    ```bash
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    ```

    保存退出（Ctrl+O, Enter, Ctrl+X），然后刷新配置：

    ```bash
    source ~/.bashrc
    ```

4.  **验证 NVCC**

    ```bash
    nvcc --version
    # 应显示 release 11.8, V11.8.89
    ```

-----

## Part 4: 配置 Python 环境 (Conda & PyTorch)

1.  **安装 Miniconda**

    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
    ```

2.  **克隆代码库**

    ```bash
    git clone https://github.com/welshel/SM-GS.git --recursive
    cd SM-GS
    ```

3.  **创建 Conda 环境**
    使用项目提供的配置文件创建环境（包含 Python 3.10 和 PyTorch 2.1.2）：

    ```bash
    conda env create -f environment.yml
    conda activate sm-gs
    ```

4.  **验证 PyTorch CUDA 版本**

    ```bash
    python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
    # 应输出 2.1.2 和 11.8
    ```

-----

## Part 5: 编译并安装子模块 (最关键的一步)

这是最容易出错的步骤。我们需要强制指定使用 GCC 11 来编译 `diff-gaussian-rasterization` 和 `simple-knn`。

1.  **设置编译器变量**
    在终端中执行：

    ```bash
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    ```

2.  **安装子模块**

    ```bash
    pip install ./submodules/diff-gaussian-rasterization
    pip install ./submodules/simple-knn
    ```

3.  **验证安装**
    如果没有报错，说明编译成功。你可以尝试运行代码进行测试。

-----

## 常见问题 (FAQ)

**Q: 为什么编译时报错 `unsupported GNU version`?**
A: 这是因为 CUDA 11.8 不支持 GCC 12+。请务必执行 Part 5 中的 `export CC=/usr/bin/gcc-11`。

**Q: `nvcc` 找不到命令？**
A: 请检查是否正确将 `/usr/local/cuda-11.8/bin` 加入了 PATH。

-----