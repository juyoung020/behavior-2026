# Installation - BEHAVIOR

> 원본: https://behavior.stanford.edu/getting_started/installation.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 대회 페이지 본문이 링크하거나 대회 준비에 직접 쓰이는 사이트 문서
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# **Installation**

## **Requirements**

Please make sure your system meets the following specs:

- **OS:** Ubuntu 22.04+ / Windows 10+
- **RAM:** 32GB+
- **GPU:** NVIDIA RTX 2070+
- **VRAM:** 8GB+

> **Why these specs?**
>
> The core simulator behind BEHAVIOR, **`OmniGibson`**, is built upon NVIDIA's [Omniverse](https://www.nvidia.com/en-us/omniverse/) and [Isaac Sim](https://developer.nvidia.com/isaac-sim) platforms, so we inherit their dependencies. For more information, please see [Isaac Sim's Requirements](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html).

## **Setup**

Choose your installation method:

- **🔧 Development Setup (Linux / Windows, Recommended)**: Use our unified setup script for editable installs and development.

> **Installation Options**
>
> - **PyPI packages** (`omnigibson`, `bddl`, `joylo`) are coming soon for easier production deployments
> - **Docker installation** is temporarily unavailable while we update it for the monorepo structure
>
>
>
> For now, please use the development setup which installs from source.

> **What does the setup script do?**
>
> **[탭: 🔧 Development Setup (Linux / Windows)]**
>
>
>
> **Use this method if you want to modify the code or need the latest development features.**
>
>
>
> 1. Clone the BEHAVIOR-1K repository:
>
>   Clone the latest stable release (recommended):
>
>   ```
>   git clone -b v3.9.3 https://github.com/StanfordVL/BEHAVIOR-1K.git
>   cd BEHAVIOR-1K
>   ```
>
>   If you want the latest development features and updates (potentially less stable), directly clone the main branch instead:
>
>   ```
>   git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
>   cd BEHAVIOR-1K
>   ```
> 2. Run the unified setup script:
>
>   **[탭: Linux]**
>
>   ```
>   # Full installation with all components
>   ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
>   # Core components only (without evaluation and teleoperation)
>   ./setup.sh --new-env --omnigibson --bddl --dataset
>   # Headless/automated installation (auto-accepts Conda TOS, NVIDIA Isaac Sim EULA, and BEHAVIOR Dataset License)
>   ./setup.sh --new-env --omnigibson --bddl --dataset \
>              --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
>   # See all options
>   ./setup.sh --help
>   ```
>
>   **[탭: Windows]**
>
>   ```
>   # Full installation with all components
>   .\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset -Eval -Primitives
>   # Core components only (without evaluation and teleoperation)
>   .\setup.ps1 -NewEnv -OmniGibson -BDDL -Dataset
>   # Headless/automated installation (auto-accepts Conda TOS, NVIDIA Isaac Sim EULA, and BEHAVIOR Dataset License)
>   .\setup.ps1 -NewEnv -OmniGibson -BDDL -Dataset `
>               -AcceptCondaTos -AcceptNvidiaEula -AcceptDatasetTos
>   # See all options
>   .\setup.ps1 -Help
>   ```
> 3. Activate the environment:
>
>   ```
>   conda activate behavior
>   ```
>
>
>
> > **What does the setup script do?**
> >
> > - Creates a new conda environment named `behavior` (when using `--new-env`)
> > - Installs all selected components in editable mode
> > - Downloads Isaac Sim and BEHAVIOR datasets (if requested)
>
>
>
> > **Using an existing Python environment**
> >
> > You can omit `--new-env` to use your current Python environment instead of creating a new conda environment:
> >
> >
> >
> >
> > ```
> > # Linux: Install in current environment
> > ./setup.sh --omnigibson --bddl --joylo --dataset
> >
> > # Skip confirmation prompt with --confirm-no-conda
> > ./setup.sh --omnigibson --bddl --confirm-no-conda
> > ```
> >
> >
> >  The script will prompt for confirmation if not in a conda environment.
>
>
>
> > **Windows Note**
> >
> > Run PowerShell as Administrator and set execution policy if needed:
> >
> >
> >
> >
> > ```
> > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> > ```
>
>
>
> **[탭: Linux]**
>
>
>
>
> ```
> # Full installation with all components
> ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
>
> # Core components only (without evaluation and teleoperation)
> ./setup.sh --new-env --omnigibson --bddl --dataset
>
> # Headless/automated installation (auto-accepts Conda TOS, NVIDIA Isaac Sim EULA, and BEHAVIOR Dataset License)
> ./setup.sh --new-env --omnigibson --bddl --dataset \
>            --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
>
> # See all options
> ./setup.sh --help
> ```
>
>
>
>
> **[탭: Windows]**
>
>
>
>
> ```
> # Full installation with all components
> .\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset -Eval -Primitives
>
> # Core components only (without evaluation and teleoperation)
> .\setup.ps1 -NewEnv -OmniGibson -BDDL -Dataset
>
> # Headless/automated installation (auto-accepts Conda TOS, NVIDIA Isaac Sim EULA, and BEHAVIOR Dataset License)
> .\setup.ps1 -NewEnv -OmniGibson -BDDL -Dataset `
>             -AcceptCondaTos -AcceptNvidiaEula -AcceptDatasetTos
>
> # See all options
> .\setup.ps1 -Help
> ```

## **Explore `BEHAVIOR`!**

> **Expect slowdown during first execution**
>
> Omniverse requires some one-time startup setup when **`OmniGibson`** is imported for the first time.
>
>
>
> The process could take up to 5 minutes. This is expected behavior, and should only occur once!

**`BEHAVIOR`** is now successfully installed! You can try teleoperating one of our robots:

```
python -m omnigibson.examples.robots.robot_control_example --quickstart # (1)!
```

1. This demo lets you choose a scene, robot, and set of controllers, and then teleoperate the robot using your keyboard. The `--quickstart` flag will automatically select the scene and robot for you - remove that if you want to change the scene or robot.

You can also try exploring some of our new scenes interactively:

```
python -m omnigibson.examples.scenes.scene_selector # (1)!
```

1. This demo lets you choose a scene and interactively move around using your keyboard and mouse. Hold down **`Shift`** and then **`Left-click + Drag`** an object to apply forces!

---

**Next:** Get quickly familiarized with **`BEHAVIOR`** from our [Quickstart Guide](https://behavior.stanford.edu/getting_started/quickstart.html)!

## **Troubleshooting**

> **CuRobo installation failed**
>
> This is likely due to incorrect configurations of cuda toolkit. If you don't plan to use system-installed cuda toolkit, here is an alternate way to get it work all in conda:
>
>
>
> 1. Install behavior without primitives support first: `./setup.sh --new-env --bddl --omnigibson [OTHER FLAGS]`
> 2. Activate conda environment: `conda activate behavior`
> 3. Install cuda (we use 12.4 here as an example): `conda install "cuda-compiler=12.4" "cuda-toolkit=12.4" -c conda-forge`
> 4. Set environment variables:
>   ```
>   export CUDA_HOME="$CONDA_PREFIX"
>   export CUDACXX="$CONDA_PREFIX/bin/nvcc"
>   export CUDA_PATH="$CUDA_HOME"
>   export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
>   export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH"
>   export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
>   ```
> 5. Install primitive support: `./setup.sh --bddl --omnigibson --primitives`
>
>
>
> Note that with this method you need to set the environment variables (step 4) the first time curobo performs JIT compilation.

> **OmniGibson is stuck at HydraEngine rtx failed creating scene renderer.**
>
> `OmniGibson` is likely using an unsupported GPU (default is id 0). Run `nvidia-smi` to see the active list of GPUs, and select an NVIDIA-supported GPU and set its corresponding ID when running `OmniGibson` with `export OMNIGIBSON_GPU_ID=<ID NUMBER>`.
