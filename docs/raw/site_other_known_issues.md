# Known Issues & Troubleshooting - BEHAVIOR

> 원본: https://behavior.stanford.edu/other/known_issues.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 대회 페이지 본문이 링크하거나 대회 준비에 직접 쓰이는 사이트 문서
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# **Known Issues & Troubleshooting**

## 🤔 **Known Issues**

> **How can I parallelize running multiple scenes in OmniGibson?**
>
> Currently, to run multiple scenes in parallel, you will need to launch separate instances of the OmniGibson environment. While this introduces some overhead due to running multiple instances of IsaacSim, we are actively working on implementing parallelization capabilities. Our goal is to enable running multiple scenes within a single instance, streamlining the process and reducing the associated overhead.

> **Why does my particle system behave unexpectedly after clearing the scene?**
>
> If you clear a particle system and immediately re-initialize it without performing a physics step in between, you may observe unexpected behavior. Our assumption is that this occurs because some internal state variables aren't properly reset during the clearing process.
>
>
>
> **Solution:** Take a physics step after clearing the particle system but before re-initializing it:
>
>
>
>
> ```
> env.scene.clear_system(system_name)
> og.sim.step()
> system = env.scene.get_system(system_name) # Now safe to re-initialize
> ```
>
>
>
>
> **Best Practice:** In most scenarios, completely clearing a particle system is unnecessary. Instead, consider: - Removing particles with `system.remove_all_particles()` or `system.remove_particles(...)` - Generating new particles with `system.generate_particles(...)`

## 🧯 **Troubleshooting**

> **I cannot open Omniverse Launcher AppImage on Linux**
>
> You probably need to [install FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) to run the Omniverse Launcher AppImage.

> **OmniGibson is stuck at HydraEngine rtx failed creating scene renderer.**
>
> `OmniGibson` is likely using an unsupported GPU (default is id 0). Run `nvidia-smi` to see the active list of GPUs, and select an NVIDIA-supported GPU and set its corresponding ID when running `OmniGibson` with `export OMNIGIBSON_GPU_ID=<ID NUMBER>`.
