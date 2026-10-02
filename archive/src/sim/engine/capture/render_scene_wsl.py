"""렌더 장면(scene.rsc) 을 WSL 에서 뜨기 — 렌더 없이 기하·재질·조명·기준 prim 만 (포팅 평가기 관측 영상용, 15절).

render 작업자 A 의 render_capture.py(Windows RTX 기준 자료용)를 고치지 않고 그대로 부르되,
- 카메라 관측은 0 영상(physx_capture.install_no_render: WSL 에는 RTX 렌더 장치가 없다),
- 스텝 0 만, 다시 그리기 0 번,
- 끝나면 결과를 저장한 뒤 Kit 종료를 건너뛴다(segfault 덤프 방지).
그 뒤 convert_scene.py 로 scene.rsc + frame_0000.rfr 을 만든다(render_scene_wsl.sh). 결과는 에셋 파생물 -> git 밖.

    python render_scene_wsl.py --dump-dir <폴더> -- <eval_instrumented 인자>
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RC_DIR = os.path.normpath(os.path.join(HERE, "..", "tests", "render", "capture"))
sys.path.insert(0, RC_DIR)
sys.path.insert(0, HERE)


def main():
    argv = sys.argv[1:]
    ours, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    dump_dir = ours[ours.index("--dump-dir") + 1]
    import render_capture as RC  # A 의 도구 (무수정)
    import physx_capture as PC

    cap = RC.RenderCapture(dump_dir, [0], 0)
    os.environ.setdefault("LOCALAPPDATA", os.path.expanduser("~/.cache"))
    RC.install(cap)
    PC.install_no_render()

    import omnigibson as og

    def shutdown(*a, **kw):  # A 의 마무리(저장)만 하고 Kit 종료는 건너뛴다
        try:
            cap.finish()
        except Exception as e:
            print(f"[render_scene_wsl] 마무리 실패: {e!r}", flush=True)
        try:  # og.shutdown 이 하던 임시 폴더(풀린 장면 USD) 정리를 대신 한다
            import shutil

            if getattr(og, "tempdir", None) and os.path.isdir(og.tempdir):
                shutil.rmtree(og.tempdir, ignore_errors=True)
        except Exception:
            pass
        print("[render_scene_wsl] 저장 끝 — Kit 종료 건너뜀", flush=True)
        sys.stdout.flush()
        os._exit(0)

    og.shutdown = shutdown
    sys.argv = [RC.INSTRUMENTED] + rest
    runpy.run_path(RC.INSTRUMENTED, run_name="__main__")


if __name__ == "__main__":
    main()
