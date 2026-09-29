@echo off
rem Build the pi0.5 native engine on Windows: MSVC 2022 + CUDA 12.8 nvcc (conda env "pi05build", no admin install).
rem   build_windows.bat            -> build_win\_pi05native.pyd (Python 3.11 extension for the evaluator process)
rem                                   build_win\pi05_verify.exe, build_win\tok_test.exe
rem Static CUDA runtime (no cudart DLL clash with torch in the evaluator). No CMake, no third-party libraries.
setlocal
set HERE=%~dp0
set B=%HERE%build_win
set CONDA_ENVS=C:\Users\user one\anaconda3\envs
set CUDA=%CONDA_ENVS%\pi05build\Library
set PY=%CONDA_ENVS%\behavior
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist "%B%\obj" mkdir "%B%\obj"
if not exist "%B%\objtok" mkdir "%B%\objtok"
set NVCC="%CUDA%\bin\nvcc.exe" -std=c++20 -O3 -gencode arch=compute_120,code=sm_120 -Xcompiler "/O2 /MD /EHsc /utf-8" -cudart static %PI05_DEFS%
cd /d "%HERE%src"
for %%f in (kernels model) do (
  %NVCC% -c %%f.cu -o "%B%\obj\%%f.obj" || exit /b 1
)
for %%f in (tokenizer weights host_io image engine_api) do (
  %NVCC% -x cu -c %%f.cpp -o "%B%\obj\%%f.obj" || exit /b 1
)
set OBJS="%B%\obj\kernels.obj" "%B%\obj\model.obj" "%B%\obj\tokenizer.obj" "%B%\obj\weights.obj" "%B%\obj\host_io.obj" "%B%\obj\image.obj" "%B%\obj\engine_api.obj"
lib /nologo /out:"%B%\pi05.lib" %OBJS% || exit /b 1
cd /d "%HERE%glue"
cl /nologo /O2 /MD /utf-8 /DPI05_STATIC /c pi05native_module.c /I"%PY%\include" /Fo"%B%\obj\pi05native_module.obj" || exit /b 1
%NVCC% -shared -o "%B%\_pi05native.pyd" "%B%\obj\pi05native_module.obj" "%B%\pi05.lib" -L"%PY%\libs" -lpython311 || exit /b 1
cd /d "%HERE%tools"
%NVCC% -o "%B%\pi05_verify.exe" verify.cpp "%B%\pi05.lib" || exit /b 1
cl /nologo /O2 /std:c++20 /EHsc /utf-8 tok_test.cpp ..\src\tokenizer.cpp ..\src\weights.cpp /Fe"%B%\tok_test.exe" /Fo"%B%\objtok\\" || exit /b 1
echo built into %B%
