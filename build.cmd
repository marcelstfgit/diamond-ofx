@echo off
setlocal DisableDelayedExpansion
set "diamondRoot=%~dp0"
set "diamondVswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%diamondVswhere%" (
    echo Install Visual Studio 2022 or Build Tools with the Desktop development with C++ workload.
    exit /b 1
)

set "diamondVsPath="
for /f "usebackq tokens=*" %%I in (`"%diamondVswhere%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "diamondVsPath=%%I"
if not defined diamondVsPath (
    echo Visual Studio C++ x64 build tools were not found.
    exit /b 1
)
call "%diamondVsPath%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

set "diamondBuildDir=%diamondRoot%build"
set "diamondBinDir=%diamondBuildDir%\Diamond.ofx.bundle\Contents\Win64"
if not exist "%diamondBinDir%" mkdir "%diamondBinDir%"
if not exist "%diamondBinDir%" exit /b 1

pushd "%diamondBuildDir%" || exit /b 1
cl.exe /nologo /std:c++17 /permissive- /W4 /utf-8 /EHsc /O2 /MT /LD ^
    /DNDEBUG /DNOMINMAX /DWIN32_LEAN_AND_MEAN ^
    /I"%diamondRoot%third_party\openfx\include" ^
    "%diamondRoot%src\Plugin.cpp" "%diamondRoot%src\Parallel.cpp" "%diamondRoot%src\Shake.cpp" ^
    "%diamondRoot%src\EffectControls.cpp" "%diamondRoot%src\ImageIO.cpp" ^
    "%diamondRoot%src\Lighting.cpp" "%diamondRoot%src\Stylize.cpp" ^
    "%diamondRoot%src\BlurMoCurves.cpp" "%diamondRoot%src\Atmosphere.cpp" ^
    "%diamondRoot%src\FilmAndSort.cpp" "%diamondRoot%src\RackDefocus.cpp" ^
    "%diamondRoot%src\NewOptics.cpp" "%diamondRoot%src\NewWarp.cpp" "%diamondRoot%src\NewStylize.cpp" ^
    "%diamondRoot%src\Video05.cpp" "%diamondRoot%src\Transitions05.cpp" "%diamondRoot%src\Gpu05.cpp" ^
    /Fe:"%diamondBinDir%\Diamond.ofx" /link /MACHINE:X64 /INCREMENTAL:NO /IMPLIB:"Diamond.lib"
set "diamondBuildStatus=%ERRORLEVEL%"
popd
exit /b %diamondBuildStatus%
