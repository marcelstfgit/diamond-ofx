@echo off
setlocal DisableDelayedExpansion
set "diamondRoot=%~dp0..\"
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

set "diamondTestDir=%diamondRoot%build\tests"
if not exist "%diamondTestDir%" mkdir "%diamondTestDir%"
if not exist "%diamondTestDir%" exit /b 1

pushd "%diamondTestDir%" || exit /b 1
cl.exe /nologo /std:c++17 /permissive- /W4 /utf-8 /EHsc /O2 /MT ^
    /DNDEBUG /I"%diamondRoot%third_party\openfx\include" ^
    "%diamondRoot%tests\Harness.cpp" /Fe:"%diamondTestDir%\Harness.exe" /link /MACHINE:X64 /INCREMENTAL:NO
set "diamondBuildStatus=%ERRORLEVEL%"
popd
exit /b %diamondBuildStatus%
