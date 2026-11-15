# Build on Windows x64

1. Install Visual Studio 2022 or Visual Studio Build Tools 2022 with the **Desktop development with C++** workload, the MSVC x64 tools, and the Windows SDK.
2. Clone the repository or extract `Diamond-0.03-source.zip`. Open a terminal at the root of the directory and run the build script.

   ```bat
   .\build.cmd
   ```

The script locates MSVC using `vswhere.exe`, loads the x64 build environment, and compiles with C++17. The OpenFX 1.5.1 headers are included in `third_party/openfx/include`. The build links the C++ runtime statically and generates the bundle at the path below.

```text
build\Diamond.ofx.bundle\Contents\Win64\Diamond.ofx
```

To use a local build, follow the [installation guide](INSTALL.md) with the `Diamond.ofx.bundle` folder generated in `build`. When distributing the binary, include the [Diamond MIT license](../LICENSE.txt) and the [OpenFX BSD 3-Clause license](../third_party/openfx/LICENSE.md).

## Source layout

- `src/Plugin.cpp` registers the effects and handles OpenFX host actions.
- `src/EffectControls.cpp` defines effect parameters, defaults, and options.
- `src/ImageIO.cpp` converts host image buffers and preserves the alpha convention.
- `src/Effects.h` defines the renderer interface. The other files in `src/` implement the effects.

## Check a change

`tests/Harness.cpp` is a small OpenFX host. It renders every effect on synthetic frames without a video editor and prints a hash of the output and the render time for each case. It fails when a render returns an error or when the plug-in calls the host from a thread other than the render thread.

Build it, then record the results of the bundle you are starting from.

```bat
.\tests\build.cmd
.\build\tests\Harness.exe build\Diamond.ofx.bundle\Contents\Win64\Diamond.ofx --write before.tsv
```

Rebuild the bundle with your change and compare. A change that should leave the picture alone must report zero different renders.

```bat
.\build.cmd
.\build\tests\Harness.exe build\Diamond.ofx.bundle\Contents\Win64\Diamond.ofx --compare before.tsv
```

Use `--size 1920x1080 --repeat 3` to measure render times, `--only D_Glow` to run one effect, `--pictures <folder>` to save the 8-bit results as PPM files, and `--cancel-after <polls>` to make the host cancel each render partway through.
