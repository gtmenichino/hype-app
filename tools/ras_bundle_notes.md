# bin/ras2025 — HEC-RAS 2025 Linux runtime bundle

The Surface step runs HEC-RAS 2025's .NET CLI (`ras`) headlessly. On Windows dev boxes the
installed app is used via `HYPE_RAS_BIN`; on Posit Connect Cloud (Linux) the app invokes the
self-contained bundle at `bin/ras2025` (`dotnet app/ras.dll <verb> …`, wired in
`hype_app/ras.py`). Everything under `bin/ras2025/**` is Git-LFS tracked.

## Contents & provenance

| Folder     | What                                                                | Source / version |
|------------|---------------------------------------------------------------------|------------------|
| `app/`     | `ras.dll` + the 42 managed IL assemblies from its deps.json closure + a rewritten `ras.runtimeconfig.json` (framework-form, rollForward LatestMinor). No `ras.deps.json` on purpose — directory probing resolves the flat folder. | HEC-RAS 2025 Alpha install (0.1.0.2965-dev) |
| `dotnet/`  | .NET Runtime **9.0.17** linux-x64 (runtime only)                    | builds.dotnet.microsoft.com (SHA512 verified) |
| `natives/` | `libhdf5.so`, `libhdf5_hl.so` (**HDF5 2.0.0**, SONAME `.so.320`, zlib/aec statically linked), `libe_sqlite3.so` | HDF Group GitHub release `hdf5-2.0.0-ubuntu-2404_gcc` (SHA256 verified); SQLitePCLRaw.lib.e_sqlite3 2.1.11 |
| `GDAL/`    | `lib/` = libgdal.so.38 + the four C# SWIG wraps (`libgdal_wrap/ogr_wrap/osr_wrap/gdalconst_wrap.so`) + deps (RUNPATH=$ORIGIN); `common/data/` = GDAL data + `proj.db` merged (one dir serves GDAL_DATA and PROJ_LIB) | MaxRev.Gdal.LinuxRuntime.Minimal.x64 + MaxRev.Gdal.Core **3.12.3.499** — GDAL **3.12.3**, the exact version the Windows install ships |

See `bin/ras2025/MANIFEST.md` for URLs, hashes, and the full file inventory.

## Two hard-won invariants (do not regress)

1. **HDF5 must load as exactly one instance.** The natives ship only under bare names
   (`libhdf5.so`); `hype_app/ras.py::prepare_linux_bundle()` creates the SONAME names
   (`libhdf5.so.320`, …) as **symlinks** in `$TMPDIR/hype_ras_hdf5links`, which is prepended
   to `LD_LIBRARY_PATH`. Never ship both names as file copies: glibc dedups by inode, so two
   copies load as two independent libhdf5 instances — the .NET binding creates datasets in
   one while `libhdf5_hl`'s internal `H5DOwrite_chunk → H5Dwrite_chunk` lands in the other,
   every chunked result write fails with `invalid dataset ID`, and the result file comes out
   silently NaN-filled. (GDAL/lib's own libhdf5 copies were removed for the same reason —
   libgdal resolves the SONAME through the symlink dir to the shared instance.)

2. **The repack forces `Equation Set = "Shallow Water Equations"` + `Time-stepping =
   Explicit`** (`hype_app/ras_h5.py::write_plan_attrs`). The template's Diffusion Wave
   solver requires a sparse-matrix backend: Intel MKL Pardiso on Windows, but on Linux the
   engine's managed-PCG fallback diverges on the first step ("Could not honor Courant
   condition"). `SolverExpSWE` is matrix-free and was verified to produce identical results
   on Windows and Linux (max depth 1.815 m on the Mink Brook smoke case on both).

## RAS CLI quirks (verified 0.1.0.2965-dev)

- `createterrain -o` must be an **absolute** path (relative → "is not rooted" crash), the
  file must not already exist, and the parent dir must exist.
- `map -o` needs a directory component (`./out.tif` fine, bare `out.tif` crashes the writer).
- The CLI writes results to `Results/<Plan> (Result).h5` (the GUI writes `<Plan>.h5`);
  `ras.py` globs `Results/*.h5` newest-first.
- Run duration comes from the **Boundary Condition file's Start/End Time** window
  (`Time Window Mode = BoundaryCondition`), not the plan's Compute Duration. All times are
  .NET ticks (100 ns since 0001-01-01).
- `--solver CPU --core-count -1` overrides the plan at run time.
- h5py note: never rewrite string members of compound datasets — HDF5's NULLTERM conversion
  truncates values that exactly fill their width (`'Normal Depth'` S12 → `'Normal Dept'`).
  Write numeric fields one at a time (`dset['Constant'] = …`).

## Verifying the bundle

Windows end-to-end (repack → mesh → solve → map → extent):

    set HYPE_RAS_BIN=<HEC-RAS 2025 Alpha folder>
    .venv\Scripts\python tools\ras_smoke.py            # add --geographic-dem for the UTM-fallback path

Linux container check (mirrors Connect Cloud; ~python:3.12-slim, no apt packages needed):

    docker run --rm -v <repo>/bin/ras2025:/bundle -v <a repacked project>:/work \
      -e DOTNET_ROOT=/bundle/dotnet -e DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
      -e RAS_GDAL=/bundle/GDAL \
      -e LD_LIBRARY_PATH=/tmp/hype_ras_hdf5links:/bundle/natives:/bundle/GDAL/lib \
      python:3.12-slim bash -c '
        mkdir -p /tmp/hype_ras_hdf5links
        ln -sf /bundle/natives/libhdf5.so /tmp/hype_ras_hdf5links/libhdf5.so.320
        ln -sf /bundle/natives/libhdf5_hl.so /tmp/hype_ras_hdf5links/libhdf5_hl.so.320
        chmod +x /bundle/dotnet/dotnet
        /bundle/dotnet/dotnet /bundle/app/ras.dll healthcheck'

`ras healthcheck` passing = .NET runtime + GDAL natives + PROJ data all load. A solve with
`grep -c HDF5-DIAG` equal to **0** = the HDF5 single-instance invariant holds.

## Connect Cloud knobs (env vars, read by hype_app/ras.py)

| Var                   | Default | Meaning |
|-----------------------|---------|---------|
| `HYPE_RAS_BIN`        | unset   | Dev override: folder of (or path to) ras.exe; bypasses the bundle |
| `HYPE_RAS_GREEN_CELLS`| 20000   | UI estimate turns amber above this |
| `HYPE_RAS_MAX_CELLS`  | 120000  | Hard cap — run refuses to solve above this (post-mesh check) |
| `HYPE_RAS_TIMEOUT_S`  | 1800    | Wall-clock kill for any single CLI step |

Timing reference: 887 cells / 6 h window / 10 s steps ≈ 21 s solve in a Linux container;
scale expectations roughly linearly with cell count × timestep count before tuning the caps.
