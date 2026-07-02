# Linux Native Bundle Manifest — HEC-RAS 2025 .NET CLI on Linux (Posit Connect Cloud)

Staged: 2026-07-01 on Windows (win-x64 host). All binaries are **linux-x64 (glibc)**.
Root: `linux-bundle/` — subfolders `dotnet/`, `natives/`, `GDAL/{lib,common/data,bin}/`.
(A pre-existing `app/` folder with the CLI's managed assemblies was already present and was not modified.)

---

## 1. .NET Runtime 9.0.17 (linux-x64) → `dotnet/`

- Source: `https://builds.dotnet.microsoft.com/dotnet/Runtime/9.0.17/dotnet-runtime-9.0.17-linux-x64.tar.gz`
- Discovered as `latest-runtime` via `https://builds.dotnet.microsoft.com/dotnet/release-metadata/9.0/releases.json`
- Download size: 33,515,787 bytes. **SHA512 verified OK** against release metadata
  (`ad6fc3ee72b6...aa657634`).
- Plain runtime (not SDK, not ASP.NET).
- Layout confirmed: `dotnet/dotnet` (host, 70,232 bytes, ELF), `dotnet/shared/Microsoft.NETCore.App/9.0.17/` (186 files), `dotnet/host/fxr/9.0.17/libhostfxr.so`.
- Extracted: 190 files, **74.4 MB**.

## 2. HDF5 2.0.0 shared libraries (linux-x64) → `natives/`

- Source: HDF Group GitHub release **2.0.0**:
  `https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0-ubuntu-2404_gcc.tar.gz`
- Download size: 54,664,501 bytes. **SHA256 verified OK** against the release's `hdf5-2.0.0.sha256sums.txt`
  (`409d00035c4bd110945d5b9eb3f0e7439ce0ad165f9e408831bd7eca67fa6e14`).
- Build: Ubuntu 24.04 / gcc (glibc). Outer tarball wraps `HDF5-2.0.0-Linux.tar.gz` →
  `HDF_Group/HDF5/2.0.0/lib/`.
- **Exact soversion: `libhdf5.so.320.0.0` (SONAME `libhdf5.so.320`); `libhdf5_hl.so.320.0.0` (SONAME `libhdf5_hl.so.320`).**
- Staged as REAL FILES (upstream symlinks materialized as copies, since the bundle is staged on
  Windows and deployed by copy):
  - `natives/libhdf5.so` (20,551,512 B) — name probed by .NET P/Invoke `DllImport("hdf5")`
  - `natives/libhdf5.so.320` (20,551,512 B, identical copy) — name required by DT_NEEDED of hdf5_hl (and of MaxRev libgdal)
  - `natives/libhdf5_hl.so` (500,544 B) — P/Invoke `DllImport("hdf5_hl")`
  - `natives/libhdf5_hl.so.320` (500,544 B, identical copy)
- ELF DT_NEEDED verified: `libhdf5.so.320` needs only `libm.so.6`, `libc.so.6`
  (zlib + libaec are statically linked in this build — the tarball ships them only as `.a`).
  `libhdf5_hl.so.320` needs `libhdf5.so.320`, `libc.so.6`. No extra dep .so files needed.

## 3. GDAL 3.12.3 + C# SWIG wrappers (linux-x64) → `GDAL/`

- Packages (nuget.org, `.nupkg` = zip):
  - `MaxRev.Gdal.LinuxRuntime.Minimal.x64` **3.12.3.499** —
    `https://www.nuget.org/api/v2/package/MaxRev.Gdal.LinuxRuntime.Minimal.x64/3.12.3.499` (53,357,997 B)
  - `MaxRev.Gdal.Core` **3.12.3.499** —
    `https://www.nuget.org/api/v2/package/MaxRev.Gdal.Core/3.12.3.499` (1,349,487 B)
  - (`MaxRev.Gdal.LinuxRuntime.Minimal` 3.12.3.499 was downloaded first but is now a **meta-package**
    with no natives; the real linux-x64 payload lives in the `.x64` package above.)
- Upstream versions per nuspec: **GDAL 3.12.3, PROJ 9.8.0, GEOS 3.14.1** —
  GDAL is an **exact match** for the app's Windows GDAL 3.12.3. No version mismatch.
- `GDAL/lib/` (74 files, 145.8 MB): everything from `runtimes/linux-x64/native/`, including the four
  SWIG wrappers `libgdal_wrap.so`, `libogr_wrap.so`, `libosr_wrap.so`, `libgdalconst_wrap.so`,
  plus `libgdal.so.38` and all its bundled deps (proj/geos/curl/ssl/xerces/arrow/parquet/netcdf/hdf4/hdf5...).
  ELF check: every non-system DT_NEEDED of `libgdal.so.38` (41 libs) is present in this directory;
  wrap libs need `libgdal.so.38` + system `libstdc++.so.6`/`libgcc_s.so.1` and carry
  `RUNPATH=$ORIGIN`, so same-directory resolution works.
- `GDAL/common/data/` (162 files, 12.7 MB): the 161 files of `gdal-data/` from `MaxRev.Gdal.Core`
  (`runtimes/any/native/gdal-data/`) **merged with** `proj.db` (10,412,032 B) from the x64 runtime
  package (`runtimes/linux-x64/native/maxrev.gdal.core.libshared/proj.db`). Single dir for both
  `GDAL_DATA` and `PROJ_LIB`, as GDALSetup expects.
- `GDAL/bin/`: empty (no CLI tools shipped in these packages; not required).

## 4. libe_sqlite3.so (linux-x64) → `natives/`

- Package: `SQLitePCLRaw.lib.e_sqlite3` **2.1.11** (newest 2.x; 3.50.3 exists but is the new 3.x line) —
  `https://www.nuget.org/api/v2/package/SQLitePCLRaw.lib.e_sqlite3/2.1.11` (20,981,758 B)
- Staged: `natives/libe_sqlite3.so` (1,348,440 B) from `runtimes/linux-x64/native/`.
- ELF DT_NEEDED: only `libc.so.6`.

---

## Folder totals

| Folder | Files | Size |
|---|---|---|
| `dotnet/` | 190 | 74.4 MB |
| `natives/` | 5 | 41.4 MB |
| `GDAL/lib/` | 74 | 145.8 MB |
| `GDAL/common/data/` | 162 | 12.7 MB |
| `GDAL/bin/` | 0 | 0 |
| **Total staged (excl. pre-existing `app/`)** | **431** | **~274 MB** |

## Top 30 files by size (entire linux-bundle, incl. pre-existing app/)

| File | Bytes |
|---|---|
| GDAL/lib/libgdal.so.38 | 45,846,056 |
| GDAL/lib/libarrow.so.2300 | 22,633,784 |
| natives/libhdf5.so.320 | 20,551,512 |
| natives/libhdf5.so | 20,551,512 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Private.CoreLib.dll | 14,836,736 |
| GDAL/common/data/proj.db | 10,412,032 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Private.Xml.dll | 8,041,472 |
| GDAL/lib/libcrypto.so.3.2.0 | 7,925,192 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/libcoreclr.so | 7,399,744 |
| GDAL/lib/libparquet.so.2300 | 7,175,592 |
| GDAL/lib/libxerces-c-3.3.so | 6,306,632 |
| GDAL/lib/libjxl.so.0.11 | 6,214,712 |
| GDAL/lib/libgeos.so.3.14.1 | 5,987,000 |
| GDAL/lib/libproj.so.25 | 5,978,120 |
| GDAL/lib/libhdf5.so.320 | 5,578,848 |
| GDAL/lib/libpoppler.so.151 | 4,577,432 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/libclrjit.so | 3,918,944 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Linq.Expressions.dll | 3,745,280 |
| GDAL/lib/libcrypto.so.1.1 | 3,211,416 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Data.Common.dll | 2,877,440 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/libmscordaccore.so | 2,518,744 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Security.Cryptography.dll | 2,302,464 |
| app/AWSSDK.Core.dll | 2,215,016 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Private.DataContractSerialization.dll | 2,075,648 |
| GDAL/lib/libcfitsio.so.10 | 2,025,896 |
| GDAL/lib/libnetcdf.so.22 | 1,994,896 |
| app/Geospatial.Core.dll | 1,891,840 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Text.Json.dll | 1,781,760 |
| GDAL/lib/libOpenEXR-3_4.so.33 | 1,763,960 |
| dotnet/shared/Microsoft.NETCore.App/9.0.17/System.Net.Http.dll | 1,753,600 |

## Notes / concerns

1. **No GDAL version mismatch**: MaxRev 3.12.3.499 ships exactly GDAL 3.12.3 (the version the app
   ships on Windows). Newer MaxRev lines (3.12.4.520, 3.13.x) exist but were deliberately not used.
2. **HDF5 2.0.0 availability**: satisfied directly from the official HDF Group 2.0.0 GitHub release
   (Ubuntu 24.04 gcc binary). Soversion is **.so.320** (320.0.0). Requires the host glibc to be
   Ubuntu-24.04-compatible (glibc >= 2.39 era); Posit Connect Cloud images should qualify — verify
   `ldd --version` on the target if load errors occur.
3. **Two different HDF5 builds coexist in the bundle** with the same SONAME `libhdf5.so.320`:
   `natives/` (HDF Group official 2.0.0, statically-linked zlib/aec, for the app's P/Invoke) and
   `GDAL/lib/libhdf5.so.320` (MaxRev's smaller HDF5 2.0.x build, dependency of libgdal's HDF5 driver).
   Same soversion → ABI-compatible; whichever directory appears first on `LD_LIBRARY_PATH` will
   satisfy both. Not expected to cause problems, but if HDF5-in-GDAL rasters misbehave, put
   `natives/` first and it will service both consumers.
4. **Symlinks materialized**: upstream `libhdf5.so` / `libhdf5.so.320` are symlinks; they are staged
   as full identical copies (2 x 20.5 MB) so the bundle survives zip/copy deployment from Windows.
   If bundle size matters, recreate them as symlinks in a Linux post-deploy step and keep only
   `libhdf5.so.320.0.0`.
5. **libe_sqlite3.so** has no SONAME and needs only libc — drop-in for SQLitePCLRaw's
   `DllImport("e_sqlite3")`.
6. System libraries the target Linux host must still provide (not bundled):
   `libstdc++.so.6`, `libgcc_s.so.1`, `libm.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2`, plus for
   .NET: `libicu` (or set `DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1`), `zlib`, `openssl` (dotnet uses
   its own bundled OpenSSL shim per distro; GDAL bundles libcrypto/libssl 3.2.0 and 1.1 itself).
7. Raw downloads kept in `../downloads/` (sibling of linux-bundle in the scratchpad) for re-staging.
