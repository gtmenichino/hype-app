# src/hypetool/cli/main.py
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

# this file â†’ cli/ â†’ hypetool/ â†’ src/ â†’ repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR   = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hypetool.core.run_from_yaml import run_from_yaml
from hypetool.inputs import load as load_cfg
from hypetool.functions import path_utils as pu  # helpers

def _default_yaml() -> Path:
    ex = _REPO_ROOT / "examples" / "basic_case" / "input" / "inputs.yaml"
    pkg_default = _SRC_DIR / "hypetool" / "inputs.yaml"
    return ex if ex.exists() else pkg_default

def _clean_output_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    for p in path.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypetool",
        description="Run the hypetool pipeline from a YAML configuration."
    )
    g_src = parser.add_mutually_exclusive_group()
    g_src.add_argument("--yaml", help="Path to YAML file (default: examples/basic_case/input/inputs.yaml or src/hypetool/inputs.yaml)")
    g_src.add_argument("--yaml-stdin", action="store_true", help="Read YAML contents from STDIN")

    parser.add_argument("--out", dest="out_dir", help="Optional override for output_directory in YAML")
    parser.add_argument("--figures", action="store_true", help="Generate Step 8 figures (heavier)")
    parser.add_argument("--dry-run", action="store_true", help="Stop after validating config & preparing workspace")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before running")
    parser.add_argument("--bootstrap-exes", action="store_true", help="If shipped mf6/mp7 are missing, download them into <package>/bin/modflow")
    parser.add_argument("--no-config-print", action="store_true", help="Do not print the config table before run")

    args = parser.parse_args(argv)

    # Determine YAML source
    tmp_yaml_path: Optional[Path] = None
    if args.yaml_stdin:
        yaml_text = sys.stdin.read()
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml", prefix="hypetool_stdin_", dir=str(_REPO_ROOT))
        tmp.write(yaml_text); tmp.flush(); tmp.close()
        yaml_path = Path(tmp.name).resolve()
        tmp_yaml_path = yaml_path
        print(f"[info] Read YAML from STDIN â†’ {yaml_path}")
    else:
        yaml_path = Path(args.yaml).expanduser().resolve() if args.yaml else _default_yaml()
        if not yaml_path.exists():
            print(f"[ERROR] YAML path not found: {yaml_path}")
            return 2

    # Prefer the packaged MF bin; optionally bootstrap if missing
    pkg_bin = pu.default_modflow_bin()
    if pkg_bin.exists():
        pu.add_modflow_executables(pkg_bin)
        print(f"[info] Using packaged MODFLOW bin: {pkg_bin}")
    elif args.bootstrap_exes:
        try:
            pu.download_modflow(pkg_bin)
            pu.add_modflow_executables(pkg_bin)
            print(f"[info] Bootstrapped MODFLOW bin: {pkg_bin}")
        except Exception as e:
            print(f"[WARN] Could not bootstrap executables: {e}")

    # Optional: print the effective configuration table
    if not args.no_config_print:
        try:
            cfg = load_cfg(yaml_path)
            if args.out_dir:
                cfg.output_directory = Path(args.out_dir).expanduser().resolve()
            pu.print_config_table(cfg, max_value_width=80)
        except Exception as e:
            print(f"[WARN] Could not load/print config table: {e}")

    # Optional: clean output directory BEFORE the run
    out_override: Optional[Path] = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if args.clean:
        try:
            if out_override:
                _clean_output_dir(out_override)
                print(f"[info] Cleaned output directory: {out_override}")
            else:
                cfg_for_clean = load_cfg(yaml_path)
                if not cfg_for_clean.output_directory:
                    print("[WARN] No output_directory in YAML; nothing to clean.")
                else:
                    _clean_output_dir(Path(cfg_for_clean.output_directory))
                    print(f"[info] Cleaned output directory: {cfg_for_clean.output_directory}")
        except Exception as e:
            print(f"[WARN] Clean requested but failed to clean: {e}")

    def _log(msg: str) -> None:
        print(str(msg))

    print("[info] Starting hypetool run â€¦")
    outputs = run_from_yaml(
        yaml_path=yaml_path,
        out_folder=out_override,
        log=_log,
        dry_run=bool(args.dry_run),
        make_figures=bool(args.figures),
    )

    if outputs:
        print("\n[info] Outputs:")
        for k, v in outputs.items():
            if v:
                print(f"  {k}: {v}")

    if tmp_yaml_path:
        try:
            tmp_yaml_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("[info] Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


