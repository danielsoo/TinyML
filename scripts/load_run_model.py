#!/usr/bin/env python3
"""
학습된 특정 버전(예: v20)의 모델을 run 디렉터리에서 가져와
models/global_model.h5 및 (선택) run_config.yaml 로 복사합니다.
이후 --skip-train 로 압축/분석만 다시 돌릴 때 사용합니다.

Usage:
  # v20 최신 run 하나 가져오기 (단일 run 또는 스윕 내 run)
  python scripts/load_run_model.py --version v20

  # v20 특정 run_id (단일 run)
  python scripts/load_run_model.py --version v20 --run-id 2026-02-22_21-12-36

  # v20 스윕 내 특정 slug
  python scripts/load_run_model.py --version v20 --sweep-dir 2026-02-22_21-12-36_hyperparam_sweep --slug num60_bs64_ep1_lr0.0003_focal0.5

  # 가져온 뒤 사용할 config도 복사
  python scripts/load_run_model.py --version v20 --copy-config
"""

import argparse
import shutil
import sys
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parent.parent


def get_runs_base() -> Path:
    return ROOT / "data" / "processed" / "runs"


def find_latest_run_dir(version: str) -> Path | None:
    """
    version(예: v20) 아래 최신 run 디렉터리 1개 반환.
    단일 run: version/run_id. 스윕: version/<sweep_id>/run_<slug> 중 models/global_model.h5 있는 것.
    """
    base = get_runs_base() / version
    if not base.exists():
        return None
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if d.name.endswith("_hyperparam_sweep"):
            for run_sub in d.iterdir():
                if run_sub.is_dir() and run_sub.name.startswith("run_"):
                    h5 = run_sub / "models" / "global_model.h5"
                    if h5.exists():
                        candidates.append((run_sub.stat().st_mtime, run_sub))
        else:
            h5 = d / "models" / "global_model.h5"
            if h5.exists():
                candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_run_dir(version: str, run_id: str | None, sweep_dir: str | None, slug: str | None) -> Path | None:
    """
    단일 run: data/processed/runs/<version>/<run_id>
    스윕 run: data/processed/runs/<version>/<sweep_dir>/run_<slug>
    """
    base = get_runs_base() / version
    if not base.exists():
        return None
    if sweep_dir and slug:
        run_dir = base / sweep_dir / f"run_{slug}"
        return run_dir if run_dir.exists() else None
    if run_id:
        run_dir = base / run_id
        return run_dir if run_dir.exists() else None
    return find_latest_run_dir(version)


def main():
    parser = argparse.ArgumentParser(
        description="Load trained model from a version run dir into models/global_model.h5"
    )
    parser.add_argument("--version", default="v20", help="Version (e.g. v20)")
    parser.add_argument("--run-id", default=None, help="Run datetime id (e.g. 2026-02-22_21-12-36)")
    parser.add_argument("--sweep-dir", default=None, help="Sweep run dir name (e.g. 2026-02-22_21-12-36_hyperparam_sweep)")
    parser.add_argument("--slug", default=None, help="Sweep combo slug (e.g. num60_bs64_ep1_lr0.0003_focal0.5)")
    parser.add_argument("--copy-config", action="store_true", help="Copy run_config.yaml to config/run_config_<version>.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only print paths, do not copy")
    parser.add_argument("--list", action="store_true", dest="list_runs", help="List run dirs with models for this version, then exit")
    args = parser.parse_args()

    if args.list_runs:
        base = get_runs_base() / args.version
        if not base.exists():
            print(f"No dir: {base}")
            return 0
        print(f"Run dirs with models/global_model.h5 under {base}:")
        for d in sorted(base.iterdir(), key=lambda x: x.name, reverse=True):
            if not d.is_dir():
                continue
            if d.name.endswith("_hyperparam_sweep"):
                for run_sub in sorted(d.iterdir(), key=lambda x: x.name):
                    if run_sub.is_dir() and run_sub.name.startswith("run_"):
                        if (run_sub / "models" / "global_model.h5").exists():
                            print(f"  {d.name}/{run_sub.name}")
            elif (d / "models" / "global_model.h5").exists():
                print(f"  {d.name}")
        return 0

    run_dir = find_run_dir(args.version, args.run_id, args.sweep_dir, args.slug)
    if run_dir is None:
        print(f"❌ No run dir found for version={args.version}", file=sys.stderr)
        if args.run_id:
            print(f"   run_id={args.run_id}", file=sys.stderr)
        if args.sweep_dir and args.slug:
            print(f"   sweep_dir={args.sweep_dir}, slug={args.slug}", file=sys.stderr)
        sys.exit(1)

    src_h5 = run_dir / "models" / "global_model.h5"
    if not src_h5.exists():
        # 레거시: run_dir에 models가 없을 수 있음. src/models 쪽은 매 run 덮어쓰므로 run별로 없음.
        print(f"❌ Run dir has no models/global_model.h5: {run_dir}", file=sys.stderr)
        print("   (이 run은 파이프라인에서 models를 run_dir로 복사하기 전에 끝났을 수 있음)", file=sys.stderr)
        sys.exit(1)

    dst_models = ROOT / "models"
    dst_models.mkdir(parents=True, exist_ok=True)
    dst_h5 = dst_models / "global_model.h5"

    if args.dry_run:
        print(f"Would copy: {src_h5} -> {dst_h5}")
        if args.copy_config:
            src_cfg = run_dir / "run_config.yaml"
            if src_cfg.exists():
                print(f"Would copy: {src_cfg} -> config/run_config_{args.version}.yaml")
        return 0

    shutil.copy2(src_h5, dst_h5)
    print(f"✅ Copied: {src_h5} -> {dst_h5}")

    if args.copy_config:
        src_cfg = run_dir / "run_config.yaml"
        if src_cfg.exists():
            config_dir = ROOT / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            dst_cfg = config_dir / f"run_config_{args.version}.yaml"
            shutil.copy2(src_cfg, dst_cfg)
            print(f"✅ Copied: {src_cfg} -> {dst_cfg}")
            print(f"   Then run: python run.py --skip-train --config {dst_cfg}")
        else:
            print(f"⚠️ No run_config.yaml in {run_dir}")

    print(f"\n다음으로 압축·분석만 재실행:")
    print(f"  python run.py --skip-train --config <config_used_for_v20>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
