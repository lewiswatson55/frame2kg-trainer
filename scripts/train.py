from __future__ import annotations
import argparse, json, os, sys, yaml
from frame2kg_train.train.run import Runner


def main():
    p=argparse.ArgumentParser(description="Frame→KG multi‑backend trainer")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--overrides", type=str, nargs="*", default=[], help="k=v pairs to override config")
    args=p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg=yaml.safe_load(f)
    for kv in args.overrides:
        if "=" not in kv: continue
        k,v = kv.split("=",1)
        # dot.notation support
        def set_in(d, path, value):
            keys=path.split(".")
            cur=d
            for kk in keys[:-1]:
                if kk not in cur or not isinstance(cur[kk], dict): cur[kk]={}
                cur=cur[kk]
            # try cast
            if isinstance(value, str) and value.lower() in {"true","false"}: value = value.lower()=="true"
            else:
                try:
                    if isinstance(value, str) and "." in value: value=float(value)
                    elif isinstance(value, str): value=int(value)
                except Exception:
                    pass
            cur[keys[-1]]=value
        set_in(cfg,k,v)
    Runner(cfg).run()

if __name__ == "__main__":
    main()
