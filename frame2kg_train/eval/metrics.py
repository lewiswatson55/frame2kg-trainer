from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import json as _json

from transformers import EvalPrediction

from frame2kg.eval.json_utils import first_json_object, normalise_json_text, slice_from_assistant


def to_box(loc) -> Tuple[float, float, float, float]:
    nums: List[float] = []
    if isinstance(loc, dict):
        for k in ("x1", "y1", "x2", "y2"):
            v = loc.get(k, None)
            try: nums.append(float(v))
            except Exception: pass
    elif isinstance(loc, (list, tuple)):
        for v in loc:
            try: nums.append(float(v))
            except Exception: pass
    else:
        import re
        for m in re.findall(r"-?\d+(?:\.\d+)?", str(loc)):
            try: nums.append(float(m))
            except Exception: pass
    while len(nums) < 4:
        nums.append(0.0)
    x1, y1, x2, y2 = nums[:4]
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    def _finite(x):
        return float(x) if np.isfinite(x) else 0.0
    return _finite(x1), _finite(y1), _finite(x2), _finite(y2)


def _valid(b):
    x1,y1,x2,y2=b; return (x2-x1)>0 and (y2-y1)>0


def iou(a,b)->float:
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    inter_x1,inter_y1=max(ax1,bx1),max(ay1,by1)
    inter_x2,inter_y2=min(ax2,bx2),min(ay2,by2)
    iw=max(0.0,inter_x2-inter_x1); ih=max(0.0,inter_y2-inter_y1)
    inter=iw*ih
    if inter==0: return 0.0
    area_a=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    area_b=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    denom=area_a+area_b-inter
    return inter/denom if denom>0 else 0.0


def match_nodes(gt: List[Dict[str, Any]], pr: List[Dict[str, Any]], thr: float = 0.5):
    def norm_label(x):
        return (str(x or "")).strip().casefold()
    matches=[]; used=set(); ious=[]
    for gi, g in enumerate(gt):
        gb = to_box(g.get("location","0,0,0,0"))
        if not _valid(gb): continue
        gl = norm_label(g.get("label"))
        best = (-1, 0.0)
        for pj, p in enumerate(pr):
            if pj in used: continue
            if norm_label(p.get("label")) != gl: continue
            pb = to_box(p.get("location","0,0,0,0"))
            if not _valid(pb): continue
            s = iou(gb, pb)
            if s >= thr and s > best[1]: best = (pj, s)
        if best[0] != -1:
            matches.append((gi, best[0])); used.add(best[0]); ious.append(best[1])
    return matches, ious



def calc_node_scores(gt, pr, thr:float=0.5):
    m, ious = match_nodes(gt,pr,thr)
    tp=len(m); fp=max(0,len(pr)-tp); fn=max(0,len(gt)-tp)
    P=tp/(tp+fp+1e-9); R=tp/(tp+fn+1e-9); F1=2*P*R/(P+R+1e-9)
    mean_iou=float(np.mean(ious)) if ious else 0.0
    return P,R,F1,{g:p for g,p in m},mean_iou


def calc_edge_scores(gt_edges, pr_edges, node_map, gt_nodes, pr_nodes):
    gt_id_by_idx = {i: n.get("id", i) for i, n in enumerate(gt_nodes)}
    pr_id_by_idx = {i: n.get("id", i) for i, n in enumerate(pr_nodes)}
    pr_idx_by_id = {v: k for k, v in pr_id_by_idx.items()}
    pridx_to_gtid = {pr_idx: gt_id_by_idx[gt_idx] for gt_idx, pr_idx in node_map.items()}

    def canon_pred(s):  # predicate only
        return (str(s or "")).strip().casefold()

    def canon_edge(e):  # for GT set
        return (canon_pred(e.get("predicate")), e.get("source"), e.get("target"))

    gt_set = {canon_edge(e) for e in gt_edges}
    pr_set = set()
    for e in pr_edges:
        s_id, t_id = e.get("source"), e.get("target")
        if s_id in pr_idx_by_id and t_id in pr_idx_by_id:
            s_pr = pr_idx_by_id[s_id]; t_pr = pr_idx_by_id[t_id]
            if s_pr in pridx_to_gtid and t_pr in pridx_to_gtid:
                pr_set.add((canon_pred(e.get("predicate")),
                            pridx_to_gtid[s_pr], pridx_to_gtid[t_pr]))
    tp = len(gt_set & pr_set); fp = len(pr_set - gt_set); fn = len(gt_set - pr_set)
    P = tp / (tp + fp + 1e-9); R = tp / (tp + fn + 1e-9); F1 = 2 * P * R / (P + R + 1e-9)
    return P, R, F1



# Token helpers for compute_metrics
import numpy as np

def _to_2d_token_array(preds):
    if isinstance(preds,(tuple,list)) and len(preds)>0:
        maybe=preds[0]
        if hasattr(maybe,"ndim") or isinstance(maybe,(list,np.ndarray)):
            preds=maybe
    if isinstance(preds,np.ndarray):
        if preds.dtype==object:
            return [list(map(int,row.tolist())) for row in preds]
        if preds.ndim==2: return preds.tolist()
        if preds.ndim==1: return [preds.tolist()]
    if hasattr(preds,"tolist"): preds=preds.tolist()
    if isinstance(preds,list):
        if len(preds)==0: return []
        if isinstance(preds[0],(list,tuple,np.ndarray)):
            return [list(map(int,list(row))) for row in preds]
        if isinstance(preds[0],(int,np.integer)):
            return [list(map(int,preds))]
    return []


def _sanitize(mat, pad_id:int):
    out=[]
    for row in mat:
        safe=[]
        for tok in row:
            try: t=int(tok)
            except Exception: t=pad_id
            if t<0: t=pad_id
            safe.append(t)
        out.append(safe)
    return out


def make_compute_metrics(tokenizer):
    # Toggle verbose logs with env vars (or flip defaults here)
    DEBUG = os.getenv("F2KG_DEBUG_METRICS", "1").lower() not in ("0", "false", "no")
    SHOW_LABELS = os.getenv("F2KG_DEBUG_SHOW_LABELS", "0").lower() in ("1", "true", "yes")
    SNIP = int(os.getenv("F2KG_DEBUG_SNIP", "300"))

    def _snip(s: str, n: int = SNIP) -> str:
        return s if len(s) <= n else s[:n] + "…"

    def compute(eval_pred: EvalPrediction | tuple):
        if isinstance(eval_pred, (tuple, list)):
            preds, labels = eval_pred
        else:
            preds, labels = eval_pred.predictions, eval_pred.label_ids

        pad = tokenizer.pad_token_id or 0

        pred_tokens = _sanitize(_to_2d_token_array(preds), pad)
        pred_str = tokenizer.batch_decode(
            pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Prefer attached gold texts when provided on compute._eval_label_texts
        if hasattr(compute, "_eval_label_texts") and getattr(compute, "_eval_label_texts"):
            label_str = list(compute._eval_label_texts)  # already JSON strings
        else:
            lab_tokens = _sanitize(_to_2d_token_array(labels), pad)
            label_str = tokenizer.batch_decode(
                lab_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # Parse labels independently
        label_jsons = []
        label_ok = []
        for i, l_txt in enumerate(label_str):
            try:
                lj = first_json_object(l_txt)
                if lj is None:
                    lj = _json.loads(l_txt)  # allow pure JSON string
            except Exception:
                lj = None
            label_jsons.append(lj)
            label_ok.append(1 if lj is not None else 0)
            if DEBUG and SHOW_LABELS:
                tag = "OK" if lj is not None else "FAIL"
                print(f"[metrics] LABEL[{i}] {tag}")
                if lj is not None:
                    print(_json.dumps(lj, ensure_ascii=False))
                else:
                    print("--- raw label ---")
                    print(_snip(repr(l_txt)))
                    print("------------------")

        # Parse predictions, log raw on failure
        pred_jsons = []
        json_ok = []
        for i, p_txt in enumerate(pred_str):
            pj = first_json_object(p_txt)
            pred_jsons.append(pj)
            json_ok.append(1 if pj is not None else 0)
            if DEBUG:
                tag = "OK" if pj is not None else "FAIL"
                print(f"[metrics] PRED[{i}] {tag}")
                if pj is not None:
                    print(_json.dumps(pj, ensure_ascii=False))
                else:
                    print("--- raw pred ---")
                    print(_snip(repr(p_txt)))
                    print("----------------")

        # Score rows with both sides parsed
        nP = []; nR = []; nF = []; nIoU = []; eP = []; eR = []; eF = []
        for i, (pj, lj) in enumerate(zip(pred_jsons, label_jsons)):
            if pj is None or lj is None:
                continue
            gt_nodes = lj.get("nodes", []); pr_nodes = pj.get("nodes", [])
            gt_edges = lj.get("edges", []); pr_edges = pj.get("edges", [])

            P, R, F, node_map, iou = calc_node_scores(gt_nodes, pr_nodes)
            p, r, f = calc_edge_scores(gt_edges, pr_edges, node_map, gt_nodes, pr_nodes)

            nP.append(P); nR.append(R); nF.append(F); nIoU.append(iou)
            eP.append(p); eR.append(r); eF.append(f)

            if DEBUG:
                # Helpful quick glance
                print(f"[metrics] ROW[{i}] nodesF1={F:.3f} (P={P:.3f}, R={R:.3f})  "
                      f"edgesF1={f:.3f} (P={p:.3f}, R={r:.3f})  IoU={iou:.3f}")

                # If zero matches, dump label sets (lowercased) to spot naming drift
                if F == 0.0 or f == 0.0:
                    gt_labs = [str(n.get('label', '')).strip().casefold() for n in gt_nodes]
                    pr_labs = [str(n.get('label', '')).strip().casefold() for n in pr_nodes]
                    print(f"[metrics]   GT labels: {sorted(set(gt_labs))}")
                    print(f"[metrics]   PR labels: {sorted(set(pr_labs))}")

        mean = lambda x: float(np.mean(x)) if x else 0.0
        return {
            "json_validity": mean(json_ok),
            "label_json_validity": mean(label_ok),
            "node_P": mean(nP), "node_R": mean(nR), "node_F1": mean(nF),
            "node_bbox_iou": mean(nIoU),
            "edge_P": mean(eP), "edge_R": mean(eR), "edge_F1": mean(eF),
            "total_examples": len(pred_str),
            "scored_examples": len(nP),
        }
    return compute


