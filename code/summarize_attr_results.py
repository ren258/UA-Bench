import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict, field
from glob import glob
from typing import Dict, List, Optional, Tuple


TASKS = {
    "Commonsense QA": {
        "answerable": ["gaia", "musique_1000"],
        "unanswerable": ["selfaware"],
    },
    "Math QA": {
        "answerable": ["olympiadbench-math"],
        "unanswerable": ["gsm8k-mip", "math-mip"],
    },
}


FNAME_RE = re.compile(r"^result__([^_]+(?:_[^_]+)*)__abstain_attr__([^_]+(?:_[^_]+)*)\.json$")


def parse_dataset_model_from_filename(path: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None
    return m.group(1), m.group(2)


def _get_int(d: Dict, key: str, default: int = 0) -> int:
    v = d.get(key, default)
    try:
        return int(v)
    except Exception:
        return default


def extract_counts(payload: Dict) -> Dict[str, int]:
    summary = payload.get("summary", {})
    counts = summary.get("counts", {}) or {}

    n_total = _get_int(counts, "n_total", 0)

    gt_answerable = _get_int(counts, "n_gt_answerable", 0)
    gt_abstain = _get_int(counts, "n_gt_abstain", max(0, n_total - gt_answerable))

    gt_answerable_correct = _get_int(
        counts,
        "gt_answerable_correct",
        _get_int(counts, "gt_answerable_pred_answerable_correct", 0),
    )

    gt_abstain_pred_data_uncertain = _get_int(counts, "gt_abstain_pred_data_uncertain", 0)

    gt_answerable_pred_model_uncertain = _get_int(counts, "gt_answerable_pred_model_uncertain", 0)
    gt_abstain_pred_model_uncertain = _get_int(counts, "gt_abstain_pred_model_uncertain", 0)

    gt_answerable_pred_data_uncertain = _get_int(counts, "gt_answerable_pred_data_uncertain", 0)


    pred_model_uncertain = _get_int(
        counts,
        "pred_model_uncertain",
        gt_answerable_pred_model_uncertain + gt_abstain_pred_model_uncertain,
    )

    return {
        "n_total": n_total,
        "gt_answerable": gt_answerable,
        "gt_abstain": gt_abstain,
        "gt_answerable_correct": gt_answerable_correct,
        "gt_abstain_pred_data_uncertain": gt_abstain_pred_data_uncertain,
        "gt_answerable_pred_model_uncertain": gt_answerable_pred_model_uncertain,
        "gt_answerable_pred_data_uncertain": gt_answerable_pred_data_uncertain,
        "gt_abstain_pred_model_uncertain": gt_abstain_pred_model_uncertain,
        "pred_model_uncertain": pred_model_uncertain,
    }

@dataclass
class MetricResult:
    gt_answerable: int = 0
    gt_abstain: int = 0
    gt_answerable_correct: int = 0

    tp_du: int = 0
    fp_mu: int = 0

    tp_mu: int = 0
    fp_du: int = 0

    by_dataset: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    @property
    def N(self) -> int:
        return self.gt_abstain

    @property
    def M(self) -> int:
        return max(0, self.gt_answerable - self.gt_answerable_correct)

    def accuracy(self) -> float:
        return self.safe_div(self.gt_answerable_correct, self.gt_answerable)

    def du_f1(self) -> float:
        N, M = self.N, self.M
        tp = self.tp_du
        fp = self.fp_du
        fn = max(0, N - tp)

        tp_t = self.safe_div(tp, N)
        fp_t = self.safe_div(fp, M)
        fn_t = self.safe_div(fn, M)

        p = self.safe_div(tp_t, tp_t + fp_t)
        r = self.safe_div(tp_t, tp_t + fn_t)
        return self.safe_div(2.0 * p * r, (p + r))

    def mu_f1(self) -> float:
        N, M = self.N, self.M
        tp = self.tp_mu
        fp = self.fp_mu
        fn = max(0, M - tp)

        tp_t = self.safe_div(tp, M)
        fp_t = self.safe_div(fp, N)
        fn_t = self.safe_div(fn, N)

        p = self.safe_div(tp_t, tp_t + fp_t)
        r = self.safe_div(tp_t, tp_t + fn_t)
        return self.safe_div(2.0 * p * r, (p + r))

    def sd_precision(self) -> float:
        N, M = self.N, self.M

        tp_mu_t = self.safe_div(self.tp_mu, M)
        tp_du_t = self.safe_div(self.tp_du, N)
        fp_mu_t = self.safe_div(self.fp_mu, N)
        fp_du_t = self.safe_div(self.fp_du, M)

        num = tp_mu_t + tp_du_t
        den = num + (fp_mu_t + fp_du_t)

        return self.safe_div(num, den)
    
    def balanced_accuracy(self) -> float:
        TP = self.tp_mu
        FN = max(0, self.M - self.tp_mu)

        TN = self.tp_du
        FP = self.fp_mu

        recall_pos = self.safe_div(TP, TP + FN)
        recall_neg = self.safe_div(TN, TN + FP)

        return 0.5 * (recall_pos + recall_neg)


    def to_summary_dict(self) -> Dict:
        return {
            "gt_answerable": self.gt_answerable,
            "gt_abstain": self.gt_abstain,
            "M_answerable_error": self.M,
            "N_unanswerable": self.N,

            "accuracy": self.accuracy(),
            "du_f1": self.du_f1(),
            "mu_f1": self.mu_f1(),
            "avg_f1": 0.5 * (self.du_f1() + self.mu_f1()),
            "sd_precision": self.sd_precision(),
            "balanced_accuracy": self.balanced_accuracy(),

            "gt_answerable_correct": self.gt_answerable_correct,
            "tp_du": self.tp_du,
            "fp_du": self.fp_du,
            "tp_mu": self.tp_mu,
            "fp_mu": self.fp_mu,
        }
def aggregate_task_for_model(
    results_index: Dict[Tuple[str, str], str],
    model: str,
    task_name: str,
    task_cfg: Dict[str, List[str]],
) -> MetricResult:
    mr = MetricResult()

    def getc(c: Dict[str, int], k: str) -> int:
        return int(c.get(k, 0) or 0)

    def add_dataset(ds: str):
        path = results_index.get((ds, model))
        if not path or not os.path.isfile(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        c = extract_counts(payload)

        gt_answerable = c["gt_answerable"]
        gt_abstain = c["gt_abstain"]
        gt_answerable_correct = c["gt_answerable_correct"]

        gt_abstain_pred_data_uncertain = c["gt_abstain_pred_data_uncertain"]
        gt_abstain_pred_model_uncertain = c["gt_abstain_pred_model_uncertain"]

        gt_answerable_pred_model_uncertain = c["gt_answerable_pred_model_uncertain"]
        gt_answerable_pred_data_uncertain = c["gt_answerable_pred_data_uncertain"]

        mr.gt_answerable += gt_answerable
        mr.gt_abstain += gt_abstain
        mr.gt_answerable_correct += gt_answerable_correct

        mr.tp_du += gt_abstain_pred_data_uncertain
        mr.fp_mu += gt_abstain_pred_model_uncertain

        mr.tp_mu += gt_answerable_pred_model_uncertain
        mr.fp_du += gt_answerable_pred_data_uncertain

        mr.by_dataset[ds] = c

    for ds in task_cfg.get("answerable", []):
        add_dataset(ds)

    # unanswerable datasets
    for ds in task_cfg.get("unanswerable", []):
        add_dataset(ds)

    return mr


def build_results_index(results_dir: str) -> Dict[Tuple[str, str], str]:
    idx: Dict[Tuple[str, str], str] = {}
    for path in glob(os.path.join(results_dir, "*.json")):
        parsed = parse_dataset_model_from_filename(path)
        if not parsed:
            continue
        dataset, model = parsed
        idx[(dataset, model)] = path
    return idx


def discover_models(results_index: Dict[Tuple[str, str], str]) -> List[str]:
    models = sorted({m for (_, m) in results_index.keys()})
    return models


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_task_outputs(output_dir: str, task_name: str, model_to_metrics: Dict[str, MetricResult]):
    safe_task = task_name.replace(" ", "_").replace("/", "_")
    task_dir = os.path.join(output_dir, safe_task)
    ensure_dir(task_dir)

    csv_path = os.path.join(task_dir, "summary.csv")
    fieldnames = [
        "model",
        "n_total",
        "gt_answerable",
        "gt_abstain",
        "accuracy",
        "data_uncertain_recall",
        "model_uncertain_rate",
        "fidelity",
        "answerable_correct",
        "unanswerable_pred_data_uncertain",
        "pred_model_uncertain",
        "wrong_total",
        "wrong_pred_model_uncertain",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model, mr in model_to_metrics.items():
            row = {"model": model, **mr.to_summary_dict()}
            w.writerow({k: row.get(k, "") for k in fieldnames})

    json_path = os.path.join(task_dir, "summary.json")
    all_obj = {
        "task": task_name,
        "models": {m: {"summary": mr.to_summary_dict(), "by_dataset": mr.by_dataset} for m, mr in model_to_metrics.items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_obj, f, ensure_ascii=False, indent=2)

    per_model_dir = os.path.join(task_dir, "per_model")
    ensure_dir(per_model_dir)
    for model, mr in model_to_metrics.items():
        out = {"task": task_name, "model": model, "summary": mr.to_summary_dict(), "by_dataset": mr.by_dataset}
        with open(os.path.join(per_model_dir, f"{model}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, help="Directory containing result__{dataset}_attr__{model}.json")
    ap.add_argument("--output_dir", type=str, help="Directory to write aggregated summaries")
    ap.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names. If empty, auto-discover from filenames.",
    )
    args = ap.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    results_index = build_results_index(results_dir)
    if not results_index:
        raise RuntimeError(f"No valid result files found in: {results_dir}")

    if args.models.strip():
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = discover_models(results_index)

    for task_name, task_cfg in TASKS.items():
        model_to_metrics: Dict[str, MetricResult] = {}
        for model in models:
            print(f'[Processing] Task="{task_name}", Model="{model}"')
            mr = aggregate_task_for_model(results_index, model, task_name, task_cfg)
            model_to_metrics[model] = mr
        save_task_outputs(output_dir, task_name, model_to_metrics)

    print(f"[OK] Wrote summaries to: {output_dir}")


if __name__ == "__main__":
    main()
