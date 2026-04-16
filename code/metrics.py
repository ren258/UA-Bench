from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


def pct(num: int, den: int) -> float:
    """Return percentage in [0, 100], rounded to 1 decimal place."""
    if den <= 0:
        return 0.0
    return round(100.0 * num / den, 1)


@dataclass
class Counts:
    n_total: int = 0

    n_gt_answerable: int = 0
    n_gt_abstain: int = 0

    pred_answerable: int = 0
    pred_data_uncertain: int = 0
    pred_model_uncertain: int = 0
    pred_parse_error: int = 0

    gt_answerable_correct: int = 0
    gt_answerable_wrong: int = 0
    gt_answerable_pred_data_uncertain: int = 0
    gt_answerable_pred_model_uncertain: int = 0
    gt_answerable_parse_error: int = 0

    gt_abstain_pred_answerable: int = 0
    gt_abstain_pred_data_uncertain: int = 0
    gt_abstain_pred_model_uncertain: int = 0
    gt_abstain_parse_error: int = 0

    effective_model_uncertain_from_wrong: int = 0


    def observe_gt(self, gt_label: str) -> None:
        g = (gt_label or "").strip().upper()
        if g == "ANSWERABLE":
            self.n_gt_answerable += 1
        elif g == "ABSTAIN":
            self.n_gt_abstain += 1
        else:
            raise ValueError(f"Unknown gt_label={gt_label!r}, expected 'ANSWERABLE' or 'ABSTAIN'.")

    def observe_pred(self, pred_label: Optional[str], parse_error: Optional[str]) -> None:
        if parse_error is not None:
            self.pred_parse_error += 1
            return

        p = (pred_label or "").strip().upper()
        if p == "ANSWERABLE":
            self.pred_answerable += 1
        elif p == "DATA_UNCERTAIN":
            self.pred_data_uncertain += 1
        elif p == "MODEL_UNCERTAIN":
            self.pred_model_uncertain += 1
        else:
            raise ValueError(
                f"Unknown pred_label={pred_label!r} with parse_error=None. "
                f"Expected 'ANSWERABLE'/'DATA_UNCERTAIN'/'MODEL_UNCERTAIN'."
            )

    def observe_outcome(
        self,
        gt_label: str,
        pred_label: Optional[str],
        parse_error: Optional[str],
        is_correct: Optional[bool],
    ) -> None:
        g = (gt_label or "").strip().upper()

        if g == "ANSWERABLE":
            if parse_error is not None:
                self.gt_answerable_parse_error += 1
                self.gt_answerable_wrong += 1
                self.effective_model_uncertain_from_wrong += 1
                return

            p = (pred_label or "").strip().upper()
            if p == "ANSWERABLE":
                if bool(is_correct):
                    self.gt_answerable_correct += 1
                else:
                    self.gt_answerable_wrong += 1
                    self.effective_model_uncertain_from_wrong += 1
            elif p == "DATA_UNCERTAIN":
                self.gt_answerable_pred_data_uncertain += 1
            elif p == "MODEL_UNCERTAIN":
                self.gt_answerable_pred_model_uncertain += 1
            else:
                raise ValueError(f"Unknown pred_label={pred_label!r} for GT=ANSWERABLE with parse_error=None.")

        elif g == "ABSTAIN":
            if parse_error is not None:
                self.gt_abstain_parse_error += 1
                return

            p = (pred_label or "").strip().upper()
            if p == "ANSWERABLE":
                self.gt_abstain_pred_answerable += 1
            elif p == "DATA_UNCERTAIN":
                self.gt_abstain_pred_data_uncertain += 1
            elif p == "MODEL_UNCERTAIN":
                self.gt_abstain_pred_model_uncertain += 1
            else:
                raise ValueError(f"Unknown pred_label={pred_label!r} for GT=ABSTAIN with parse_error=None.")
        else:
            raise ValueError(f"Unknown gt_label={gt_label!r}, expected 'ANSWERABLE' or 'ABSTAIN'.")


    def to_dict(self) -> Dict:
        return asdict(self)

    def compute_percents(self) -> Dict[str, float]:
        return {
            "pred_answerable_pct": pct(self.pred_answerable, self.n_total),
            "pred_data_uncertain_pct": pct(self.pred_data_uncertain, self.n_total),
            "pred_model_uncertain_pct": pct(self.pred_model_uncertain, self.n_total),
            "pred_parse_error_pct": pct(self.pred_parse_error, self.n_total),

            "gt_answerable_acc_pct": pct(self.gt_answerable_correct, self.n_gt_answerable),
            "gt_answerable_wrong_pct": pct(self.gt_answerable_wrong, self.n_gt_answerable),
            "gt_answerable_pred_data_uncertain_pct": pct(
                self.gt_answerable_pred_data_uncertain, self.n_gt_answerable
            ),
            "gt_answerable_pred_model_uncertain_pct": pct(
                self.gt_answerable_pred_model_uncertain, self.n_gt_answerable
            ),
            "gt_answerable_parse_error_pct": pct(
                self.gt_answerable_parse_error, self.n_gt_answerable
            ),

            "gt_abstain_pred_answerable_pct": pct(
                self.gt_abstain_pred_answerable, self.n_gt_abstain
            ),
            "gt_abstain_pred_data_uncertain_pct": pct(
                self.gt_abstain_pred_data_uncertain, self.n_gt_abstain
            ),
            "gt_abstain_pred_model_uncertain_pct": pct(
                self.gt_abstain_pred_model_uncertain, self.n_gt_abstain
            ),
            "gt_abstain_parse_error_pct": pct(
                self.gt_abstain_parse_error, self.n_gt_abstain
            ),

            "effective_model_uncertain_from_wrong_pct_over_gt_answerable": pct(
                self.effective_model_uncertain_from_wrong, self.n_gt_answerable
            ),
        }

    def compute_matrix(self) -> Dict[str, Dict[str, int]]:
        return {
            "GT_ANSWERABLE": {
                "Pred_ANSWERABLE_correct": self.gt_answerable_correct,
                "Pred_ANSWERABLE_wrong": self.gt_answerable_wrong,
                "Pred_DATA_UNCERTAIN": self.gt_answerable_pred_data_uncertain,
                "Pred_MODEL_UNCERTAIN": self.gt_answerable_pred_model_uncertain,
                "Pred_PARSE_ERROR": self.gt_answerable_parse_error,
            },
            "GT_ABSTAIN": {
                "Pred_ANSWERABLE": self.gt_abstain_pred_answerable,
                "Pred_DATA_UNCERTAIN": self.gt_abstain_pred_data_uncertain,
                "Pred_MODEL_UNCERTAIN": self.gt_abstain_pred_model_uncertain,
                "Pred_PARSE_ERROR": self.gt_abstain_parse_error,
            },
        }
    

    def to_percent_dict(self) -> Dict[str, float]:
        return self.compute_percents()

    def to_matrix_dict(self) -> Dict[str, Dict[str, int]]:
        return self.compute_matrix()
