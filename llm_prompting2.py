from __future__ import annotations

import google.generativeai as genai
import os, csv, json, random, logging
from collections import defaultdict, deque
from typing import Dict, Tuple, Any, List


logger = logging.getLogger("tsp_llm.llm")
GEMINI_API_KEY = "your_API_key"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-pro") 
STRICT_AI_ONLY = os.getenv("LLM_AI_ONLY", "1").lower() in ("1", "true", "yes")

STRICT_NO_FALLBACK = os.getenv("LLM_STRICT", "0") in ("1", "true", "True")

# ---- tiny “operator registry” so the LLM knows what each name means ----
OPERATOR_REGISTRY: Dict[str, str] = {
    "RandomHillClimber": "Intensification: greedy local improvement using a mutation; good right after an improvement.",
    "VariableNeighbourhoodSearch": "Shake + local descent (2-opt, swap); good after stagnation to escape shallow minima.",
    "TabuSearchComponent": "Memory-based intensification with short-term tabu; good when cycling around a local minimum.",
    "RuinAndRecreate": "Large perturbation then repair; strong diversification after long stagnation.",
    "HybridGeneticComponent": "Population-based recombination + mutation; good for exploring new basins periodically.",
    "AdaptiveOperatorSelection":"Numeric learner that picks one core operator per call based on past performance (Δ = New -Old). Optionally bootstraps weights from history logs."
    # (Do not list Repeat/AOS here)
}
CONTROLLERS: Dict[str, str] = {"Repeat":"Run a single core operator multiple times in a row to intensify or consolidate gains"
                        }



def summarise_history_for_llm(log_path: str, k: int = 40):
    """
    Return (recent_rows, stats) where
      recent_rows: list[dict]  (last k rows)
      stats: dict with per-operator success rate, avg delta, last_improvement_iter, stagnation
    """
    if not os.path.exists(log_path):
        return [], {}

    rows = []
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    recent = rows[-k:]
    # compute stats
    per = defaultdict(lambda: {"n": 0, "succ": 0, "sum_delta": 0.0})
    last_improved_iter = None
    last_iter = None
    
    
    for r in rows:
        name = r.get("Component_Name")
        delta = float(r.get("Delta", "0") or 0)
        suc = (r.get("Success", "") in ("TRUE", "true", "1")) or (delta < 0)
        it = int(r.get("Iteration", "0") or 0)
        last_iter = it
        if suc and delta < 0:
            last_improved_iter = it
        if name:
            per[name]["n"] += 1
            per[name]["sum_delta"] += delta
            per[name]["succ"] += int(suc)

    stats = {}
    for name, d in per.items():
        n = max(1, d["n"])
        stats[name] = {
            "trials": d["n"],
            "success_rate": d["succ"] / n,
            "avg_delta": d["sum_delta"] / n,  # negative is good
        }

    if last_iter is not None:
        stats["stagnation_iters"] = (last_iter - (last_improved_iter or last_iter))
        stats["last_iter"] = last_iter

    return recent, stats


def ask_llm_for_component_name(
    current_solution,
    available_components: dict,
    *,
    history_log: str | None = "llm_prediction_log.csv",
    aos_weights: dict | None = None,
    temperature: float = 0.2,negative_is_improvement: bool = True,
                               meta: dict | None = None
):
    """
    Returns (component_name: str, args: dict).
    - Filters out controllers via CONTROLLERS.
    - Gives LLM concise operator cards, recent history summary, and (optional) AOS weights.
    """

    #1) filter to atomic operators(components other than Repeat and AOS)
    atomics = {k: v for k, v in available_components.items() if k not in CONTROLLERS}
    if not atomics:
        atomics = dict(available_components)  # last-resort fallback (shouldn’t happen)
    def _fail(reason: str):
        # never silently pick random; force explicit failure in AI-only mode
        msg = f"[LLM SELECTION FAILED] {reason}"
        if meta is not None:
            meta.update({"source":"fail", "reason":reason, "model":None})
        raise RuntimeError(msg)
    if model is None:
        if STRICT_AI_ONLY:
            return _fail("no_model_configured_or_missing_api_key")
        else:
            # (Optional: if you ever want non-strict, you could random-pick here)
            return _fail("no_model_and_fallback_disabled_by_default")
    # ---- 2) digest recent history ----
    recent, stats = summarise_history_for_llm(history_log) if history_log else ([], {})
    recent_light = [
        {
            "it": int(r.get("Iteration", 0)),
            "op": r.get("Operator_Atomic") or r.get("Component_Name") or r.get("Component"),
            "d": float(r.get("Delta", 0) or 0),
        }
        for r in recent[-15:]  # keep it short
    ]

    # ---- 3) build operator cards for the LLM ----
    cards = []
    for name in atomics.keys():
        cards.append({
            "name": name,
                
            "role": OPERATOR_REGISTRY.get(name, "No description available") ,
            "success_rate": stats.get(name, {}).get("success_rate"),
            "avg_delta": stats.get(name, {}).get("avg_delta"),  # negative is good
            "aos_weight": (aos_weights or {}).get(name),
        })

    # ---- 4) JSON prompt payload ----
    prompt_payload = {
        "task": "You are a decision controller inside a TSP metaheuristic solver.Choose the next operator based on the passed information.",
        "constraints": {
            "must_choose_one_of": list(atomics.keys()),
            "output_format": {
                "component_name": "string (one of allowed)",
                "args": "object (optional; may be empty)"
            },
            "do_not_explain": True
        },
        "state": {
            "current_cost": float(current_solution.get_objective()),
            "num_cities": len(current_solution.solution),
            "stagnation_iters": stats.get("stagnation_iters"),
            "last_iter": stats.get("last_iter"),
        },
        "operator_cards": cards,
        "recent_history": recent_light
    }

    try:
        resp = model.generate_content(
            json.dumps(prompt_payload),
            generation_config={
                "temperature": temperature,
                "top_p": 0.9,
                "response_mime_type": "application/json",
            },
        )
        raw = (resp.text or "").strip()
        data = json.loads(raw)

        name = data.get("component_name")
        args = data.get("args", {}) or {}

        if meta is not None:
            meta.update({"source":"ai", "model":getattr(model, "model_name", "gemini"), "raw": raw})

        # Validate
        if not isinstance(name, str) or name not in atomics:
            return _fail(f"invalid_component_name: {name!r}")
        if not isinstance(args, dict):
            return _fail("args_not_object")

        return name, args

    except Exception as e:
        # Treat any API/parse failures as hard errors in AI-only mode
        return _fail(f"api_or_parse_error:{type(e).__name__}:{e}")

    