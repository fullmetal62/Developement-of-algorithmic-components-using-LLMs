# Development of algorithmic components using LLMs

Course / research code: an LLM (Google Gemini) chooses which TSP metaheuristic **component** to run next inside a larger hybrid solver.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Set your API key (PowerShell):

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

Or in `cmd.exe`: `set GEMINI_API_KEY=your-key-here`

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

Optional environment variables (see `llm_prompting2.py`):

- `LLM_AI_ONLY` — default `1`; strict AI selection behaviour
- `LLM_STRICT` — default `0`

## Run

From this directory, with `GEMINI_API_KEY` set:

```bash
python runllm
```

By default `runllm` uses the included instance file `pr76.txt`. Pass another TSPLIB-style instance path by editing `runllm` or importing `run_once` from your own script.

## Layout

| File | Role |
|------|------|
| `llm_decision_selector_final.py` | `LLMDrivenAlgorithm` — orchestrates components and LLM decisions |
| `llm_prompting2.py` | Gemini prompting and operator registry |
| `BaseClasses.py`, `neighbourhood.py`, `dictionaries.py` | Metaheuristic building blocks and wiring |
| `my_*.py` | Instance, solution, and mutation implementations |
| `pr76.txt` | Sample TSP instance (76 cities) |
| `utilslogger.py` | CSV logging (`llm_prediction_log.csv`, gitignored when generated) |

## Notes

- Do **not** commit API keys; configuration uses `GEMINI_API_KEY` only.
- Python **3.10+** recommended (uses `str \| None` style hints in places).
