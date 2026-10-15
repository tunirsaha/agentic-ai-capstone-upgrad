# Sahayak Health AI: Multi-Agent Triage for ASHA Workers

Capstone B for the upGrad AI agents programme. Sahayak is a decision-support assistant for ASHA (community health) workers in rural India. Given a patient's symptoms in plain language, it recommends exactly one next step, built as a six-agent pipeline on Google ADK.

| Level | Meaning |
|---|---|
| `WAIT` | manage at home and keep watching |
| `DOCTOR` | see a doctor or clinic soon |
| `ER` | go now to the nearest hospital / CHC / PHC, or call 108 |

It never names a disease, never suggests a medicine or dose, and every reply carries a disclaimer. It is decision support only and **not a medical device**.

Sending a sick person home (under-triage) can cost a life, while sending a mild case to a clinic (over-triage) only costs clinic time. So the system is judged on under-triage and ER cases sent home first, and on accuracy second.

```
patient text -> symptom_parser -> severity_scorer -> followup_asker
                                                          |
                                         [pause: health worker answers]
                                                          v
              safety evaluator <- response_formatter <- triage_decider (4 tools)
```

A worrying follow-up answer can only raise the level, never lower it. All reported safety numbers come from a deterministic rule judge, not from an LLM grading its own output.

## Results

Final run with `gpt-4o-mini`. Tables, failure analysis and limits are in [`final_report.pdf`](final_report.pdf), and every number comes from the executed notebooks in [`solution/executed/`](solution/executed/).

| Metric | Rule baseline (locked 50) | Final agent (locked 50) | Final agent (held-out test, 50) |
|---|---|---|---|
| Accuracy | 48% | 68% | **82%** |
| ER recall | 0% | 93.8% | **100%** |
| Under-triage | 44% | 8% | **0%** |
| Over-triage | n/a | 24% | 18% |
| ER sent home | n/a | 0 | 0 |

A few things worth noting:

- The rule-based baseline misses every ER case, because pneumonia patients describe their illness in everyday words that no keyword matches.
- Follow-up policy compliance and relevance were both 100%, and the loop-closure probe reached its target in 100% of cases with no de-escalations.
- The recalibrated severity rubric clearly beat the first one on a 30-case train-split A/B. On the locked 50 the result was mixed: higher accuracy and less over-triage, but one more missed ER case.
- On the locked 50, under-triage (8%) misses the workflow's 5% target and ER recall (93.8%) is below the starter code's 95% gate. The held-out sample meets every threshold measured on it.
- Samples are small (10 ER cases in the held-out set) and an LLM does not repeat itself exactly. Across three full runs the first rubric scored 66%, 52% and 62% accuracy.

![Trust dashboard](solution/figures/trust_dashboard.png)

## Data

[`gretelai/symptom_to_diagnosis`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) (Apache 2.0): 853 train and 212 test cases, 22 diagnosis labels, first-person symptom text. It is bundled in `Overview + Dataset/dataset.zip`.

The diagnosis is mapped to a care level (by the provided `TRIAGE_MAP`) only to build evaluation ground truth. The agent never outputs a diagnosis. The 212-case test split was never used for EDA, prompt writing or calibration.

## Repository layout

| Path | What it is |
|---|---|
| `Capstone B - Starter Files/` | the starter package exactly as provided by upGrad, untouched |
| `Overview + Dataset/` | the provided workflow PDF and the dataset |
| `solution/` | the completed, runnable project |
| `solution/learner/` | the four notebooks and `sahayak_starter.py` with my solution, plus the provided helpers |
| `solution/executed/` | the same four notebooks after a full run, with outputs |
| `solution/Run_All_in_Colab.ipynb` | runs the four notebooks in order in one Colab session |
| `solution/src/`, `tests/`, `data/` | provided dashboard, evaluation script, 12 harness tests and case data |
| `final_report.pdf` | the final report |

## Notebooks

| # | Notebook | What happens |
|---|---|---|
| 1 | `adk_foundations` | `LlmAgent`, `SequentialAgent`, session state, and a measured `ParallelAgent` speedup |
| 2 | `data_understanding_and_baseline` | EDA, triage-map review, the locked 50-case sample, a rule-based baseline and a parser plus severity trace |
| 3 | `agent_pipeline_development` | the six agents, the follow-up loop and its metrics, the deterministic judge, the 12 tests and four traced runs |
| 4 | `agent_evaluation_and_optimisation` | confusion matrix, failure analysis, train-only recalibration with an A/B table, and the final acceptance tables |

## What was provided and what I wrote

Provided: the notebook scaffolding and instructions, `utils.py`, `data_loader.py`, `sahayak_tools.py`, `src/`, `tests/`, `data/`, the problem statement and the workflow PDF.

Written by me:

- In `sahayak_starter.py`: `score_severity`, `decide_triage`, `run_policy_triage`, `safety_evaluator_agent`, the six agent instruction constants and `build_agentic_sahayak_pipeline`.
- In the notebooks: every marked solution cell, plus extra cells for the follow-up and loop-closure metrics, the train-only A/B and the held-out run.
- A small connector change in the notebooks and the two `src/` scripts so agents call OpenAI instead of a local Ollama server, each marked with a one-line comment.
- `Run_All_in_Colab.ipynb` and the final report.

## Running it

Built to run on Google Colab with a normal CPU runtime. No GPU is needed because the model is called through an API.

1. Upload the `solution/` folder to `MyDrive/` in Google Drive.
2. In Colab, add a secret named `OPENAI_API_KEY` and switch on notebook access.
3. Open `solution/Run_All_in_Colab.ipynb`, run the setup cell, then run the four notebook cells in order.

Executed copies are written to `solution/executed/` and the originals in `learner/` are not overwritten. A full run makes a few hundred `gpt-4o-mini` calls and took about half an hour. The last cell is optional and starts the dashboard. No Hugging Face token is needed.

To run the harness tests on their own, from `solution/`:

```bash
PYTHONPATH=learner python -m pytest tests -v
```

## Stack

Python 3.11+, Google ADK (`LlmAgent`, `SequentialAgent`, `ParallelAgent`, `Runner`), LiteLlm with `gpt-4o-mini`, FastAPI for the dashboard, pandas, scikit-learn, matplotlib and seaborn, Hugging Face `datasets`, and pytest. The provided code can also use Gemini (`GOOGLE_API_KEY`) or a local Ollama `hermes3:8b` when no OpenAI key is set.

## Limitations

- It is decision support, not a medical device. It has not been clinically validated and uses text only.
- Triage labels are a proxy built from diagnosis names, so some "errors" are label noise. Even trained nurses agree on exact triage level only about 74% of the time, so 100% accuracy is not a sensible target.
- The model is not deterministic and the samples are small, so single-digit percentage differences are within noise.
- The held-out cell ran in every full execution of the notebook. No prompt was changed in response to its numbers, but strictly it was not a single run.
- The dashboard header still says `hermes3:8b / Ollama local`. That is fixed text in the provided HTML, and the model that actually ran was `gpt-4o-mini`. The follow-up demo screenshot also shows a reply whose wording and label disagree, which the deterministic judge cannot catch.

The workflow PDF, task brief, notebook scaffolding and provided helper code are course material from upGrad and carry their own copyright. They are included so the project runs and reads end to end.
