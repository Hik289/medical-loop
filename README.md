<h1 align="center">HumanChoice</h1>

<p align="center"><strong>Learning When and What to Ask through Outcome-Grounded Expert Iteration</strong></p>

<p align="center">
  <a href="#method"><img alt="Method" src="https://img.shields.io/badge/Method-HumanChoice-245C87?style=flat-square"></a>
  <a href="docs/results.md"><img alt="Results" src="https://img.shields.io/badge/Results-E1%E2%80%93E8-166865?style=flat-square"></a>
  <a href="docs/reproduction.md"><img alt="Reproduction guide" src="https://img.shields.io/badge/Guide-Reproduce-86541B?style=flat-square"></a>
  <a href="#citation"><img alt="Citation" src="https://img.shields.io/badge/Citation-BibTeX-475569?style=flat-square"></a>
</p>

<p align="center">
  <img src="assets/humanchoice-intuition.png" width="100%" alt="HumanChoice overview. The autonomous agent executes its proposal. HumanChoice selects a local question, asks now or waits, and compares the original and revised actions after a human response.">
</p>

**Can an agent improve a decision by asking one well-timed question?** HumanChoice keeps an autonomous action ready. At a decision point it selects one local multiple-choice question, decides whether to ask now or wait, and, if a person answers, chooses between its original action and a revised one. The agent completes the task in either case.

## Method

| Module | Decision | Training signal |
| --- | --- | --- |
| **Q · Question Selector** | Which local question to present | Relative utility of completed question branches |
| **G · Ask Gate** | Ask now, or execute the next step and retain the option to ask later | Utility of Ask versus Wait |
| **A · Action Selector** | Which valid action to execute after the response | Final task quality of the original and revised actions |

A fixed language model generates the candidate questions and actions. The method uses three small MLPs (18,819 parameters in total). Its **Expert Iteration** loop scores completed training branches and updates **A → Q → G**, so question selection and timing are trained against the decisions the downstream modules actually make. The archived minibatch runner is a pipeline check; its simplified targets and replay are documented separately from the paper method.

## Results at a glance

| Setting | Archived observation | Interpretation |
| --- | --- | --- |
| Medical record categorization, same-model simulated respondent (270 records) | **45.56% → 46.30%** accuracy; net rescue **+0.74 pp**, 95% CI **[0.00, 1.85]** | Small effect; the interval touches zero |
| Expert Iteration, simulated responses (3 seeds) | **−1.11 pp → 0.00 pp** net rescue from iteration 0 to 1; unchanged through iteration 3 | Recovered autonomous performance |
| NoisyToolBench, benchmark gold clarification (200 tasks) | **14.88% → 47.71%** parameter match | A controlled information contrast; no human participant |
| Human availability set to zero, medical sweep | **45.56%** accuracy and **100%** completion, matching the autonomous baseline | Autonomous fallback in the archived sweep |

[Experiment ledger and limitations →](docs/results.md)

## Quick start

Python 3.10+ is recommended. These checks run locally without an account, credentials, or medical data.

~~~bash
git clone https://github.com/Hik289/medical-loop.git
cd medical-loop
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/anchor2_assert.py
python scripts/check_results.py
python figures/draw_intuition.py
~~~

## Repository scope

| Path | Contents |
| --- | --- |
| [experiments/](experiments/) | Autonomous and few-shot baselines, pipeline, module training, metrics, native ClarifyBench wrapper, data audits and pilot form generators |
| [experiments/results/](experiments/results/) | Reviewed aggregate results and coverage summaries |
| [figures/](figures/) · [assets/](assets/) | Source and PNG for the overview figure; archived figure utilities |
| [docs/reproduction.md](docs/reproduction.md) | How to rerun the available components and what is missing for full reproduction |

Medical records, participant responses, answer keys, per-record predictions and trained weights are not distributed. See the [MIMIC-IV access terms](https://physionet.org/content/mimiciv/3.1/) and [PhysioNet guidance for LLM use](https://physionet.org/news/post/llm-responsible-use/) before working with the source data.

## Citation

The repository accompanies a research manuscript. No public paper URL or venue acceptance is listed here.

~~~bibtex
@misc{humanchoice2026,
  title = {HumanChoice: Learning When and What to Ask through Outcome-Grounded Expert Iteration},
  year  = {2026},
  url   = {https://github.com/Hik289/medical-loop},
  note  = {Research code and experiment artifacts}
}
~~~
