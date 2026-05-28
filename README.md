# DS_Team7

2026-1 Data Science Team 7 term project.

## Project

Prediction of Profit Cost for Company & Device Recommendation for Customer

This repository contains the GSM smartphone dataset, preprocessing outputs,
modeling code, model outputs, plots, and handoff documents for the team project.

## Main Folders

- `content/`: raw and processed GSM CSV files used by the final model.
- `Preprocessing/`: preprocessing and feature engineering notebook/document.
- `inspection_data/`: missing-value, duplicate, distribution, and column inspection outputs.
- `modeling/`: final analysis/modeling notebook, executable Python scripts, model artifacts, outputs, plots, and evaluation documents.
- `presentation/`: final PPT presentation files and usage notes.
- `PATH_MAPPING.md`: local/GitHub/Colab path mapping.
- `check_project_paths.py`: path validation helper for local and Colab.

## Reproduce Modeling

```bash
python3 modeling/run_modeling.py --run
python3 modeling/two_way_solution.py --mode demo
python3 modeling/two_way_solution.py --mode recommend --budget-eur 400 --top-n 5
```

## Run In Colab

- Colab guide: `COLAB_RUN_GUIDE.md`
- Path mapping: `PATH_MAPPING.md`
- Open `modeling/GSM_modeling_colab.ipynb` in Colab. The Korean-name notebook `modeling/GSM__모델링.ipynb` has the same content.
- The notebook uses the GitHub root layout directly: `DS_Team7/content/*.csv` and `DS_Team7/modeling/run_modeling.py`.
- ZIP files are not used for model input loading. If the repo is not present in Colab, the first code cell clones `https://github.com/zwonhong/DS_Team7.git` into `/content/DS_Team7`.

## Term Project Compliance Notes

- Detailed requirement audit: `modeling/docs/07_TERM_PROJECT_SPEC_COMPLIANCE_AUDIT.md`
- External library/method explanations: `modeling/docs/04_EXTERNAL_LIBRARY_METHOD_EXPLANATIONS.md`
- Teamwork and learning writeup template: `modeling/docs/08_TEAMWORK_CONTRIBUTION_AND_LEARNING_TEMPLATE.md`
- Source citation checklist: `modeling/docs/09_SOURCE_CITATION_CHECKLIST.md`
- Dataset-change alignment note: `modeling/docs/12_DATASET_CHANGE_AND_PROPOSAL_ALIGNMENT.md`
- PPT presentation: `presentation/DS_Team7_GSM_Final_Presentation.pptx`
