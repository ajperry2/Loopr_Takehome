# Loopr_Takehome
Classification of Carbon Fiber Defects

### Work In Progress


I typed up the code for this ML pipeline tonight, but I still need to go back and perform these tasks:
- Modularize training code
- Remove code smells
- Write documentation/report
- Deploy Model

### Current Arch.
- Transfered learning on larger class sets Unet Encoder -> MLP 
- Finetune layers for encoding separation with contrastive learning

Completed:
- My Architecture is implemented
- It has ... results

### Training Pipeline


- https://www.kaggle.com/code/alanjacobperry/steel-defect-segmentation-unet/edit


### Create Environment
---

Make environment
- `uv venv looper-takehome`
- `source looper-takehome/bin/activate`

Install Dependencies (Consideration: You may need to edit pyproject.toml to use your GPU driver):
- `uv sync`