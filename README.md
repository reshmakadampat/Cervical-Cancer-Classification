\# 🔬 Explainable AI for Automated Detection of Cervical Cancer



A comparative deep learning study using \*\*ResNet18\*\*, \*\*EfficientNet-B0\*\*, and \*\*Vision Transformer (ViT-B/16)\*\* with integrated XAI techniques — \*\*Grad-CAM\*\*, \*\*SHAP\*\*, and \*\*Attention Rollout\*\* — on the Herlev Pap Smear Dataset.



> \*\*MSc Dissertation | Amity University Bengaluru | 2026\*\*



\---



\## 📌 Overview



This project addresses a critical gap in medical AI: most deep learning models for cervical cancer detection focus solely on accuracy, ignoring clinical interpretability. This framework evaluates three architecturally diverse models alongside their respective explainability techniques, making predictions transparent enough for real-world clinical use.



\*\*Core idea:\*\* Accuracy alone is not enough in medical diagnosis. Doctors need to understand \*why\* the model made a prediction.



\---



\## 📊 Results Summary



| Model | Test Accuracy | Macro F1 | Weighted F1 | XAI Method |

|---|---|---|---|---|

| ResNet18 | \*\*86.50%\*\* | \*\*0.884\*\* | 0.866 | Grad-CAM |

| ViT-B/16 | \*\*86.69%\*\* | 0.882 | \*\*0.868\*\* | Attention Rollout |

| EfficientNet-B0 | 65.78% | 0.681 | 0.652 | SHAP (GradientExplainer) |



\---



\## 🗂️ Repository Structure



```

├── Explainable\_AI\_for\_Automated\_Detection\_of\_Cervical\_Cancer\_RestNet18.ipynb

├── Explainable\_AI\_for\_Automated\_Detection\_of\_Cervical\_Cancer\_EfficientNet.ipynb

├── Explainable\_AI\_for\_Automated\_Detection\_of\_Cervical\_Cancer\_VIT.ipynb

└── README.md

```



\---



\## 🧠 Models and XAI Techniques



\### ResNet18 → Grad-CAM

\- 18-layer residual CNN pretrained on ImageNet

\- Fine-tuned: `layer4` + classification head unfrozen

\- Grad-CAM generates heatmaps showing which regions of the cell influenced the prediction

\- Faithfulness metric used to quantitatively validate explanation quality



\### EfficientNet-B0 → SHAP (GradientExplainer)

\- Compound-scaled CNN with MBConv blocks and squeeze-and-excitation attention

\- Fine-tuned: last 3 feature blocks + classifier unfrozen

\- SHAP provides signed pixel-level attribution maps grounded in cooperative game theory

\- Model-agnostic — applicable across all architectures



\### Vision Transformer (ViT-B/16) → Attention Rollout

\- Processes 224×224 images as 196 non-overlapping 16×16 patches

\- Fine-tuned: last 2 encoder layers + classification head unfrozen

\- Attention Rollout propagates attention across all 12 encoder layers

\- Reveals global spatial focus of the transformer on the input image



\---



\## 📁 Dataset



\*\*Herlev Pap Smear Dataset\*\* — \[Download here](https://mde-lab.aegean.gr/index.php/downloads/)



| Class | Category | Samples |

|---|---|---|

| Superficial Squamous | Normal | 74 |

| Intermediate Squamous | Normal | 70 |

| Columnar Epithelial | Normal | 98 |

| Mild Dysplasia | Abnormal | 182 |

| Moderate Dysplasia | Abnormal | 146 |

| Severe Dysplasia | Abnormal | 197 |

| Carcinoma in Situ | Abnormal | 150 |

| \*\*Total\*\* | | \*\*917\*\* |



\- Split: \*\*70% train / 15% val / 15% test\*\* (stratified, SEED=42)

\- Normal classes: \~26% | Abnormal classes: \~74%



\---



\## ⚙️ Setup and Installation



\### Prerequisites

\- Python 3.10+

\- CUDA-enabled GPU (recommended) or Google Colab



\### Install Dependencies



```bash

pip install torch torchvision

pip install grad-cam

pip install shap

pip install scikit-learn matplotlib numpy opencv-python

```



\### Running on Google Colab (Recommended)



1\. Upload the `.ipynb` file to Google Colab

2\. Mount Google Drive and set the dataset path:

```python

from google.colab import drive

drive.mount('/content/drive')

SPLIT\_ROOT = '/content/drive/MyDrive/smear\_split'

```

3\. Run all cells in order



\---



\## 🔁 Reproducibility



All experiments use a fixed global seed across every random module:



```python

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual\_seed(SEED)

torch.cuda.manual\_seed\_all(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

```



\---



\## 🧪 Training Configuration



| Parameter | ResNet18 | EfficientNet-B0 | ViT-B/16 |

|---|---|---|---|

| Pretrained | ImageNet | ImageNet | ImageNet |

| Optimizer | Adam | Adam | Adam |

| Learning Rate | 0.001 | 5×10⁻⁵ | 1×10⁻⁵ |

| Epochs | 10 | 20 | 20 |

| Batch Size | 32 | 32 | 32 |

| Loss | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |



\---



\## 🌡️ XAI Implementation Notes



\*\*Grad-CAM\*\* (ResNet18)

```python

from pytorch\_grad\_cam import GradCAM

target\_layers = \[model.layer4\[-1]]

cam = GradCAM(model=model, target\_layers=target\_layers)

```



\*\*SHAP GradientExplainer\*\* (EfficientNet-B0)

```python

import shap

explainer = shap.GradientExplainer(model, background)  # 20 background images

shap\_values = explainer.shap\_values(test\_images)

```



\*\*Attention Rollout\*\* (ViT-B/16)

```python

\# Forward hooks capture attention from all 12 encoder layers

\# Rollout = recursive matrix product with residual: 0.5\*A + 0.5\*I

\# CLS token row → reshape (14,14) → upsample (224,224)

```



\---



\## 📈 Evaluation Metrics



Each model is evaluated on:

\- Accuracy, Precision, Recall, F1-Score (per-class + macro + weighted)

\- Confusion Matrix

\- ROC Curves with per-class AUC

\- Classification Report (exported to `.csv`)

\- \*\*Faithfulness Drop\*\* (Grad-CAM specific): confidence drop when top-50% salient region is masked



\---



\## 💡 Key Findings



\- ResNet18 and ViT-B/16 achieve nearly identical accuracy (\~86.5%) despite a 7× parameter difference — ResNet18 is far more efficient for small datasets

\- EfficientNet-B0 underperforms (65.78%), suggesting compound scaling requires larger datasets to generalise

\- Grad-CAM consistently highlights the \*\*nuclear region\*\* — clinically valid, as nuclear morphology is the primary diagnostic criterion in Pap smear analysis

\- Attention Rollout is \*\*class-agnostic\*\* — useful for auditing transformer spatial focus but less suitable for per-prediction clinical explanation

\- SHAP's \*\*signed attribution\*\* maps reveal both supporting and suppressing features per class



\---



| Library | Purpose |

|---|---|

| `torch`, `torchvision` | Model training and transforms |

| `pytorch-grad-cam` | Grad-CAM explainability |

| `shap` | SHAP explainability |

| `scikit-learn` | Metrics, confusion matrix, ROC |

| `matplotlib`, `opencv-python` | Visualisation |

| `numpy` | Array operations |



\---



\## 📄 Citation



If you use this work, please cite:



```

@mastersthesis{ReshmaK2026,

&#x20; author    = {Reshma K},

&#x20; title     = {Explainable AI for Automated Detection of Cervical Cancer: 

&#x20;              A Comparative Study of ResNet18, EfficientNet-B0, and Vision 

&#x20;              Transformer with Grad-CAM, SHAP, and Attention Rollout on 

&#x20;              the Herlev Dataset},

&#x20; school    = {Amity University Bengaluru},

&#x20; year      = {2026},

&#x20; type      = {MSc Dissertation},

&#x20; note      = {Unpublished}

}

Reshma. K, "Explainable AI for Automated Detection of Cervical Cancer:

A Comparative Study of ResNet18, EfficientNet-B0, and Vision Transformer

with Grad-CAM, SHAP, and Attention Rollout on the Herlev Dataset,"

MSc Dissertation, Amity University Bengaluru, 2025.

```



\---



\## 📬 Contact



\*\*Reshma K\*\*  

MSc Data Science Student | Amity University Bengaluru  

📧 \[abruti787883@gmail.com]  

🔗 [LinkedIn](https://www.linkedin.com/in/reshma-k-8017452a6/?isSelfProfile=false) | [GitHub](https://github.com/reshmakadampat)



\---



\## ⚠️ Disclaimer



This project is for academic research purposes only. The models and XAI outputs presented here are \*\*not validated for clinical use\*\* and should not be used for medical diagnosis without proper clinical validation and regulatory approval.



\---



\*Built with ❤️ for interpretable, clinically trustworthy AI in cervical cancer screening.\*

