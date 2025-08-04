
# 🧠 MLLM for Postmenstrual Age Prediction (dHCP)

This repository contains code and workflows for **fine-tuning Qwen2.5-VL (7B)** to **predict postmenstrual age (in days or weeks)** based on **neonatal brain MRI cortical feature projections**, using data from the [developing Human Connectome Project (dHCP)](http://www.developingconnectome.org/project/).

## 🔬 Task Description

We aim to **predict postmenstrual age at the time of scan** from four 2D projected cortical feature maps extracted from neonatal brain MRIs:

1. **Cortical Thickness**
2. **Cortical Curvature**
3. **Cortical Myelination**
4. **Sulcal Depth**

These maps are stacked into 4-channel inputs and provided to a multi-modal LLM (Qwen2.5-VL) for regression.

---

## 📊 Dataset

- **Source**: Developing Human Connectome Project (dHCP)
- **Inputs**: 2D cortical feature maps (240×320×4, `.npy` format)
- **Labels**: Postmenstrual Age at scan (in days or weeks)
- **Metadata**: Provided via a `meta_2022_pma.pkl` DataFrame

---

## 🧩 Model

- Base model: [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Fine-tuning via: **LoRA (PEFT)** on selected vision + language layers
- Fine-tuning strategy:
  - Multi-image input using vision-aware prompt template
  - Target: numerical regression output (e.g. "269.5")

---

## 🧪 Evaluation Metrics

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **95% Confidence Intervals** via Bootstrap Sampling

Also includes **model explanation output** (reasoning for age estimation).

---

## 🚀 How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision transformers accelerate peft bitsandbytes sentencepiece tiktoken scikit-image imageio SimpleITK seaborn qwen-vl-utils[decord]
```

### 2. Run Training

```python
python dhcp_mllm_qwen2_vl_multiimage_github.py
```

> Model will automatically use LoRA to fine-tune the vision-language layers on your dHCP data.

### 3. Inference & Visualization

- Supports **both days and weeks** output unit
- Generates explanations with structured output:
  ```
  Predicted Gestational Age: 269.5
  Explanation: [model-generated rationale]
  ```

---


## 🧠 Example Prompt for Inference

```python
"You are provided with four 2D projection maps from a neonatal brain MRI scan. 
These represent, in order: Image 1: cortical thickness, Image 2: cortical curvature, 
Image 3: cortical myelination, and Image 4: sulcal depth. 
Based on these four images, please first predict the gestational age at the time of the scan (in days or weeks)."
```

---

## 📁 Project Structure

```
├── dhcp_mllm_qwen2_vl_multiimage_github.py  # Main training + inference script
├── /data
│   ├── 2D_projection_L_sub-XXX.npy          # Cortical feature projections
│   └── meta_2022_pma.pkl                    # Labels and metadata
└── requirements.txt                         # Optional: dependencies
```

---


