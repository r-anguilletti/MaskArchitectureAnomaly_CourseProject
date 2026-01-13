# EMAS: Energy-Regularized Mask-Based Anomaly Segmentation

This repository contains the implementation of **EMAS** (implemented starting from EoMT architecture) for Anomaly Detection and Semantic Segmentation.

## Project Structure

* `models/`: Contains the core architecture definitions. Specifically, `segmenter.py` implements the standard architecture, while `segmenter_HN.py` is dedicated to the Hard Negative mining configuration.
* `eval.py`: Computes anomaly detection metrics (AUPRC, FPR@95TPR) by comparing model predictions against ground truth masks. It supports multiple anomaly scoring methods, including MSP, MaxLogit, MaxEntropy and RbA.
* `eval_miou.py`: Evaluates the semantic segmentation performance on the Cityscapes dataset by calculating the mean Intersection over Union (mIoU). It loads a trained checkpoint and computes the metric using torchmetrics, ensuring void classes are ignored.
* `evalTemp.py`: Performs anomaly detection evaluation by applying Temperature Scaling to the model's logits. It iterates through various temperature values to find the optimal setting that maximizes metrics like AuPRC.
* `datasets/`: Directory to manage the creation of datasets. Includes `hybrid_anomaly.py`, which implements a custom dataset to generate synthetic anomaly training samples by overlaying COCO objects onto Cityscapes scenes via a "Cut & Paste" strategy.
* `cnp_dataset.py`: Script that orchestrates the creation of the custom "Cut & Paste" dataset by leveraging the `HybridAnomalyDataset` class (defined in `hybrid_anomaly.py`) to generate synthetic anomaly samples and saving them to disk..
* `cnp_zip_dataset.py`: Implements a specialized data loader designed to efficiently read the generated Cut & Paste dataset directly from the compressed zip archives during training..
* `train.py`: Main entry point for training the model.
* `train_HN.py`: Script specific for Hard Negative mining training.
* `train.ipynb`: Jupyter notebook designed for training and inference.
* `eval.ipynb` : Jupyter notebook designed for evaluation.

---
## Usage

The project workflow is streamlined through two designated Jupyter Notebooks, optimized for Google Colab but adaptable for local environments.

### 1. Training Pipeline (`train.ipynb`)
This notebook handles the entire setup and training phase:
* **Environment Setup**: Installs all dependencies (`lightning`, `torch`, `ood-metrics`, etc.) and mounts Google Drive.
* **Dataset Preparation**:
    * Generates the **Cut & Paste (CnP)** dataset if missing.
    * Copies and unzips **Cityscapes** to the local SSD for high-performance I/O.
* **Training Execution**:
    * Launches the standard training (`train.py`).
    * Alternatively, launches the Hard-Negative mining training (`train_HN.py`).

### 2. Evaluation Pipeline (`eval.ipynb`)
This notebook is dedicated to validating the model's performance:
* **Semantic Segmentation**: Calculates **mIoU** on the Cityscapes validation set using `eval_miou.py`.
* **Anomaly Detection**: Runs `eval.py` to compute metrics (AuPRC, FPR@95) on OOD datasets (e.g., RoadAnomaly, RoadObstacle, LostAndFound).
    * *Supported methods:* `msp`, `maxlogit`, `maxentropy`, `rba`.
* **Temperature Scaling**: Uses `evalTemp.py` to find the optimal temperature $T$ for maximizing anomaly separation.

> **Note:** Inside `eval.ipynb`, remember to update the `--ckpt` paths and select the appropriate dataset/image format (`.png`, `.jpg`, `.webp`) for the specific test you want to run.
---
