# Bachelor Thesis: Evaluation of the Effectiveness of Metrics Measuring Shortcut Behavior in Deep Neural Networks
## Table of Contents

- [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Datasets](#datasets)
- [Model Training (optional)](#model-training-optional)
- [Preprocessing](#preprocessing)
- [Bias Mitigation](#bias-mitigation)
- [Evaluation](#evaluation)

## Prerequisites
### Installation

We use Python 3.10.13 and PyTorch 2.2.1. To install the required packages, run:

```bash 
pip install -r requirements.txt
```

### Datasets
Secondly, the datasets need to be downloaded. This includes ISIC2019 & ImageNet & Waterbirds

To download ISIC2019:
```bash
mkdir datasets
cd datasets
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.zip
unzip ISIC_2019_Training_Input.zip
unzip ISIC_2019_Training_GroundTruth.zip
```

To download Waterbirds, we have to download the CUB Dataset and Places Dataset and place them together in one folder to create the Waterbirds dataset. 
ImageNet can be downloaded from: [https://www.image-net.org/](https://www.image-net.org/)

## Model Training

With the required packages installed and the datasets downloaded, the models can be trained. To consolidate training parameters in a unified file, we utilize configuration files (`.yaml`-files). These files specify training hyperparameters, such as architecture, optimizer and learning rate, as well as dataset parameters and output directories. 

We provide scripts to conveniently generate the config files, that can be run as follows (here for ISIC2019):

```bash 
python config_files/training/config_generator_training_isic.py
```

Using the previously generated config files, we can train the models by running:

```bash
python -m model_training.start_training --config_file "config_files/training/isic/your_config_file.yaml"
```


## Preprocessing

All bias identification, annotation and mitigation approaches require latent activations or relevance scores, commonly aggregated via max- or avg-pooling. Therefore, we provide a pre-processing script that pre-computes activations and relevances for all considered layers of the networks for the entire dataset. These pre-computed values can be used for example to compute CAVs or to run SpRAy. The script can be run for a given config-file as follows:

 ```bash
python -m experiments.preprocessing.run_preprocessing --config_file "config_files/your_config_file.yaml"
```

It is also possible to directly compute a "clean" CAV for the next mitigation step with 

 ```bash
python -m experiments.evaluation.get_perfect_cav --config_file "config_files/your_config_file.yaml"
```


## Bias Mitigation
Lastly, we further provide implementations for the bias mitigation step of the Reveal2Revise framework, utilizing annotations generated via our approaches.
We mitigate biases via Right for the Right Reasons (RRR), Right-Reason ClArC (RR-ClArC) and the training free approaches Projective ClArC (P-ClArC) and reactive P-ClArC (rP-ClArC).

Again, we use config-files (`.yaml`) to specify hyperparameters, such as model details, bias mitigation approach, and mitigation parameters, such as &lambda;-values.
These files are located in in `config_files/bias_mitigation_controlled/` and can be generated as follows:

```bash
python -m config_files.bias_mitigation_controlled.config_generator_mitigation_hyper_kvasir_attacked"
```

Having generated the config-files, the bias mitigation step can be performed as follows:

```bash
python -m experiments.mitigation_experiments.start_model_correction --config_file "config_files/bias_mitigation_controlled/hyper_kvasir_attacked/your_config_file.yaml"
```

## Evaluation
Lastly, having mitigated the biases, we (re-)evaluate the models. Here are some example scripts, with further script in ``experiments/evaluation``

```bash
CONFIG_FILE="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/your_config_file.yaml"
# 1) Evaluate on different subsets (train/val/test) in different settings (clean/attacked) and calculate performance metrics, per-class metrics and per-group metrics
python -m experiments.evaluation.evaluate_by_subset_attacked --config_file $CONFIG_FILE

# 2) Measure relevance on artifact region in input space
python -m experiments.evaluation.compute_artifact_relevance --config_file $CONFIG_FILE

# 3) Measure TCAV score wrt bias concept, before you should also run get_perfect_cav to get a clean CAV 
python -m experiments.evaluation.measure_tcav --config_file $CONFIG_FILE

# 4) Compute heatmaps for biased samples for original and corrected models
python -m experiments.evaluation.qualitative.plot_heatmaps --config_file $CONFIG_FILE

# 5) Same as (1) but with class and group balancing
python -m experiments.evaluation.compute_balanced_metrics --config_file $CONFIG_FILE

# 6) RCS Score
python -m experiments.evaluation.compute_rcs_score --config_file $CONFIG_FILE

# 7) Compute MI Based Metrics and Alignment Loss
python -m experiments.evaluation.compute_mi --config_file $CONFIG_FILE
```

We can use the config files in ``config_files/evaluate_metrics_clean`` to evaluate metrics without further training and/or mitigation



Code is based on [Medical-AI-Safety by Pahde et al.](https://github.com/frederikpahde/medical-ai-safety/) and [Reveal to Revise Framework by Pahde et al.](https://github.com/maxdreyer/Reveal2Revise)
