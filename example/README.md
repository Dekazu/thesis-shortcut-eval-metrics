## Config Files

We use config files specifying all required parameters and information for an experiment, including dataset paths, model architecture, checkpoint path, and others. Config file generators, for example for all "revealing"-experiments, are provided in 
`config_files/revealing/config_generator_isic.py`.
Example scripts for ISIC2019 experiments with VGG16 and ResNet50d are provided in `example/configs/`. Make sure to update the specified paths.

Model checkpoints can be downloaded [here](https://datacloud.hhi.fraunhofer.de/s/nkPR6HXbpSCSYQd).

## CRP Preprocessing

First, we have to run the preprocessing script. Given a config file specifying dataset and model, this script computes ActMax, RelMax and receptive field info for each neuron and stores them into the specified directory. The script can be run as follows:

```bash
python -m example.crp_run_preprocessing  --config_file "example/configs/resnet50d_last_conv.yaml"
```

## CRP Explanation for given prediction

Now, we can pass test samples through the model and generate CRP explanations for the given prediction as follows:

```bash
python -m example.run_crp_given_sample  --config_file "example/configs/resnet50d_last_conv.yaml" --sample_ids "531,547,527,541"
```