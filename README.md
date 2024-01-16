# Simple ASR
## Overview
This is an implementation of an LSTM Seq2Seq model that predicts characters from waveforms. It was completed as part of the 10-618 course at CMU. 
The model is trained on a subset of TIMIT dataset and different training algorithms: MLE, Scheduled Sampling, and OCD are tested.

## Setting up the environment
1. Create a Virtual Environment with `venv` or a Conda Environment:
   ```
   conda create --name myenv python=3.10 pip
   conda activate myenv
   ```
2. Run the following command to install dependencies from requirements.txt:
   ```
   pip install -r requirements.txt
   ```

## Usage
To select the desired training mode, use the following command line arguments:
- **Maximum likelihood estimation (MLE)**:  `--mode mle`
  
  The ground truth is fed into the decoder and we minimize cross entropy of the ground truth.
- **Scheduled Sampling (SS)**: `--mode ss` with desired `--beta` or `--mode ss_linear` for SS with beta on a linear decay schedule

  The ground truth is fed with probability _beta_ or the model prediction with probability _1 - beta_. 
- Optimal Completion Distillation (OCD) [DAgger training]: `--mode ocd` with desired `--beta` or `--mode ocd_linear` for OCD with beta on a linear decay schedule

  A dynamic oracle completion is fed in with probability _beta_ or the model prediction with probability _1 - beta_.

