# DECA + Imitator Setup Guide

This guide outlines how to set up and run the DECA and Imitator pipelines for personalized 3D face mesh animation. Follow the steps below based on your system environment.


---

## 🔧 DECA Setup (Recommended on Ubuntu via WSL for Windows users)

DECA is not well-supported on native Windows environments. It is **highly recommended** to use **WSL (Windows Subsystem for Linux)** and install DECA within Ubuntu.

### 1. Environment Setup
Follow the official DECA GitHub page instructions to install the required dependencies and environment.
DECA: https://github.com/yfeng95/DECA

### 2. Reconstruct Meshes
Once the environment is set up, navigate to the DECA root directory and run:

```bash
python demos/demo_reconstruct.py -I frames_biden --saveDepth true --saveObj True
```
This will reconstruct meshes from the images in the frames_biden folder, and save depth maps and .obj mesh files.

## Imitator Setup (Windows-based Experiment)
1. Environment Setup
Follow the official Imitator GitHub page instructions to install the required dependencies and environment.
Imitator:https://github.com/bala1144/Imitator

2. Set Environment Variables
Before running any script, set the following environment variables in your command prompt:
```bash
set LOGHOME=<Your Path>\Imitator
set HOME=<Your Path>\Imitator
set PYTHONPATH=.
```

### Training Procedure
#### Stage 1: Style Adaptation
Run the following command to start the stage 1 training:
```bash
python main.py -b cfg/style_adaption/biden_stg01_latest.yaml --gpus 0 --train
```
After training, find your checkpoint in:
```bash
Imitator\logs\tb\<latest_version>\checkpoints
```

#### Stage 2: Decoder Fine-tuning
Modify the YAML config biden_stg02_latest.yaml, and set init_from_ckpt to the path of your Stage 1 checkpoint, for example:
```bash
init_from_ckpt: logs/tb/version_18/checkpoints/epoch=49-step=900.ckpt
```
Then run Stage 2 training:
```bash
python main.py -b cfg/style_adaption/biden_stg02_latest.yaml --gpus 0 --train
```
The trained stylized model will be saved under:
```bash
logs/tb/<latest_version>/checkpoints
```

#### Inference and Testing
Run the following command to generate predictions and rendered results:
```bash
python imitator/test/test_model_external_audio.py \
  -m logs/tb/<Your Version> \
  -a personalized_Ebiden/audio_split_5s/biden_00001.wav \
  -t biden \
  -c 0 \
  -r \
  -d \
  --template_obj personalized_Ebiden/template_latest_v1.obj
```
Argument Descriptions:
-a: Path to the audio file

-t: Subject identity (e.g., "biden")

-c: Condition ID (0–7) from VOCA used for testing

-r: Render the results as videos

-d: Dump the prediction as .npy files

--template_obj: Use a specific .obj file as the base mesh (usually the one used during training)





