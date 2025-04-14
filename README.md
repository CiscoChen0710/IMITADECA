# DECA + Imitator Setup Guide

This guide outlines how to set up and run the DECA and Imitator pipelines for personalized 3D face mesh animation. Follow the steps below based on your system environment.

---

## 🔧 DECA Setup (Recommended on Ubuntu via WSL for Windows users)

DECA is not well-supported on native Windows environments. It is **highly recommended** to use **WSL (Windows Subsystem for Linux)** and install DECA within Ubuntu.

### 1. Environment Setup
Follow the official DECA GitHub page instructions to install the required dependencies and environment.

### 2. Reconstruct Meshes
Once the environment is set up, navigate to the DECA root directory and run:

```bash
python demos/demo_reconstruct.py -I frames_biden --saveDepth true --saveObj True
```
This will reconstruct meshes from the images in the frames_biden folder, and save depth maps and .obj mesh files.

Imitator Setup (Windows-based Experiment)
1. Set Environment Variables
Before running any script, set the following environment variables in your command prompt:
```bash
set LOGHOME=<Your Path>\Imitator
set HOME=<Your Path>\Imitator
set PYTHONPATH=.
```

Training Procedure
Stage 1: Style Adaptation
Run the following command to start the stage 1 training:
```bash
python main.py -b cfg/style_adaption/biden_stg01_latest.yaml --gpus 0 --train
```
After training, find your checkpoint in:
```bash
Imitator\logs\tb\<latest_version>\checkpoints
```






