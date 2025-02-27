# Test-Time Fine-Tuning of Image Compression Models for Multi-Task Adaptability

This repository contains the source code of our LoRA_comp.

<p align="center">
<img width="500" src="overview.png">
</p>

## Abstract
>The field of computer vision initially focused on human visual perception and has progressively expanded to encompass machine vision, with ongoing advancements in technology expected to drive further expansion. Consequently, image compressors must effectively respond not only to human visual perception but also to the current and future machine vision tasks. Towards this goal, this paper proposes a Test-Time Fine-Tuning (TTFT) approach for adapting Learned Image Compression (LIC) to multiple tasks. A large-scale LIC model, originally trained for human perception, is adapted to both closed-set and open-set machine tasks through TTFT using Singular Value Decomposition based Low Rank Adaptation (SVD-LoRA). The SVD-LoRA layers are applied to both the encoder and decoder of backbone LIC model, with a modified learning scheme employed for the decoder during TTFT to train only the singular values, preventing excessive bitstream overhead. This enables instance-specific optimization for the target task. Experimental results demonstrate that the proposed method effectively adapts the backbone compressor to diverse machine tasks, outperforming competing methods.

## Dataset
The following datasets are used and needed to be downloaded.
- ImageNet1K
- COCO 2017 Train/Val

## Example Usage 
Specify the mode, task_model, data paths, target rate point, corresponding lambda, and checkpoint in the config file accordingly.

### Classification
`python classification_lora.py`<br>

### Object Detection
`python detection_segmentation_lora.py task_model=faster_rcnn`<br>

### Instance Segmentation
`python detection_segmentation_lora.py task_model=mask_rcnn`<br>

## Ackownledgement
Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [TransTIC](https://github.com/NYCU-MAPL/TransTIC). We sincerely appreciate the authors for sharing their code openly.
