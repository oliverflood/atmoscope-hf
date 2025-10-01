---
title: Cloud Classifier
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: Cloud type classification with ConvNeXt and NASA GAZE data
---

# ☁️ Atmoscope on Hugging Face 🤗
[Atmoscope](https://github.com/oliverflood/atmoscope/) is an end-to-end ML project focused on cloud type (genus) classification from phone pictures. This repo is for the live model deployed on Hugging Face!

## Website
[Check the live model here!](https://huggingface.co/spaces/oliverflood/Atmoscope_v1) 

## Extra
This is a deployment of a multi-label cloud classifier trained on the NASA GAZE dataset. The model itself is ConvNeXt with a linear layer, and was fine tuned from ImageNet pretrained weights. Parts of the process are documented on [my blog](https://oliverflood.com/). The remote repo is on [GitHub](https://github.com/oliverflood/atmoscope-hf) and [Hugging Face](https://huggingface.co/spaces/oliverflood/Atmoscope_v1/tree/main). 
