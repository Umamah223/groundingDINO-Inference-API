# GroundingDINO-Inference-API

## Overview
This repository provides a FastAPI-based inference API for the [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) model. It allows users to:

- Provide text prompts and upload images
- Run object detection with GroundingDINO
- Receive bounding box predictions for objects detected in the uploaded images via a FastAPI endpoint

This project is built upon the initial [GroundingDINO model](https://github.com/IDEA-Research/GroundingDINO) and has been adapted to offer easy API access for object detection.

## Features
- **FastAPI**: Fast and efficient inference API for object detection.
- **Gradio**: A simple web interface for uploading images and interacting with the model.
- **Bounding Box Predictions**: The model will return bounding boxes around detected objects with labels.

## Setup & Installation

### Requirements
- Python 3.8+
- FastAPI
- Gradio
- Torch
- GroundingDINO model dependencies

### Install Dependencies
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Umamah223/groundingDINO-Inference-API.git
cd groundingDINO-Inference-API
pip install -r requirements.txt

