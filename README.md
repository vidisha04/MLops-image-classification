# MLops-image-classification

## Project Title:
Real-Time Human Detection System with SMS Alarm Notifications

## Goal:
We aim to build a real-time detection system that monitors a video feed and identifies whether a human or an animal enters the frame. If a human is detected, the system will send an SMS alarm within one minute.

## Features:
Video Input: Continuously capture video from the laptop's camera using OpenCV.
Human Detection: Process video frames with a pretrained model from TIMM, wrapped in PyTorch Lightning for modularity and scalability.
SMS Alarm: Trigger an SMS alert using Twilio API when a human is detected in the frame.
CLI Interface: Start/stop the system and configure phone numbers via a simple command-line interface.
