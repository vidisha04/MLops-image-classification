# Real-time Species Detection and Classification System documentation!

## Description

We aim to build an automated system for real-time detection and classification of animal and plant species using Deep Learning and ML-Ops tools.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://species-detection/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://species-detection/data/` to `data/`.


