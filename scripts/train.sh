#!/bin/bash
# Launches distributed PyTorch training
torchrun --nproc_per_node=8 python/dlcm/training/trainer.py
