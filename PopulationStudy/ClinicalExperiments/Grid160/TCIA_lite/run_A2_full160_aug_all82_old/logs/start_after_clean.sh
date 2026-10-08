#!/bin/bash
# Two whole-volume lite runs at once exceed RAM (page cache ~240 GB each) and get oom-killed -> run after lite-clean78.
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet lite-clean78; do sleep 60; done
echo "$(date) lite-clean78 finished, starting old-loss all-82" >> logs/launcher.log
/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python -u scripts/train_all82_old.py --gpu 0 > logs/train.log 2>&1
echo "$(date) exited $?" >> logs/launcher.log
