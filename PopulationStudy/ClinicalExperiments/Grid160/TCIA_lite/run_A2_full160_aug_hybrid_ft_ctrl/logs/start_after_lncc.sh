#!/bin/bash
cd "$(dirname "$0")/.."
while kill -0 3569083 2>/dev/null; do sleep 60; done
echo "$(date) lncc done, starting ctrl" >> logs/launcher.log
/home/abhishek/Documents/LEARN-GUI/LEARN-GUI-Python/.venv/bin/python -u scripts/train_ft.py --gpu 1 > logs/train.log 2>&1
echo "$(date) ctrl exited $?" >> logs/launcher.log
