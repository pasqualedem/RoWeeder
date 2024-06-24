#!/bin/bash
conda init
conda activate SSLWeedMap
python main.py experiment $@ --parallel
