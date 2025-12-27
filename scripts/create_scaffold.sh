#!/usr/bin/env bash
mkdir -p data/neu/train data/neu/test data/mvtec
mkdir -p notebooks src/data src/features src/models src/evaluation src/visualization configs reports/figures assets/screenshots models
touch notebooks/.gitkeep configs/.gitkeep reports/figures/.gitkeep assets/screenshots/.gitkeep src/data/.gitkeep
echo "Scaffold created"
