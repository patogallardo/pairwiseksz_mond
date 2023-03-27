#!/usr/bin/bash
python fit_with_cov.py L43_150

git add --all .
git commit -m "refreshed plots"
git push origin main
