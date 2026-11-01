#!/usr/bin/env bash
# usage: ./push.sh https://github.com/<you>/<repo>.git
set -e
git init -b main
git add .
git commit -m "Initial commit: Seraph Zenith human detection eval, docs, deck"
git remote add origin "$1"
git push -u origin main
