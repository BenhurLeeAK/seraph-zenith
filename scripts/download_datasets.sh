#!/bin/bash
# Downloads HERIDAL and SARD datasets for SAR human detection training.
# Run from the project root: bash scripts/download_datasets.sh

set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "=============================================="
echo "  HERIDAL Dataset"
echo "=============================================="
echo "HERIDAL is freely downloadable (CC BY 3.0 license), no login required."
echo "Source: IPSAR project, University of Split"
echo ""
echo "MANUAL STEP REQUIRED (the site doesn't support direct scripted download):"
echo "  1. Visit: http://ipsar.fesb.unist.hr/HERIDAL database.html"
echo "  2. Download the full dataset archive (patches + ~500 full-size labeled images)"
echo "  3. Extract into: $DATA_DIR/heridal_raw/"
echo ""
echo "Expected structure after extraction:"
echo "  data/raw/heridal_raw/"
echo "    ├── images/          (full-size 4000x3000 real-world images)"
echo "    └── annotations/     (VOC XML format bounding boxes)"
echo ""

echo "=============================================="
echo "  SARD Dataset (via Kaggle)"
echo "=============================================="
echo "Requires a free Kaggle account + API token."
echo ""
echo "Setup (one-time):"
echo "  1. Go to kaggle.com -> Account -> Create New API Token"
echo "  2. This downloads kaggle.json"
echo "  3. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
echo ""

if command -v kaggle &> /dev/null; then
    echo "Kaggle CLI found. Downloading SARD..."
    kaggle datasets download -d nikolasgegenava/sard-search-and-rescue -p "$DATA_DIR/sard_raw" --unzip
    echo "SARD downloaded to $DATA_DIR/sard_raw"
else
    echo "Kaggle CLI not found. Install with: pip install kaggle"
    echo "Then re-run this script, or download manually from:"
    echo "  https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue"
    echo "  and extract into $DATA_DIR/sard_raw/"
fi

echo ""
echo "=============================================="
echo "  VisDrone (optional, for Stage 1 pretraining)"
echo "=============================================="
echo "Ultralytics ships a ready-to-use auto-download config for VisDrone."
echo "You don't need to manually download it -- the Stage 1 training script"
echo "(train_stage1_visdrone.py) will trigger the Ultralytics auto-download"
echo "on first run via the standard 'VisDrone.yaml' dataset config."
echo ""
echo "Done. Next step: python scripts/convert_heridal.py"
