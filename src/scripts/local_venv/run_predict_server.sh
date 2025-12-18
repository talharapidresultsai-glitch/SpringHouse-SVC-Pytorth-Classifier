#!/bin/bash
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PARENTDIR="$(dirname "$DIR")"

echo "Starting virtual env"
source venv.sh

echo "Starting predictor web server"
cd "../../../src/classifier"
python predictor.py --port 8080 --model_path "${PARENTDIR}/pretrains/test_5_item.pt"
cd "/scripts/local_venv"

#read -n 1 -s -r -p "Press any key to continue "

echo "Stopped"
