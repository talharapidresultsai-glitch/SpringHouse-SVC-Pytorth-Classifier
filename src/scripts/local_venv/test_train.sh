#!/bin/bash
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#temp dir for generated config files
WORK_DIR=`mktemp -d -p "$DIR"`

# check if tmp dir was created
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

# deletes the temp directory
function cleanup {      
  rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
  echo 'Done'
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

echo 'Starting virtual env'
source venv.sh

#in the temp folder, create a config file, using paths on current machine
rm -f $WORK_DIR/final_data.yaml $WORK_DIR/temp.yaml
( echo "cat <<EOF >$WORK_DIR/final_data.yaml";
  cat data_config/data_config.yaml;
  echo "EOF";
) >$WORK_DIR/temp.yaml
. $WORK_DIR/temp.yaml
cat $WORK_DIR/final_data.yaml

echo 'Starting training'
python ../../classifier/train_local.py

echo 'Training complete'