#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

jupyter nbconvert --to notebook --execute $SCRIPT_DIR/training.ipynb
