#!/usr/bin/bash

export DLDIR=$( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )/
export PYTHONPATH=$DLDIR:$PYTHONPATH
export PATH=$DLDIR/bin:$PATH
export PATH=$DLDIR/scripts:$PATH


