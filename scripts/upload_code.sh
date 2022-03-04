#!/usr/bin/env bash
SHELL_FOLDER=$(dirname $(dirname $(readlink -f "$0")))
TARGET_FOLDER=/home/docker/codes/hash/methods/bit-select
if [[ $1 == '' ]];then
  param="$SHELL_FOLDER gpu5:$TARGET_FOLDER/"
elif [[ $1 == 1 ]];then
  param="$2 $SHELL_FOLDER"
elif [[ $1 == 2 ]];then
  param="$SHELL_FOLDER $2"
fi
echo $param
rsync -avm -e ssh --exclude='results/' --exclude='data/' --include={'*/','*.yaml','*.py','*.zip','*.sh'} --exclude='*' $param