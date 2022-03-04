#!/usr/bin/env bash
SHELL_FOLDER=$(dirname $(dirname $(readlink -f "$0")))
echo $SHELL_FOLDER
echo $1
echo $2
rsync -avm -e ssh --include={'*/','*.json','*.csv','*.txt'} --exclude='*' $1/ $SHELL_FOLDER/results/$2