#!/bin/bash


FOLDER=$1
OUTPUT=$2

list_items () {
    ls $FOLDER/*.csv
}

handle_files () {
    read f
    cp $f $OUTPUT

    xargs sed -s '1'd >>$OUTPUT
}

list_items | handle_files
