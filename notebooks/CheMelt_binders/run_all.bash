#!/usr/bin/env bash

for nb in *.ipynb; do
    if [[ "$nb" =~ ^[0-9]+_[A-Za-z]{3}\.ipynb$ ]]; then
        echo "Running $nb"
        jupyter nbconvert --to notebook --execute "$nb"
    fi
done