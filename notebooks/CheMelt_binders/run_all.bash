#!/usr/bin/env bash
mkdir -p executed

for nb in *_*.ipynb; do
    case "$nb" in
        *_join_results.ipynb|*_plot_all.ipynb)
            continue
            ;;
    esac

    echo "Running $nb"
    papermill "$nb" "executed/$nb"
done