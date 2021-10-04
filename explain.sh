#!/bin/sh
set -e

rm -rf /tmp/oral-lesions-detection-tmp
python explain.py $@
pdflatex /tmp/oral-lesions-detection-tmp/report.tex
rm -f report.aux report.log