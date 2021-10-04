#!/bin/sh
set -e

rm -rf /tmp/oral-lesions-detection-tmp
python build-diagnosis.py $@
python build-report.py --diagnosis /tmp/oral-lesions-detection-tmp/diagnosis.json
pdflatex report.tex
rm -f report.tex report.aux report.log