#!/bin/sh

python exaplin.py $@
pdflatex /tmp/build_diagnosis/report.tex
rm -f report.aux report.log