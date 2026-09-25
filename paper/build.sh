#!/bin/sh
set -e
cd "$(dirname "$0")"
export TEXINPUTS="./acmart:.:$TEXINPUTS"
export BSTINPUTS="./acmart:.:$BSTINPUTS"
pdflatex -interaction=nonstopmode article.tex
bibtex article
pdflatex -interaction=nonstopmode article.tex
pdflatex -interaction=nonstopmode article.tex
echo "PDF: $(pwd)/article.pdf"
