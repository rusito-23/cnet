#!/usr/local/bin/gnuplot --persist -c
# USAGE:
# ./metrics.plt path/to/data.dat


set title "Loss";
set terminal qt 0;

plot ARG1 using 0:1 with lines notitle;

set title "Accuracy";
set terminal qt 1;

plot ARG1 using 0:2 with lines notitle;
