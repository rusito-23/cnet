#!/usr/local/bin/gnuplot --persist -c


set title "Loss vs Accuracy";

plot for [col=1:2] ARG1 using 0:col with lines notitle 
