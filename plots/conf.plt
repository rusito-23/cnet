#!/usr/local/bin/gnuplot --persist -c
# USAGE:
# ./conf.plt path/to/data.dat
# Expects a 10 x 10 Confusion Matrix

FILE = ARG1

set title "Confusion Matrix" 
set cblabel "Score"
unset key

set autoscale yfix
set autoscale xfix

set xlabel "Predicted"
set xrange [ -0.500000 : 9.50000 ] noreverse nowriteback
set x2range [ * : * ] noreverse writeback
set xtics border in scale 0,0 mirror norotate  autojustify
set xtics (\
    "0" 9.0,\
    "1" 8.0,\
    "2" 7.0,\
    "3" 6.0,\
    "4" 5.0,\
    "5" 4.0,\
    "6" 3.0,\
    "7" 2.0,\
    "8" 1.0,\
    "9" 0.0)


set ylabel "Real"
set yrange [ -0.500000 : 9.50000 ] noreverse nowriteback
set y2range [ * : * ] noreverse writeback
set ytics border in scale 0,0 mirror norotate  autojustify
set ytics (\
    "0" 9.0,\
    "1" 8.0,\
    "2" 7.0,\
    "3" 6.0,\
    "4" 5.0,\
    "5" 4.0,\
    "6" 3.0,\
    "7" 2.0,\
    "8" 1.0,\
    "9" 0.0)

set pm3d map

splot FILE matrix with image
