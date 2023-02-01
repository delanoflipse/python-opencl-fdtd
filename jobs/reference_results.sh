#!/bin/sh

# reference
# python cli_run.py -n reference-and-start

# # different schemas
# python cli_run.py -w iwb
# python cli_run.py -w iso
# python cli_run.py -w iso2
# python cli_run.py -w dwm

# # with and without furniture
# python cli_run.py -s real-reference-furniture -b 96

# # oversampling
# python cli_run.py -o 10
# python cli_run.py -o 12
# python cli_run.py -o 14
# python cli_run.py -o 16
# python cli_run.py -o 18
# python cli_run.py -o 20
# python cli_run.py -o 24
# python cli_run.py -o 28
# python cli_run.py -o 32

# # bands
# python cli_run.py -b 3
# python cli_run.py -b 6
# python cli_run.py -b 9
# python cli_run.py -b 12
# python cli_run.py -b 16
# python cli_run.py -b 18
# python cli_run.py -b 20
# python cli_run.py -b 22
# python cli_run.py -b 24

# time
python cli_run.py -t 0.025
python cli_run.py -t 0.05
python cli_run.py -t 0.1
python cli_run.py -t 0.15
python cli_run.py -t 0.2
python cli_run.py -t 0.25
python cli_run.py -t 0.3
python cli_run.py -t 0.4
python cli_run.py -t 0.6
python cli_run.py -t 0.8
python cli_run.py -t 1.0
python cli_run.py -t 1.5
python cli_run.py -t 2.0
