#!/bin/bash
./submit.sh roofline_run_0.sh
./submit.sh roofline_run_1.sh
./submit.sh roofline_run_2.sh
./submit.sh roofline_run_3.sh
./submit.sh roofline_run_4.sh

# Submit separately (see comment above) -- do not uncomment into the same burst:
# ./submit.sh roofline_staged.sh
