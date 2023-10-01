#! /bin/bash

tests=`ls -1 tests/*.py`

for i in $tests
do
    python -m unittest "$i"
done
