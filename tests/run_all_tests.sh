#! /bin/bash
set -e

function_tests=`ls -1 test_*.py`

#echo "Testing the following files:"
#echo $FILES
for i in $function_tests
do
    bname=`basename $i`
    echo "Running $bname..."
    python "$i"
done
