#!/bin/bash
set -x

TEST_FILE="3"
DEVICE_TYPE="dgpu"

{
    LOCAL2=(4 8 16 32 64 128 256 512 1024)
    # realization 2
    run_command="./test.exe --input test_data/in${TEST_FILE}.txt --output out${TEST_FILE}.txt --device-type ${DEVICE_TYPE} --device-index 0"
    
    for local2 in "${LOCAL2[@]}"; do

        compile_command="clang *.c -O3 -lOpenCL -Wall -Wextra -o test.exe -DLOCAL2=$local2"
        $compile_command 2>/dev/null

        if [ $? -eq 0 ]; then
            $run_command > .tmp
            if [ $? -eq 0 ]; then
                echo "Realization 2: $local2"
                cat .tmp
            fi
        fi

        echo
    done
}
