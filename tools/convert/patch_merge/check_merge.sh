#!/bin/bash
# set -e

ROOT_DIR="/path/to/MindSpeed-Core-MS/"
CONVERT_TOOL_DIR="/path/to/patch_merge/"
PATCH_JSON_PATH=${CONVERT_TOOL_DIR}"merge_patch_info.json" #"/path/to/merge_patch_info.json"
CUT_PY=${CONVERT_TOOL_DIR}"modules/cut_patch.py"
MERGE_SH=${CONVERT_TOOL_DIR}"run_merge.sh"
CHECK_PY=${CONVERT_TOOL_DIR}"modules/coverage.py"
MODEL_SHELL="mtp_tune_deepseek3_671b_4k_full_ptd_cann_8p_test.sh"
LOSS_PATTERN="lm loss: 11.5952320098876953 | mtp_1 loss: 11.8836364746093750 | loss scale: 1.0 | grad norm: 12.7480366728578041"

echo_error() {
    echo -e "\033[31m$1\033[0m"
}

echo_success() {
    echo -e "\033[32m$1\033[0m"
}

# process patch merging
do_merge() {
    local work_dir=$1
    local local_json_path=$2
    local log_file=$work_dir"merge.log"

    echo "Process patch merging in $work_dir, with patches $local_json_path"

    bash $MERGE_SH $work_dir $local_json_path

    grep -C 3 "bad handled cases" $log_file

    pattern='ERROR|Traceback'
    if [ ! -f "$log_file" ]; then
        echo "...File not found: $log_file" >&2
        return 1
    fi

    grep -Eq "$pattern" "$log_file"
    if [ $? -eq 0 ]; then
        echo "...Merge failed, check log: $log_file"
        return 1
    else
        echo "...Merge success" 
        return 0
    fi

}

# Run model shell testing
do_test() {
    local work_dir=$1

    echo "...in do test, work_dir ${work_dir}"

    export PYTHONPATH=$work_dir"MindSpeed-LLM":$PYTHONPATH
    export PYTHONPATH=$work_dir"MindSpeed":$PYTHONPATH
    export PYTHONPATH=$work_dir"Megatron-LM":$PYTHONPATH
    export PYTHONPATH=$work_dir"transformers/src":$PYTHONPATH
    export PYTHONPATH=$work_dir"MSAdapter/mindtorch/":$PYTHONPATH

    cd $work_dir/"MindSpeed-LLM"

    if [ ! -f $MODEL_SHELL ]; then
        echo_error "File $MODEL_SHELL not Found, exit..."
        exit 1
    fi

    bash $MODEL_SHELL>${work_dir}test.log 2>&1 &
    wait
    echo "...finish run $MODEL_SHELL"
}

check_log() {
    work_dir=$1
    json_file=$2

    log_file=${work_dir}"test.log"

    if [ ! -f "$log_file" ]; then
        echo "...File not found: $log_file" >&2
        return 1
    fi
    echo "...checking log $log_file"

    grep -Fq "$LOSS_PATTERN" $log_file
    if [ $? -eq 0 ]; then
        echo "...grep success, $LOSS_PATTERN in $log_file, start checking coverage"
        python $CHECK_PY --json-file $json_file --log-file $log_file
        return 0 
    else
        echo "...grep failed, $LOSS_PATTERN not in $log_file"
        return 1
    fi
}


# Binary search debugging for error cases
binary_search() {
    local patch_json_path=$1
    local json_name=$(basename "$patch_json_path")

    # Preparation workspace
    local workspace_dir=$ROOT_DIR/"test_patch_merge_binsearch/" #$(date +"%Y%m%d%H_%M%S")"

    if [ -d $workspace_dir ]; then
        rm -rf $workspace_dir
    fi
    mkdir -p $workspace_dir
    echo "...making dir $workspace_dir..."
    cd $workspace_dir

    local raw_case_dir=$workspace_dir"raw/"
    mkdir -p $raw_case_dir
    echo "...making dir $raw_case_dir..."

    cp -r $ROOT_DIR"MindSpeed-LLM" $raw_case_dir
    cp -r $ROOT_DIR"MindSpeed" $raw_case_dir
    cp -r $ROOT_DIR"Megatron-LM" $raw_case_dir
    cp -r $ROOT_DIR"transformers" $raw_case_dir
    cp -r $ROOT_DIR"MSAdapter" $raw_case_dir
    echo "...copying raw MindSpeed-LLM, MindSpeed, Megatron-LM, transformers, MSAdapter from $ROOT_DIR to $raw_case_dir ..."
    echo "binary_search workspace: ${workspace_dir}, raw_case dir: ${raw_case_dir}"

    # Copy the patch file
    local raw_patch=${raw_case_dir}${json_name}
    cp $patch_json_path $raw_patch

    local left=0
    local len=$(python $CUT_PY --input $raw_patch --count)
    local check_id=0
    echo "...Total $len patches in $raw_patch"
    local right=$((len - 1))

    # Binary search generates a new patch.json
    while [ $left -le $right ]; do
        local mid=$((left + (right - left) / 2))
        local check_case="case_binary_$(basename "$patch_json_path" .json)_l${left}_m${mid}_r${right}"

        echo "...left $left, right $right, mid $mid"

        # Create the test_case folder and enter it
        local test_case_dir=${workspace_dir}${check_case}/
        mkdir -p $test_case_dir
        cd $test_case_dir

        cp -r $raw_case_dir $test_case_dir
        echo "...copying from $raw_case_dir to $test_case_dir ..."

        # Take the first half of the patch for verification
        local test_patch=${test_case_dir}${json_name}
        if [ -f $test_patch ]; then
            rm $test_patch
        fi

        # generate binsearch patch
        python $CUT_PY --input $raw_patch --output $test_patch --left $left --right $mid
        if [ $? -ne 0 ]; then
            echo "...Run $CUT_PY failed while selecting [$left,$mid] patches from $raw_patch and write to $test_patch"
            exit 1
        fi
        echo "...Selected [$left,$mid] patches from $raw_patch and write to $test_patch"

        # patch merging
        echo "...Start merge patch $test_patch in $test_case_dir"
        do_merge $test_case_dir $test_patch
        ret_merge=$?

        if [ $ret_merge -ne 0 ]; then  # merge failed
            echo_error "...Merge patch failed: $test_patch"
            exit 1
        else
            echo_success "...Merge patch $test_patch in $test_case_dir finished"
        fi

        # run test
        echo "...Start run model test $test_patch"
        do_test $test_case_dir
        check_log $test_case_dir $test_patch
        local ret_test=$?

        if [ $ret_test -ne 0 ]; then  # test failed
            echo_error "...run model test failed: $test_patch"
            if [ $left -eq $right ]; then
                echo "...found bad case"
                break
            fi
            right=$mid
        else
            echo_success "...run model test $test_patch success"
            left=$((mid + 1))
        fi

        check_id=$((check_id + 1)) 
    done

    echo "...last check id: {$check_id}, check workspace_dir $workspace_dir"
}

# Single patch.json test
test_single_case() {
    local patch_json_path=$1
    if [ ! -f $patch_json_path ]; then
        echo_error "File not Found: $patch_json_path"
        exit 1
    fi
    local json_name=$(basename "$patch_json_path")

    # Preparation workspace
    local workspace_dir=$ROOT_DIR/"test_patch_merge_single_case/" #$(date +"%Y%m%d%H_%M%S")"

    if [ -d $workspace_dir ]; then
        rm -rf $workspace_dir
    fi
    mkdir -p $workspace_dir
    echo "...making dir $workspace_dir..."
    cd $workspace_dir

    cp -r $ROOT_DIR"MindSpeed-LLM" $workspace_dir
    cp -r $ROOT_DIR"MindSpeed" $workspace_dir
    cp -r $ROOT_DIR"Megatron-LM" $workspace_dir
    cp -r $ROOT_DIR"transformers" $workspace_dir
    cp -r $ROOT_DIR"MSAdapter" $workspace_dir
    echo "...copying raw MindSpeed-LLM, MindSpeed, Megatron-LM, transformers, MSAdapter from $ROOT_DIR to $workspace_dir ..."

    # Copy the patch file
    local test_patch=$workspace_dir/$(basename "$patch_json_path")
    cp $patch_json_path $test_patch

    local len=$(python $CUT_PY --input $test_patch --count)
    echo "...Total $len module patches in $test_patch"

    # patch merging
    do_merge $workspace_dir $test_patch
    ret_merge=$?

    if [ $ret_merge -ne 0 ]; then  # merge failed
        echo_error "...Merge patch failed: $test_patch"
        exit 1
    else
        echo_success "...Merge patch success: $test_patch"
    fi

    # run test
    do_test $workspace_dir
    check_log $workspace_dir $test_patch
    local ret_test=$?

    if [ $ret_test -ne 0 ]; then  # test failed
        echo_error "...test patch failed: $test_patch"
    else
        echo_success "...test patch success: $test_patch"
    fi


    echo "...done test check workspace_dir $workspace_dir"
}


test_single_case ${PATCH_JSON_PATH}
# binary_search ${PATCH_JSON_PATH}
