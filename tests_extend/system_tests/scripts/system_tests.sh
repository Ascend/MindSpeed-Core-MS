#/bin/bash

docker_name=''
current_dir=$(dirname $(readlink -f $0))
pipeline_dir=$(dirname $(dirname $(dirname $(dirname $current_dir))))
megatron_dir=$pipeline_dir/Megatron-LM
cann_dir=/usr/local/Ascend/

#Check Parameters
for para in $*
do
    if [[ $para == --cann_dir* ]];then
        cann_dir=`echo ${para#*=}`
    elif [[ $para == --docker* ]]; then
        docker_name=`echo ${para#*=}`
    fi
done

#Check Megatron-LM
if [ ! -d $megatron_dir ];
then
    echo "> Please confirm that Megatron-LM and AscendSpeed are in the same folder!!!"
    exit 0
else
    echo "> Megatron-LM exists in ${pipeline_dir}."
    cp -r $pipeline_dir/AscendSpeed/tests_extend $megatron_dir
fi

#Check CANN
echo "> Sourcing ${cann_dir}/ascend-toolkit/set_env.sh."
echo "> If you need to change the path, please use --cann_dir=/Dirname/of/ascend-toolkit/"

test ! -d $cann_dir/ascend-toolkit && echo "Error: Ascend-toolkit is not installed in ${cann_dir}!!!" && exit 0

#Check Python
if [ ! -z "$docker_name" ];
then
    echo "> Use Docker env......"
    docker start $docker_name
    for sub_task in llama2 gpt;
    do
        docker exec $docker_name bash ${current_dir}/test_pipeline.sh ${cann_dir} ${pipeline_dir} ${sub_task}
    done
else
    echo "> Use local env......"
    for sub_task in llama2 gpt;
    do
        bash ${current_dir}/test_pipeline.sh ${cann_dir} ${pipeline_dir} ${sub_task}
    done
fi
