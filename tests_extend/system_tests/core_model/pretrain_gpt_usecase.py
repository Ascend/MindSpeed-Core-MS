import time
import copy
import json
import os

import yaml



def get_yaml(file_path):
    """
    Read the case YAML file.
    @param file_path: file path
    @return:
    """
    with open(file=file_path, mode='r', encoding='utf-8') as f:
        crf = f.read()
        yaml_data = yaml.safe_load(stream=crf)
        return yaml_data


def get_usecase_config_list(usecase_data):
    """
    Obtaining the Case Configuration List
    @param usecase_data: case data
    @return:
    """
    all_config_list = []
    for config_dict in usecase_data['products']:
        if not isinstance(config_dict, dict):
            raise AssertionError(f'config_dict format error, expected dict, currently : {type(config_dict)}')
        config_keys = dict(config_dict).keys()

        config_list = [{}]
        for key in config_keys:
            if not isinstance(config_dict[key], list):
                raise AssertionError(f"config {key} format error, expected list, currently :{type(config_dict[key])}")
            new_config_list = []
            for val in config_dict[key]:
                tmp_config_list = copy.deepcopy(config_list)
                for uc_conf in tmp_config_list:
                    uc_conf[key] = val
                    new_config_list.append(uc_conf)
            config_list = copy.deepcopy(new_config_list)
        all_config_list.append(config_list)

    return all_config_list


def xtest_gpt_usecase(usecase_yaml, usecase_script):
    # Case files are in the current directory.
    usecase_file = os.path.join(os.path.dirname(__file__), usecase_yaml)

    # Read Case Data
    usecase_data = get_yaml(file_path=usecase_file)

    # Specified parameters
    sepc_param_dict = dict(usecase_data['spec'])

    # Obtains the case group list (a row indicates a group of cases).
    usecase_config_list = get_usecase_config_list(usecase_data)
    # Create a log directory.
    logs_dir = f"./{time.strftime('%Y_%m_%d', time.localtime(int(time.time())))}logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    for case_sequence_index, usecase in usecase_config_list:  # Executing Cases in Sequence
        for case_group_index, uc_instance in usecase:  # Use cases in the same group
            if 'yaml_cfg' in uc_instance.keys():
                uc_instance['yaml_cfg'] = os.path.join(os.path.dirname(__file__), uc_instance['yaml_cfg'])
            try:
                # Preparing Case Parameters
                param_list = [f"{str(key).upper()}={str(value)}" for key, value in uc_instance.items()]
                param_str = " ".join(param_list)
                script_path = os.path.join(os.path.dirname(__file__), usecase_script)
                print("=================== STARTING ======================")
                print(f"param : {param_str}")

                # The configuration of the use case takes precedence over the specified configuration.
                # (If the parameter configured in the case is deleted from the specified parameter)
                [sepc_param_dict.pop(k) for k in uc_instance.keys() if k in sepc_param_dict.keys()]
                sepc_param = " ".join([f"{str(key).upper()}={str(value)}" for key, value in sepc_param_dict.items()])
                print(f"spec_param : {sepc_param}")

                log_save_path = os.path.join(logs_dir,
                                             f"usecase-{case_sequence_index}-{case_group_index}-{time.time_ns()}.log")
                # run usecase
                os.system(f"sh {script_path} {param_str} {sepc_param} 2>&1 | tee {log_save_path}")
                print("==================== ENDING =======================")
            except Exception as e:
                # Save failed case information
                print("usecase :", usecase)
                print("\n")
                print(str(e))
                print(" ===================\n ")


if __name__ == "__main__":
    xtest_gpt_usecase("gpt-usecase.yaml", "pretrain_gpt_usecase.sh")
