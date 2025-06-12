import argparse
import os
import shutil


patch_texts = """    def patch_datasets(self):
+        from mindspeed_llm.mindspore.training.checkpointing import load_wrapper
+        MegatronAdaptation.register('torch.load', load_wrapper)"""


def transfer_load(mindspeed_llm_path):
    copy_weights_transfer_tool_file(mindspeed_llm_path)
    patch_torch_load(mindspeed_llm_path)


def copy_weights_transfer_tool_file(mindspeed_llm_path):
    source_directory = os.path.dirname(os.path.abspath(__file__))
    checkpointing_file = os.path.join(source_directory, "checkpointing.py")
    serialization_file = os.path.join(source_directory, "serialization.py")
    if not os.path.exists(checkpointing_file):
        raise FileNotFoundError(f"load ms weights to pt failed, {checkpointing_file} does not exist")
    if not os.path.exists(serialization_file):
        raise FileNotFoundError(f"load ms weights to pt failed, {serialization_file} does not exist")

    target_directory = os.path.join(mindspeed_llm_path, "mindspeed_llm/mindspore/training/")
    if not os.path.exists(target_directory):
        raise FileNotFoundError(f"load ms weights to pt failed, {target_directory} does not exist")
    shutil.copy(checkpointing_file, target_directory)
    shutil.copy(serialization_file, target_directory)


def patch_torch_load(mindspeed_llm_path):
    patch_file_path = os.path.join(mindspeed_llm_path, "mindspeed_llm/tasks/megatron_adaptor.py")
    if not os.path.exists(patch_file_path):
        raise FileNotFoundError(f"load ms weights to pt failed, {patch_file_path} does not exist")
    with open(patch_file_path, 'r', encoding='UTF-8') as file:
        data = file.read()

    lines = [(line[0], line[1:]) for line in patch_texts.split('\n') if line != '']
    pattern = '\n'.join([line for type, line in lines if type != '+'])
    replace = '\n'.join([line for type, line in lines if type != '-'])
    if pattern in data:
        data = replace.join(data.split(pattern))
    else:
        raise ValueError(f"{patch_file_path} replace fail, pattern {pattern} doesn't exist in {patch_file_path}")

    with open(patch_file_path, 'w', encoding='UTF-8') as file:
        file.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindspeed_llm_path", type=str, required=True,
                        help="the path of mindspeed-llm package")

    args = parser.parse_args()
    mindspeed_llm_path = args.mindspeed_llm_path
    transfer_load(mindspeed_llm_path)