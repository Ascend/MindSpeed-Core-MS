# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""One key translator"""
import os
import sys


def txt_to_py(txt_file_path, py_file_path):
    """
    Convert txt file to py file.

    :param txt_file_path: txt file path
    :param py_file_path: Path to the py file to be created
    """
    # Check if the txt file exists
    if not os.path.exists(txt_file_path):
        print(f"Error: The file {txt_file_path} does not exist.")
        return

    # Read the contents of the txt file
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Write to py file
    with open(py_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    # Check if the py file was successfully created
    if not os.path.exists(py_file_path):
        print(f"Error: Unable to create file {py_file_path}.")
    else:
        print("Transformation is generated.")


# Check if command line parameters are passed in
if len(sys.argv) < 2:
    print("usage method: python one_key_trans.py <py_file_path>")
    sys.exit(1)

py_path = sys.argv[1]  # The output file path is controlled by command-line parameters passed through the script
txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrain_gpt.txt')   # Enter the pretrain_gpt. txt file and place it in the current directory

# Call function
txt_to_py(txt_path, py_path)
