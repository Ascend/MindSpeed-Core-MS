#!/bin/bash

script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")

# Replace the import statement
sed -i 's/from rules.line_rules import/from .rules.line_rules import/' ${MindSpeed_Core_MS_PATH}/tools/transfer.py
# Check whether the previous command was successful
if [ $? -eq 0 ]; then
  echo "Success: Import statement replaced successfully!"
else
  echo "Error: Import statement replaced failed!."
  exit 1
fi

# Comment statements
sed -i -E 's/^(from \.debug_utils import|__all__ = )/#&/' ${MindSpeed_Core_MS_PATH}/tools/__init__.py
# Check whether the previous command was successful
if [ $? -eq 0 ]; then
  echo "Success: Annotation successfully!"
else
  echo "Error: Annotation failed!."
  exit 1
fi
