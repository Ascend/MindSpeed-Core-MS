# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Setup."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mindspeed_ms",
    version="0.0.1",
    author="MindSpeed-Core-MS",
    author_email="ybwang19@lzu.edu.cn",
    description="MindSpeed-Core-MS Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/ascend/MindSpeed-Core-MS",
    project_urls={
        "Bug Tracker": "https://gitee.com/ascend/MindSpeed-Core-MS/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=["", "tools"],
    package_data={"": ["*.sh"], "tools": ['*', '*/*', '*/*/*']},
    python_requires=">=3.9",
    install_requires=[
        "mindspore>=2.4",
        "requests",
    ],
)
