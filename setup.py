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
