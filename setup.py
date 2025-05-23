"""Setup."""

import sys
import os
import shutil
import stat
import platform
from importlib import import_module
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py
from setuptools.command.install import install


def get_readme_content():
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, 'README.md'), encoding='UTF-8') as f:
        return f.read()


def get_platform():
    """
    Get platform name.

    Returns:
        str, platform name in lowercase.
    """
    return platform.system().strip().lower()


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    os_info = get_platform()
    cpu_info = platform.machine().strip()

    return 'mindspeed_ms platform: %s, cpu: %s' % (os_info, cpu_info)


def get_install_requires():
    """
    Get install requirements.

    Returns:
        list, list of dependent packages.
    """
    with open('requirements.txt') as file:
        return file.read().strip().splitlines()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def write_commit_id():
    ret_code = os.system("git rev-parse --abbrev-ref HEAD > ./mindspeed_ms/.commit_id "
                         "&& git log --abbrev-commit -1 >> ./mindspeed_ms/.commit_id")
    if ret_code != 0:
        sys.stdout.write("Warning: Can not get commit id information. Please make sure git is available.")
        os.system("echo 'git is not available while building.' > ./mindspeed_ms/.commit_id")


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        egg_info_dir = os.path.join(os.path.dirname(__file__), 'mindspeed_ms.egg-info')
        shutil.rmtree(egg_info_dir, ignore_errors=True)
        super().run()
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """Build py files."""

    def run(self):
        mindspeed_core_ms_lib_dir = os.path.join(os.path.dirname(__file__), 'build', 'lib', 'mindspeed_ms')
        shutil.rmtree(mindspeed_core_ms_lib_dir, ignore_errors=True)
        super().run()
        update_permissions(mindspeed_core_ms_lib_dir)


class Install(install):
    """Install."""

    def run(self):
        super().run()
        if sys.argv[-1] == 'install':
            pip = import_module('pip')
            mindspeed_core_ms_dir = os.path.join(os.path.dirname(pip.__path__[0]), 'mindspeed_ms')
            update_permissions(mindspeed_core_ms_dir)


if __name__ == '__main__':
    version_info = sys.version_info
    if (version_info.major, version_info.minor) < (3, 7):
        sys.stderr.write('Python version should be at least 3.7\r\n')
        sys.exit(1)

    write_commit_id()

    setup(
        name='mindspeed_ms',
        version='1.0.0',
        author='The MindSpore Authors',
        author_email='',
        url='',
        download_url='',
        project_urls={
            'Sources': 'https://gitee.com/ascend/MindSpeed-Core-MS',
            'Issue Tracker': 'https://gitee.com/ascend/MindSpeed-Core-MS/issues',
        },
        description=get_description(),
        long_description=get_readme_content(),
        long_description_content_type="text/markdown",
        # test_suite="tests",
        # packages=find_packages(exclude=["*tests*"]),
        platforms=[get_platform()],
        include_package_data=True,
        package_data={'mindspeed_ms': ['.commit_id']},
        cmdclass={
            'egg_info': EggInfo,
            'build_py': BuildPy,
            'install': Install,
        },
        python_requires='>=3.7',
        install_requires=get_install_requires(),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Environment :: Web Environment',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='mindspeed_ms',
    )