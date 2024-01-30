import sys
import os
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by AscendSpeed.")

__description__ = 'AscendSpeed for LLMs of Ascend'
__version__ = '0.0.1'
__author__ = 'Ascend'
__long_description__ = 'AscendSpeed for LLMs of Ascend'
__url__ = 'https://gitee.com/ascend/AscendSpeed'
__download_url__ = 'https://gitee.com/ascend/AscendSpeed/release'
__keywords__ = 'Ascend, langauge, deep learning, NLP'
__license__ = 'See https://gitee.com/ascend/AscendSpeed'
__package_name__ = 'ascendspeed'
__contact_names__ = 'Ascend'


try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ''


###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


def req_file(filename):
    try:
        with open(filename) as f:
            content = f.readlines()
        res = [x.strip() for x in content]
    except FileNotFoundError:
        res = []
    return res


install_requires = req_file("requirements.txt")
cmd_class = {}
exts = []

try:
    import imp
    import glob
    from torch_npu.utils import cpp_extension
    from torch.utils.cpp_extension import BuildExtension

    ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME")
    op_files = []
    op_files += glob.glob("ascendspeed/ops/csrc/*.cpp")
    op_files += glob.glob("ascendspeed/ops/csrc/atb/*.cpp")
    op_files += glob.glob("ascendspeed/ops/csrc/cann/*.cpp")
    ext_ops = cpp_extension.NpuExtension(
        name="ascendspeed_ops",
        sources=op_files,
        extra_compile_args=[
        '-Wno-sign-compare',
        '-Wno-deprecated-declarations',
        '-Wno-return-type',
        '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
        '-I' + os.path.join(imp.find_module('torch_npu')[1], "include", "third_party", "acl", "inc"),
        '-I' + os.path.join(ASCEND_TOOLKIT_HOME, 'include'),
        '-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack',
        '-fPIC', '-pie', '-Wl,--disable-new-dtags,--rpath', '-s',
    ],
    )
    if os.getenv('ENABLE_ATB', '0') == '1':
        ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
        if ATB_HOME_PATH is None:
            print('Can not find env : ATB_HOME_PATH, will not enable ATB')
            exts.append(ext_ops)
        else:
            op_files += glob.glob("ascendspeed/ops/csrc/atb/utils/*.cpp")
            ext_ops = cpp_extension.NpuExtension(
                name="ascendspeed_ops",
                sources=op_files,
                extra_compile_args=[
                '-Wno-sign-compare',
                '-Wno-deprecated-declarations',
                '-Wno-return-type',
                '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
                '-I' + os.path.join(imp.find_module('torch_npu')[1], "include", "third_party", "acl", "inc"),
                '-I' + os.path.join(ASCEND_TOOLKIT_HOME, 'include'),
                '-I' + os.path.join(ATB_HOME_PATH, 'include'), '-D ENABLE_ATB',
                '-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack',
                '-fPIC', '-pie', '-Wl,--disable-new-dtags,--rpath', '-s',
            ],
            extra_link_args=[
                '-L' + os.path.join(ATB_HOME_PATH, 'lib'),
                '-latb'
            ],
            )
            exts.append(ext_ops)
    else:
        exts.append(ext_ops)
    cmd_class = {"build_ext": BuildExtension.with_options(use_ninja=False)}
except Exception:
    print('Can not find env : ASCEND_TOOLKIT_HOME or ATB_HOME_PATH, ops setup failed')

setuptools.setup(
    package_data={'ascendspeed':['ascendspeed/data/Makefile']},
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url=__url__,
    author=__contact_names__,
    maintainer=__contact_names__,
    # The licence under which the project is released
    license=__license__,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Supported python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
    cmdclass=cmd_class,
    ext_modules=exts
)
