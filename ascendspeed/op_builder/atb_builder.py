from ascendspeed.op_builder.builder import AscendSpeedOpBuilder
import os

ATB_HOME_PATH = "ATB_HOME_PATH"

class AtbOpBuilder(AscendSpeedOpBuilder):
    def __init__(self, name):
        super(AtbOpBuilder, self).__init__(name)

    def get_atb_path(self):
        if ATB_HOME_PATH in os.environ or os.path.exists(os.environ[ATB_HOME_PATH]):
            return os.environ[ATB_HOME_PATH]
        return None

    def include_paths(self):
        paths = super().include_paths()
        paths.append(os.path.join(self._torch_npu_path, 'include', 'third_party', 'acl', 'inc'))
        paths.append('ops/csrc/atb/extensions/inc')
        paths.append(os.path.join(self.get_atb_path(), 'include'))
        return paths
    
    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'",
            '-D ENABLE_ATB',
            '-fstack-protector-all', 
            '-Wl,-z,relro,-z,now,-z,noexecstack',
            '-fPIC', '-pie', '-Wl,--disable-new-dtags,--rpath', '-s'
        ]
        return args
    
    def extra_ldflags(self):
        flags = ['-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu',
                 '-L' + os.path.join(self.get_atb_path(), 'lib'), '-latb']
        return flags