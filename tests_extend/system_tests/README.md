# Tests Usage

1. Put `AscendSpeed` and `Megatron-LM` in the same folder.

2. Install `AscendSpeed` and enable 'AscendSpeed Patch'.

3. Run all system tests by one scripts.
   ```shell
   bash /Path/To/AscendSpeed/tests_extend/system_tests/scripts/system_tests.sh
   ```
   Tips:
   1. `--cann_dir=` can be used to change the path of cann-toolkit. Default is `/usr/local/Ascend/`.
   2. `--docker=` can be used to run tests on docker env. Default is local env.

4. All test results will be collected in `/Dirname/To/AscendSpeed/logs`.
   