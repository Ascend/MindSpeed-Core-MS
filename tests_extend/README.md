# Tests Usage

1. Copy the entire `tests_extend` to the root path of `Megatron-LM`
   
    ```shell
    cp -r tests_extend {PATH_TO_MEGATRON_LM}
    ```

2. Run a single test by pytest command line under `Megatron-LM` root path
   ```shell
   pytest tests_extend/unit_tests/optimizer/test_distrib_optimizer.py
   ```
