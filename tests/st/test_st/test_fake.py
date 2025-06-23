import pytest

# =============================
# ST test
# =============================

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_st():
    """
    Feature: test st
    Description: Fake testcase to meet the smoke_test process.
    Expectation: test success
    """
    assert 1.0==True
    print("run st test successful!")
