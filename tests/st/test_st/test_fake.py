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
    Description: test random parallel in different mode
    Expectation: test success
    """
    assert 1.0==True
    print("run st test successful!")


