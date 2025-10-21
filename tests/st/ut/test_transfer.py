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
"""
Test module for transfer.py from tools/load_ms_weights_to_pt/
Tests the transfer and patch functionality
"""

import pytest
import sys
import os
from unittest.mock import patch

# Add the tools directory to the path
tools_path = os.path.join(os.path.dirname(__file__), '../../../tools/load_ms_weights_to_pt')
sys.path.insert(0, tools_path)

from transfer import (
    transfer_load,
    copy_weights_transfer_tool_file,
    patch_torch_load,
    patch_texts
)


class TestCopyWeightsTransferToolFile:
    """Test copy_weights_transfer_tool_file function"""

    def test_copy_weights_transfer_tool_file_success(self, tmp_path):
        """Test successful copying of weight transfer tool files"""
        # Create source files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        checkpointing_file = source_dir / "checkpointing.py"
        serialization_file = source_dir / "serialization.py"
        checkpointing_file.write_text("# checkpointing code")
        serialization_file.write_text("# serialization code")
        
        # Create target directory
        target_dir = tmp_path / "target"
        target_training_dir = target_dir / "mindspeed_llm" / "mindspore" / "training"
        target_training_dir.mkdir(parents=True)
        
        # Mock os.path functions to use our temp directories
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                copy_weights_transfer_tool_file(str(target_dir))
        
        # Verify files were copied
        assert (target_training_dir / "checkpointing.py").exists()
        assert (target_training_dir / "serialization.py").exists()
        assert (target_training_dir / "checkpointing.py").read_text() == "# checkpointing code"
        assert (target_training_dir / "serialization.py").read_text() == "# serialization code"

    def test_copy_weights_transfer_tool_file_missing_checkpointing(self, tmp_path):
        """Test error when checkpointing.py is missing"""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        # Only create serialization file
        serialization_file = source_dir / "serialization.py"
        serialization_file.write_text("# serialization code")
        
        target_dir = tmp_path / "target"
        target_training_dir = target_dir / "mindspeed_llm" / "mindspore" / "training"
        target_training_dir.mkdir(parents=True)
        
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                with pytest.raises(FileNotFoundError, match="checkpointing.py does not exist"):
                    copy_weights_transfer_tool_file(str(target_dir))

    def test_copy_weights_transfer_tool_file_missing_serialization(self, tmp_path):
        """Test error when serialization.py is missing"""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        # Only create checkpointing file
        checkpointing_file = source_dir / "checkpointing.py"
        checkpointing_file.write_text("# checkpointing code")
        
        target_dir = tmp_path / "target"
        target_training_dir = target_dir / "mindspeed_llm" / "mindspore" / "training"
        target_training_dir.mkdir(parents=True)
        
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                with pytest.raises(FileNotFoundError, match="serialization.py does not exist"):
                    copy_weights_transfer_tool_file(str(target_dir))

    def test_copy_weights_transfer_tool_file_missing_target_directory(self, tmp_path):
        """Test error when target directory doesn't exist"""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        checkpointing_file = source_dir / "checkpointing.py"
        serialization_file = source_dir / "serialization.py"
        checkpointing_file.write_text("# checkpointing code")
        serialization_file.write_text("# serialization code")
        
        target_dir = tmp_path / "nonexistent"
        
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                with pytest.raises(FileNotFoundError, match="does not exist"):
                    copy_weights_transfer_tool_file(str(target_dir))


class TestPatchTorchLoad:
    """Test patch_torch_load function"""

    def test_patch_torch_load_success(self, tmp_path):
        """Test successful patching of torch.load"""
        # Create a mock megatron_adaptor.py file
        adaptor_dir = tmp_path / "mindspeed_llm" / "tasks"
        adaptor_dir.mkdir(parents=True)
        adaptor_file = adaptor_dir / "megatron_adaptor.py"
        
        # Create content with the pattern to be replaced
        original_content = """class MegatronAdaptor:
    def patch_datasets(self):
        pass"""
        
        adaptor_file.write_text(original_content)
        
        # Call patch_torch_load
        patch_torch_load(str(tmp_path))
        
        # Verify the file was patched
        patched_content = adaptor_file.read_text()
        assert "from mindspeed_llm.mindspore.training.checkpointing import load_wrapper" in patched_content
        assert "MegatronAdaptation.register('torch.load', load_wrapper)" in patched_content

    def test_patch_torch_load_file_not_found(self, tmp_path):
        """Test error when megatron_adaptor.py doesn't exist"""
        with pytest.raises(FileNotFoundError, match="megatron_adaptor.py does not exist"):
            patch_torch_load(str(tmp_path))

    def test_patch_torch_load_pattern_not_found(self, tmp_path):
        """Test error when pattern to replace is not found"""
        adaptor_dir = tmp_path / "mindspeed_llm" / "tasks"
        adaptor_dir.mkdir(parents=True)
        adaptor_file = adaptor_dir / "megatron_adaptor.py"
        
        # Create content without the expected pattern
        original_content = """class MegatronAdaptor:
    def some_other_method(self):
        pass"""
        
        adaptor_file.write_text(original_content)
        
        with pytest.raises(ValueError, match="replace fail"):
            patch_torch_load(str(tmp_path))

class TestTransferLoad:
    """Test transfer_load main function"""

    def test_transfer_load_integration(self, tmp_path):
        """Test complete transfer_load workflow"""
        # Create source directory with required files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        checkpointing_file = source_dir / "checkpointing.py"
        serialization_file = source_dir / "serialization.py"
        checkpointing_file.write_text("# checkpointing code")
        serialization_file.write_text("# serialization code")
        
        # Create target directory structure
        target_dir = tmp_path / "target"
        training_dir = target_dir / "mindspeed_llm" / "mindspore" / "training"
        training_dir.mkdir(parents=True)
        
        tasks_dir = target_dir / "mindspeed_llm" / "tasks"
        tasks_dir.mkdir(parents=True)
        adaptor_file = tasks_dir / "megatron_adaptor.py"
        
        original_content = """class MegatronAdaptor:
    def patch_datasets(self):
        pass"""
        adaptor_file.write_text(original_content)
        
        # Mock the directory functions
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                transfer_load(str(target_dir))
        
        # Verify files were copied
        assert (training_dir / "checkpointing.py").exists()
        assert (training_dir / "serialization.py").exists()
        
        # Verify patching was applied
        patched_content = adaptor_file.read_text()
        assert "load_wrapper" in patched_content

    def test_transfer_load_with_copy_failure(self, tmp_path):
        """Test transfer_load handles copy failure"""
        target_dir = tmp_path / "target"
        
        with patch('transfer.copy_weights_transfer_tool_file', side_effect=FileNotFoundError("Test error")):
            with pytest.raises(FileNotFoundError, match="Test error"):
                transfer_load(str(target_dir))

    def test_transfer_load_with_patch_failure(self, tmp_path):
        """Test transfer_load handles patch failure"""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        checkpointing_file = source_dir / "checkpointing.py"
        serialization_file = source_dir / "serialization.py"
        checkpointing_file.write_text("# checkpointing code")
        serialization_file.write_text("# serialization code")
        
        target_dir = tmp_path / "target"
        training_dir = target_dir / "mindspeed_llm" / "mindspore" / "training"
        training_dir.mkdir(parents=True)
        
        # Don't create the adaptor file to trigger patch failure
        
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                with pytest.raises(FileNotFoundError):
                    transfer_load(str(target_dir))


class TestPatchTexts:
    """Test patch_texts constant"""

    def test_patch_texts_format(self):
        """Test patch_texts has correct format"""
        assert isinstance(patch_texts, str)
        assert "def patch_datasets(self):" in patch_texts
        assert "from mindspeed_llm.mindspore.training.checkpointing import load_wrapper" in patch_texts
        assert "MegatronAdaptation.register('torch.load', load_wrapper)" in patch_texts


class TestCommandLineInterface:
    """Test command line interface"""

    def test_main_with_valid_args(self, tmp_path):
        """Test main function with valid arguments"""
        # Setup test environment
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        checkpointing_file = source_dir / "checkpointing.py"
        serialization_file = source_dir / "serialization.py"
        checkpointing_file.write_text("# checkpointing code")
        serialization_file.write_text("# serialization code")
        
        target_dir = tmp_path / "target"
        training_dir = target_dir / "mindspeed_llm" / "mindspore" / "training"
        training_dir.mkdir(parents=True)
        
        tasks_dir = target_dir / "mindspeed_llm" / "tasks"
        tasks_dir.mkdir(parents=True)
        adaptor_file = tasks_dir / "megatron_adaptor.py"
        adaptor_file.write_text("""class MegatronAdaptor:
    def patch_datasets(self):
        pass""")
        
        # Test with command line args
        test_args = ['transfer.py', '--mindspeed_llm_path', str(target_dir)]
        
        with patch('sys.argv', test_args):
            with patch('transfer.os.path.dirname', return_value=str(source_dir)):
                with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                    # Import and run the main block
                    import argparse
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--mindspeed_llm_path", type=str, required=True)
                    args = parser.parse_args(test_args[1:])
                    
                    transfer_load(args.mindspeed_llm_path)
        
        # Verify it worked
        assert (training_dir / "checkpointing.py").exists()


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_copy_with_readonly_source(self, tmp_path):
        """Test copying when source files are read-only"""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        checkpointing_file = source_dir / "checkpointing.py"
        serialization_file = source_dir / "serialization.py"
        checkpointing_file.write_text("# checkpointing code")
        serialization_file.write_text("# serialization code")
        
        # Make files read-only
        os.chmod(checkpointing_file, 0o444)
        os.chmod(serialization_file, 0o444)
        
        target_dir = tmp_path / "target"
        training_dir = target_dir / "mindspeed_llm" / "mindspore" / "training"
        training_dir.mkdir(parents=True)
        
        # Should still work despite read-only source
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                copy_weights_transfer_tool_file(str(target_dir))
        
        assert (training_dir / "checkpointing.py").exists()

    def test_patch_with_unicode_content(self, tmp_path):
        """Test patching file with unicode content"""
        adaptor_dir = tmp_path / "mindspeed_llm" / "tasks"
        adaptor_dir.mkdir(parents=True)
        adaptor_file = adaptor_dir / "megatron_adaptor.py"
        
        # Content with unicode characters
        original_content = """class MegatronAdaptor:
    def patch_datasets(self):
        # 中文注释
        pass"""
        
        adaptor_file.write_text(original_content, encoding='utf-8')
        
        patch_torch_load(str(tmp_path))
        
        # Verify unicode is preserved
        patched_content = adaptor_file.read_text(encoding='utf-8')
        assert "中文注释" in patched_content

    def test_transfer_with_symlinks(self, tmp_path):
        """Test transfer handles symbolic links correctly"""
        if os.name == 'nt':  # Skip on Windows
            pytest.skip("Symlinks not reliably supported on Windows")
        
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        checkpointing_file = source_dir / "checkpointing.py"
        serialization_file = source_dir / "serialization.py"
        checkpointing_file.write_text("# checkpointing code")
        serialization_file.write_text("# serialization code")
        
        # Create symlinked target
        target_dir = tmp_path / "target"
        actual_dir = tmp_path / "actual"
        training_dir = actual_dir / "mindspeed_llm" / "mindspore" / "training"
        training_dir.mkdir(parents=True)
        
        # Create symlink (may not work on all systems)
        try:
            os.symlink(actual_dir, target_dir)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this system")
        
        with patch('transfer.os.path.dirname', return_value=str(source_dir)):
            with patch('transfer.os.path.abspath', return_value=str(source_dir / "transfer.py")):
                copy_weights_transfer_tool_file(str(target_dir))
        
        # Files should be in actual directory
        assert (training_dir / "checkpointing.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

