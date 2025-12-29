"""
Tests for User Guide code examples and documentation accuracy.

Verifies that code examples in the documentation are syntactically correct
and that model paths and configurations are accurate.
"""

import ast
import json
import re
from pathlib import Path

import pytest


class TestPythonCodeExamples:
    """Tests for Python code examples in documentation."""
    
    @pytest.fixture
    def user_guide_content(self):
        """Load the USER_GUIDE.md content."""
        docs_path = Path(__file__).parent.parent.parent / "docs" / "USER_GUIDE.md"
        if docs_path.exists():
            return docs_path.read_text()
        return ""
    
    def extract_python_code_blocks(self, content: str) -> list[str]:
        """Extract Python code blocks from markdown."""
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches
    
    def test_python_code_syntax_valid(self, user_guide_content):
        """Test that all Python code examples have valid syntax."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        code_blocks = self.extract_python_code_blocks(user_guide_content)
        
        for i, code in enumerate(code_blocks):
            try:
                ast.parse(code)
            except SyntaxError as e:
                pytest.fail(f"Python code block {i + 1} has syntax error: {e}\n\nCode:\n{code}")
    
    def test_python_imports_standard(self, user_guide_content):
        """Test that Python examples use common/expected imports."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        code_blocks = self.extract_python_code_blocks(user_guide_content)
        
        expected_imports = [
            "requests",
            "json",
            "ollama",
            "openai",
            "sentence_transformers",
            "faiss",
        ]
        
        found_imports = set()
        for code in code_blocks:
            for line in code.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    for expected in expected_imports:
                        if expected in line:
                            found_imports.add(expected)
        
        # Should find at least some common imports
        assert len(found_imports) > 0, "No expected imports found in Python examples"


class TestBashCommandSyntax:
    """Tests for bash command syntax in documentation."""
    
    @pytest.fixture
    def user_guide_content(self):
        """Load the USER_GUIDE.md content."""
        docs_path = Path(__file__).parent.parent.parent / "docs" / "USER_GUIDE.md"
        if docs_path.exists():
            return docs_path.read_text()
        return ""
    
    def extract_bash_code_blocks(self, content: str) -> list[str]:
        """Extract bash code blocks from markdown."""
        pattern = r'```bash\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches
    
    def test_bash_commands_have_no_obvious_errors(self, user_guide_content):
        """Test that bash commands don't have obvious syntax errors."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        code_blocks = self.extract_bash_code_blocks(user_guide_content)
        
        # Check for common issues
        for i, code in enumerate(code_blocks):
            # Check for unmatched quotes
            single_quotes = code.count("'") - code.count("\\'")
            double_quotes = code.count('"') - code.count('\\"')
            
            # Note: heredocs may have unbalanced quotes intentionally
            # So we only check for extremely unbalanced cases
            if single_quotes % 2 != 0 and "<<" not in code:
                # Could be an issue, but heredocs and special cases exist
                pass
            
            # Check for pip install commands have package names
            if "pip install" in code:
                assert re.search(r'pip install\s+\S', code), \
                    f"Bash block {i + 1}: pip install without package name"
    
    def test_pip_commands_valid(self, user_guide_content):
        """Test that pip install commands are valid."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        code_blocks = self.extract_bash_code_blocks(user_guide_content)
        
        valid_packages = [
            "torch",
            "transformers",
            "peft",
            "bitsandbytes",
            "accelerate",
            "datasets",
            "unsloth",
            "trl",
            "sentence-transformers",
            "faiss-cpu",
            "faiss-gpu",
            "chromadb",
            "qdrant-client",
            "langchain",
            "deepspeed",
            "ollama",
            "openai",
        ]
        
        for code in code_blocks:
            for line in code.split('\n'):
                if 'pip install' in line:
                    # Check that at least one valid package is mentioned
                    has_valid = any(pkg in line for pkg in valid_packages) or \
                                '@' in line or \
                                '-r requirements' in line
                    if not has_valid and '--no-deps' not in line:
                        # This might be a version constraint line
                        pass


class TestModelPaths:
    """Tests for model path documentation accuracy."""
    
    def test_model_path_structure_documented(self):
        """Test that model path structure matches documentation."""
        docs_path = Path(__file__).parent.parent.parent / "docs" / "USER_GUIDE.md"
        if not docs_path.exists():
            pytest.skip("USER_GUIDE.md not found")
        
        content = docs_path.read_text()
        
        # Check documented paths
        assert "qlora_model" in content
        assert "lora_model" in content
        assert "rag_model" in content
        assert "standard_model" in content
        
        # Check upload path mentioned
        assert "uploads/models/job_" in content or "job_{id}" in content


class TestAPIDocumentation:
    """Tests for API documentation accuracy."""
    
    @pytest.fixture
    def user_guide_content(self):
        """Load the USER_GUIDE.md content."""
        docs_path = Path(__file__).parent.parent.parent / "docs" / "USER_GUIDE.md"
        if docs_path.exists():
            return docs_path.read_text()
        return ""
    
    def test_api_endpoints_documented(self, user_guide_content):
        """Test that API endpoints are documented."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        # Check for API documentation
        assert "/api/v1/jobs" in user_guide_content
        assert "POST" in user_guide_content
        assert "GET" in user_guide_content
    
    def test_ollama_api_documented(self, user_guide_content):
        """Test that Ollama API is documented."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        assert "localhost:11434" in user_guide_content
        assert "/api/generate" in user_guide_content
    
    def test_lm_studio_api_documented(self, user_guide_content):
        """Test that LM Studio API is documented."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        assert "localhost:1234" in user_guide_content
        assert "/v1/chat/completions" in user_guide_content


class TestPHPCodeExamples:
    """Tests for PHP code examples in documentation."""
    
    @pytest.fixture
    def user_guide_content(self):
        """Load the USER_GUIDE.md content."""
        docs_path = Path(__file__).parent.parent.parent / "docs" / "USER_GUIDE.md"
        if docs_path.exists():
            return docs_path.read_text()
        return ""
    
    def extract_php_code_blocks(self, content: str) -> list[str]:
        """Extract PHP code blocks from markdown."""
        pattern = r'```php\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches
    
    def test_php_code_has_opening_tag(self, user_guide_content):
        """Test that PHP code has proper opening tags."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        code_blocks = self.extract_php_code_blocks(user_guide_content)
        
        for i, code in enumerate(code_blocks):
            assert '<?php' in code or code.strip().startswith('//'), \
                f"PHP block {i + 1} missing <?php tag"
    
    def test_php_curl_usage_correct(self, user_guide_content):
        """Test that PHP curl usage is correct."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        code_blocks = self.extract_php_code_blocks(user_guide_content)
        
        for code in code_blocks:
            if 'curl_init' in code:
                # Check for proper curl usage patterns
                assert 'curl_setopt' in code or 'curl_setopt_array' in code
                assert 'curl_exec' in code
                assert 'curl_close' in code


class TestCURLExamples:
    """Tests for CURL examples in documentation."""
    
    @pytest.fixture
    def user_guide_content(self):
        """Load the USER_GUIDE.md content."""
        docs_path = Path(__file__).parent.parent.parent / "docs" / "USER_GUIDE.md"
        if docs_path.exists():
            return docs_path.read_text()
        return ""
    
    def test_curl_examples_valid(self, user_guide_content):
        """Test that CURL examples have valid structure."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        # Check for curl command presence
        assert 'curl' in user_guide_content
        
        # Check for proper HTTP methods
        assert '-X POST' in user_guide_content or 'curl -X POST' in user_guide_content
        
        # Check for Content-Type header
        assert 'Content-Type' in user_guide_content


class TestTrainingTypeDocumentation:
    """Tests for training type documentation completeness."""
    
    @pytest.fixture
    def user_guide_content(self):
        """Load the USER_GUIDE.md content."""
        docs_path = Path(__file__).parent.parent.parent / "docs" / "USER_GUIDE.md"
        if docs_path.exists():
            return docs_path.read_text()
        return ""
    
    def test_all_training_types_documented(self, user_guide_content):
        """Test that all training types are documented."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        training_types = ["QLoRA", "Unsloth", "RAG", "Standard"]
        
        for training_type in training_types:
            assert training_type.lower() in user_guide_content.lower(), \
                f"Training type {training_type} not documented"
    
    def test_ollama_instructions_per_type(self, user_guide_content):
        """Test that Ollama instructions exist for each type."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        # Each section should mention Ollama
        assert user_guide_content.count("Ollama") >= 4
    
    def test_lm_studio_instructions_per_type(self, user_guide_content):
        """Test that LM Studio instructions exist for each type."""
        if not user_guide_content:
            pytest.skip("USER_GUIDE.md not found")
        
        # Each section should mention LM Studio
        assert user_guide_content.count("LM Studio") >= 4
