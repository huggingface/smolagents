#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Security tests for SafeSerializer to verify pickle is properly blocked.
"""

import base64
import pickle
import warnings
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from smolagents.serialization import SafeSerializer, SerializationError


# Test class that CAN be pickled (module-level)
class PicklableCustomClass:
    def __init__(self):
        self.value = 42


class TestSafeSerializationSecurity:
    """Test that safe mode properly blocks pickle."""

    def test_safe_mode_blocks_custom_classes(self):
        """Verify custom classes cannot be serialized in safe mode."""

        class CustomClass:
            def __init__(self):
                self.value = 42

        obj = CustomClass()

        # Should raise SerializationError in safe mode
        with pytest.raises(SerializationError, match="Cannot safely serialize"):
            SafeSerializer.dumps(obj, allow_pickle=False)

    def test_safe_mode_blocks_pickle_deserialization(self):
        """Verify pickle data is rejected in safe mode."""

        # Create pickle data (no "safe:" prefix)
        pickle_data = base64.b64encode(pickle.dumps({"test": "data"})).decode()

        # Should raise error in safe mode
        with pytest.raises(SerializationError, match="Pickle data rejected"):
            SafeSerializer.loads(pickle_data, allow_pickle=False)

    def test_pickle_fallback_with_warning(self):
        """Verify pickle fallback works but warns in legacy mode."""

        obj = PicklableCustomClass()

        # Should work but emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            serialized = SafeSerializer.dumps(obj, allow_pickle=True)

            # Check warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "insecure pickle" in str(w[0].message).lower()

        # Should deserialize successfully (with warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = SafeSerializer.loads(serialized, allow_pickle=True)

            assert result.value == 42
            assert len(w) == 1
            assert "pickle data" in str(w[0].message).lower()


class TestSafeSerializationRoundtrip:
    """Test that safe types serialize and deserialize correctly."""

    def test_primitives(self):
        """Test basic Python types."""
        test_cases = [
            None,
            True,
            False,
            42,
            3.14,
            "hello",
            b"bytes",
            complex(1, 2),
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            assert serialized.startswith("safe:")
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_collections(self):
        """Test collections."""
        test_cases = [
            [1, 2, 3],
            {"key": "value", "nested": {"a": 1}},
            (1, 2, 3),
            {1, 2, 3},
            frozenset([1, 2, 3]),
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_datetime_types(self):
        """Test datetime module types."""
        now = datetime.now()
        test_cases = [
            now,
            now.date(),
            now.time(),
            timedelta(days=1, hours=2, minutes=3),
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_special_types(self):
        """Test Decimal and Path."""
        test_cases = [
            Decimal("3.14159"),
            Path("/tmp/test.txt"),
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_complex_nested_structure(self):
        """Test deeply nested structures."""
        obj = {
            "primitives": [1, 2.5, "string", None, True],
            "collections": {
                "list": [1, 2, 3],
                "tuple": (4, 5, 6),
                "set": {7, 8, 9},
            },
            "datetime": datetime.now(),
            "path": Path("/tmp"),
            "bytes": b"binary data",
        }

        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        assert serialized.startswith("safe:")
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        # Check structure is preserved
        assert result["primitives"] == obj["primitives"]
        assert result["collections"]["list"] == obj["collections"]["list"]
        assert result["datetime"] == obj["datetime"]
        assert result["path"] == obj["path"]
        assert result["bytes"] == obj["bytes"]


class TestNumpySupport:
    """Test numpy array serialization (optional, skip if not installed)."""

    def test_numpy_array(self):
        """Test numpy array roundtrip."""
        pytest.importorskip("numpy")
        import numpy as np

        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

        serialized = SafeSerializer.dumps(arr, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        np.testing.assert_array_equal(result, arr)
        assert result.dtype == arr.dtype

    def test_numpy_scalars(self):
        """Test numpy scalar types."""
        pytest.importorskip("numpy")
        import numpy as np

        test_cases = [
            np.int32(42),
            np.float64(3.14),
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj.item()


class TestPILSupport:
    """Test PIL Image serialization (optional, skip if not installed)."""

    def test_pil_image(self):
        """Test PIL Image roundtrip."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a simple test image
        img = Image.new("RGB", (10, 10), color="red")

        serialized = SafeSerializer.dumps(img, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        assert isinstance(result, Image.Image)
        assert result.size == img.size
        assert result.mode == img.mode


class TestBackwardCompatibility:
    """Test that legacy pickle data can still be read when explicitly allowed."""

    def test_read_legacy_pickle_data(self):
        """Verify we can read old pickle data when allow_insecure=True."""

        # Simulate legacy pickle data (no "safe:" prefix)
        legacy_data = {"key": "value", "number": 42}
        pickle_encoded = base64.b64encode(pickle.dumps(legacy_data)).decode()

        # Should work with allow_pickle=True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = SafeSerializer.loads(pickle_encoded, allow_pickle=True)

            assert result == legacy_data
            assert len(w) == 1  # Warning emitted
            assert "pickle data" in str(w[0].message).lower()

    def test_safe_data_is_preferred(self):
        """Verify safe serialization is used even when pickle is allowed."""

        # Basic dict should use safe serialization
        obj = {"key": [1, 2, 3]}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            serialized = SafeSerializer.dumps(obj, allow_pickle=True)

            # Should use safe format (no warning)
            assert serialized.startswith("safe:")
            assert len(w) == 0  # No warning because safe was used


class TestDefaultBehavior:
    """Test that defaults are secure."""

    def test_dumps_defaults_to_safe(self):
        """Verify dumps defaults to safe mode."""
        obj = {"key": "value"}

        # Call without safe_serialization parameter - should default to True
        serialized = SafeSerializer.dumps(obj)
        assert serialized.startswith("safe:")

        # Should be deserializable in safe mode
        result = SafeSerializer.loads(serialized)
        assert result == obj

    def test_loads_defaults_to_safe(self):
        """Verify loads defaults to safe mode."""
        # Create safe data
        obj = {"key": "value"}
        serialized = SafeSerializer.dumps(obj, allow_pickle=False)

        # Call without safe_serialization parameter - should default to True
        result = SafeSerializer.loads(serialized)
        assert result == obj

        # Create pickle data
        pickle_data = base64.b64encode(pickle.dumps(obj)).decode()

        # Should reject pickle data by default
        with pytest.raises(SerializationError, match="Pickle data rejected"):
            SafeSerializer.loads(pickle_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
