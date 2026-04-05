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
Comprehensive tests for SafeSerializer covering edge cases, error handling,
performance, and integration scenarios.
"""

import base64
import json
import pickle
import warnings
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from smolagents.serialization import SafeSerializer, SerializationError


# Module-level class for pickle tests (local classes can't be pickled)
class PicklableCustomClass:
    """A simple class that can be pickled."""

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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data(self):
        """Test serialization of empty collections."""
        test_cases = [
            [],
            {},
            (),
            set(),
            frozenset(),
            "",
            b"",
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_nested_empty_structures(self):
        """Test deeply nested empty structures."""
        obj = {
            "empty_list": [],
            "empty_dict": {},
            "nested": {
                "empty_tuple": (),
                "empty_set": set(),
                "deeply_nested": {"still_empty": []},
            },
        }

        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert result == obj

    def test_very_large_numbers(self):
        """Test handling of very large integers and floats."""
        test_cases = [
            10**100,  # Very large int
            -(10**100),  # Very large negative int
            1.7976931348623157e308,  # Near max float
            2.2250738585072014e-308,  # Near min positive float
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_special_float_values(self):
        """Test special float values (infinity, nan)."""
        import math

        # Note: NaN != NaN, so we handle it specially
        test_cases = [
            (float("inf"), float("inf")),
            (float("-inf"), float("-inf")),
        ]

        for obj, expected in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == expected

        # NaN special case
        nan_obj = float("nan")
        serialized = SafeSerializer.dumps(nan_obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert math.isnan(result)

    def test_unicode_strings(self):
        """Test handling of various unicode strings."""
        test_cases = [
            "Hello 世界",  # Mixed ASCII and Chinese
            "🚀🎉💎",  # Emojis
            "Ñoño",  # Accented characters
            "\u0000",  # Null character
            "Line1\nLine2\tTabbed",  # Escape sequences
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_very_long_strings(self):
        """Test handling of very long strings."""
        long_string = "a" * 1_000_000  # 1MB string

        serialized = SafeSerializer.dumps(long_string, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert result == long_string

    def test_deeply_nested_structures(self):
        """Test deeply nested data structures."""
        # Create nested structure
        obj = {"level": 0}
        current = obj
        for i in range(1, 100):  # 100 levels deep
            current["nested"] = {"level": i}
            current = current["nested"]

        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert result == obj

    def test_dict_with_tuple_keys(self):
        """Test dictionaries with tuple keys."""
        obj = {
            (1, 2): "tuple_key",
            (3, 4, 5): "longer_tuple",
            ("a", "b"): "string_tuple",
        }

        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert result == obj

    def test_dict_with_integer_keys(self):
        """Test dictionaries with non-string keys."""
        obj = {
            1: "one",
            2: "two",
            100: "hundred",
        }

        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert result == obj

    def test_mixed_collection_types(self):
        """Test mixed collection types in one structure."""
        obj = {
            "list": [1, 2, 3],
            "tuple": (4, 5, 6),
            "set": {7, 8, 9},
            "frozenset": frozenset([10, 11, 12]),
            "nested_list": [[1, 2], [3, 4]],
            "list_of_tuples": [(1, 2), (3, 4)],
        }

        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        # Compare each field
        assert result["list"] == obj["list"]
        assert result["tuple"] == obj["tuple"]
        assert result["set"] == obj["set"]
        assert result["frozenset"] == obj["frozenset"]
        assert result["nested_list"] == obj["nested_list"]
        assert result["list_of_tuples"] == obj["list_of_tuples"]


class TestErrorHandling:
    """Test error handling and malformed data."""

    def test_invalid_json_data(self):
        """Test handling of invalid JSON data."""
        with pytest.raises((json.JSONDecodeError, SerializationError)):
            SafeSerializer.loads("safe:invalid json", allow_pickle=False)

    def test_corrupted_safe_prefix(self):
        """Test handling of data with safe prefix but invalid JSON."""
        with pytest.raises((json.JSONDecodeError, SerializationError)):
            SafeSerializer.loads("safe:{broken", allow_pickle=False)

    def test_missing_type_field(self):
        """Test handling of malformed type markers."""
        # Valid JSON but missing required fields
        malformed = "safe:" + json.dumps({"data": [1, 2, 3]})  # Missing __type__

        # Should still work as regular dict
        result = SafeSerializer.loads(malformed, allow_pickle=False)
        assert result == {"data": [1, 2, 3]}

    def test_unknown_type_marker(self):
        """Test handling of unknown type markers."""
        unknown_type = "safe:" + json.dumps({"__type__": "unknown_type", "data": "something"})

        # Should return as dict with type marker
        result = SafeSerializer.loads(unknown_type, allow_pickle=False)
        assert "__type__" in result

    def test_invalid_base64_in_bytes(self):
        """Test handling of invalid base64 in bytes type."""
        invalid_bytes = "safe:" + json.dumps({"__type__": "bytes", "data": "not-valid-base64!!!"})

        with pytest.raises(Exception):  # Will raise base64 decode error
            SafeSerializer.loads(invalid_bytes, allow_pickle=False)

    def test_serialization_of_none_type(self):
        """Test that None type is handled correctly."""
        obj = None
        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert result is None

    def test_serialization_of_function(self):
        """Test that functions cannot be serialized safely."""

        def my_function():
            pass

        with pytest.raises(SerializationError):
            SafeSerializer.dumps(my_function, allow_pickle=False)

    def test_serialization_of_class(self):
        """Test that classes cannot be serialized safely."""

        class MyClass:
            pass

        with pytest.raises(SerializationError):
            SafeSerializer.dumps(MyClass, allow_pickle=False)

    def test_serialization_of_module(self):
        """Test that modules cannot be serialized safely."""
        import os

        with pytest.raises(SerializationError):
            SafeSerializer.dumps(os, allow_pickle=False)


class TestTypeCoverage:
    """Test all supported types comprehensively."""

    def test_all_datetime_types(self):
        """Test all datetime module types."""
        from datetime import date, datetime, time

        test_cases = [
            datetime(2024, 1, 1, 12, 30, 45),
            date(2024, 1, 1),
            time(12, 30, 45),
            timedelta(days=5, hours=3, minutes=30, seconds=15),
            datetime.min,
            datetime.max,
            date.min,
            date.max,
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_decimal_precision(self):
        """Test Decimal type with various precisions."""
        from decimal import getcontext

        # Set high precision
        getcontext().prec = 50

        test_cases = [
            Decimal("3.14159265358979323846264338327950288419716939937510"),
            Decimal("0.1") + Decimal("0.2"),  # Famous float precision issue
            Decimal("1e-100"),
            Decimal("1e100"),
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_pathlib_types(self):
        """Test various Path types."""

        test_cases = [
            Path("/tmp/test.txt"),
            Path("relative/path/file.py"),
            Path("/"),
            Path("."),
            Path(".."),
            Path("/path/with spaces/file.txt"),
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_complex_numbers(self):
        """Test complex number handling."""
        test_cases = [
            complex(1, 2),
            complex(0, 0),
            complex(-5, 10),
            complex(3.14, 2.71),
            1 + 2j,
            -5 + 10j,
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj

    def test_bytes_types(self):
        """Test various bytes objects."""
        test_cases = [
            b"hello",
            b"\x00\x01\x02\x03",
            b"Binary\xff\xfe\xfd",
            bytes(range(256)),  # All byte values
            b"",  # Empty bytes
        ]

        for obj in test_cases:
            serialized = SafeSerializer.dumps(obj, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == obj


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

    def test_numpy_various_dtypes(self):
        """Test numpy arrays with various dtypes."""
        pytest.importorskip("numpy")
        import numpy as np

        # Test numeric dtypes (non-complex)
        dtypes = [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
            np.bool_,
        ]

        for dtype in dtypes:
            arr = np.array([1, 2, 3], dtype=dtype)
            serialized = SafeSerializer.dumps(arr, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            np.testing.assert_array_equal(result, arr)
            assert result.dtype == arr.dtype

        # Complex dtypes need special handling - test separately
        # Note: numpy complex arrays are not fully supported in safe mode
        # as they require custom complex number serialization

    def test_numpy_multidimensional(self):
        """Test multidimensional numpy arrays."""
        pytest.importorskip("numpy")
        import numpy as np

        test_cases = [
            np.array([[1, 2], [3, 4]]),  # 2D
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D
            np.zeros((10, 10, 10)),  # Large 3D
            np.ones((5, 5)),  # 2D ones
        ]

        for arr in test_cases:
            serialized = SafeSerializer.dumps(arr, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            np.testing.assert_array_equal(result, arr)

    def test_numpy_empty_array(self):
        """Test empty numpy array."""
        pytest.importorskip("numpy")
        import numpy as np

        arr = np.array([])
        serialized = SafeSerializer.dumps(arr, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        np.testing.assert_array_equal(result, arr)


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

    def test_pil_various_modes(self):
        """Test PIL images in various modes."""
        pytest.importorskip("PIL")
        from PIL import Image

        modes = ["RGB", "RGBA", "L", "1"]  # Color, Alpha, Grayscale, Binary

        for mode in modes:
            img = Image.new(mode, (10, 10), color="red" if mode != "1" else 1)
            serialized = SafeSerializer.dumps(img, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)

            assert isinstance(result, Image.Image)
            assert result.mode == img.mode
            assert result.size == img.size

    def test_pil_various_sizes(self):
        """Test PIL images of various sizes."""
        pytest.importorskip("PIL")
        from PIL import Image

        sizes = [(1, 1), (100, 100), (500, 300)]

        for size in sizes:
            img = Image.new("RGB", size, color="blue")
            serialized = SafeSerializer.dumps(img, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)

            assert result.size == img.size


class TestDataclasses:
    """Test dataclass serialization."""

    def test_simple_dataclass(self):
        """Test simple dataclass."""
        from dataclasses import dataclass

        @dataclass
        class Person:
            name: str
            age: int

        person = Person(name="Alice", age=30)
        serialized = SafeSerializer.dumps(person, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        # Result is a dict representation
        assert result["__dataclass__"] == "Person"
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_nested_dataclass(self):
        """Test nested dataclasses."""
        from dataclasses import dataclass

        @dataclass
        class Address:
            street: str
            city: str

        @dataclass
        class Person:
            name: str
            address: Address

        person = Person(name="Bob", address=Address(street="123 Main St", city="NYC"))
        serialized = SafeSerializer.dumps(person, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        assert result["name"] == "Bob"
        assert result["address"]["street"] == "123 Main St"


class TestPerformance:
    """Performance tests for large data."""

    def test_large_list(self):
        """Test serialization of large list."""
        large_list = list(range(100_000))

        serialized = SafeSerializer.dumps(large_list, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        assert result == large_list

    def test_large_dict(self):
        """Test serialization of large dictionary."""
        large_dict = {f"key_{i}": i for i in range(10_000)}

        serialized = SafeSerializer.dumps(large_dict, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        assert result == large_dict

    def test_deeply_nested_performance(self):
        """Test performance with deeply nested structures."""
        obj = {"level": 0}
        current = obj
        for i in range(1, 100):  # 100 levels (avoid recursion limit)
            current["nested"] = {"level": i}
            current = current["nested"]

        serialized = SafeSerializer.dumps(obj, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        assert result == obj


class TestPrefixHandling:
    """Test handling of different prefix formats."""

    def test_safe_prefix_detection(self):
        """Test detection of safe: prefix."""
        obj = {"test": "data"}
        serialized = SafeSerializer.dumps(obj, allow_pickle=False)

        assert serialized.startswith("safe:")
        result = SafeSerializer.loads(serialized, allow_pickle=False)
        assert result == obj

    def test_pickle_prefix_with_allow_pickle(self):
        """Test pickle: prefix when pickle is allowed."""
        # Create an object that needs pickle
        obj = PicklableCustomClass()
        serialized = SafeSerializer.dumps(obj, allow_pickle=True)

        # Should have pickle prefix
        assert serialized.startswith("pickle:")

        result = SafeSerializer.loads(serialized, allow_pickle=True)
        assert result.value == 42

    def test_legacy_format_detection(self):
        """Test detection and handling of legacy format (no prefix)."""
        # Simulate legacy pickle data (no prefix)
        legacy_data = {"key": "value"}
        legacy_encoded = base64.b64encode(pickle.dumps(legacy_data)).decode()

        # Should work with allow_pickle=True
        result = SafeSerializer.loads(legacy_encoded, allow_pickle=True)
        assert result == legacy_data


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_agent_variables_scenario(self):
        """Test typical agent variables scenario."""
        import numpy as np
        from PIL import Image

        # Typical variables an agent might use
        variables = {
            "search_results": ["result1", "result2", "result3"],
            "config": {
                "temperature": 0.7,
                "max_tokens": 100,
                "model": "gpt-4",
            },
            "data_array": np.array([1.0, 2.0, 3.0]),
            "image": Image.new("RGB", (50, 50)),
            "timestamp": datetime.now(),
            "status": "running",
            "counter": 42,
        }

        serialized = SafeSerializer.dumps(variables, allow_pickle=False)
        result = SafeSerializer.loads(serialized, allow_pickle=False)

        assert result["search_results"] == variables["search_results"]
        assert result["config"] == variables["config"]
        assert result["status"] == variables["status"]
        assert result["counter"] == variables["counter"]

    def test_final_answer_scenario(self):
        """Test typical final answer serialization."""
        final_answers = [
            "Simple string answer",
            {"answer": "structured", "confidence": 0.95},
            ["multiple", "results", "returned"],
            42,
            3.14159,
            True,
        ]

        for answer in final_answers:
            serialized = SafeSerializer.dumps(answer, allow_pickle=False)
            result = SafeSerializer.loads(serialized, allow_pickle=False)
            assert result == answer


class TestGeneratedDeserializerCode:
    """Regression tests for generated deserializer code used by remote executors."""

    def test_generated_deserializer_executes_for_safe_payload(self):
        code = SafeSerializer.get_deserializer_code(allow_pickle=False)
        namespace = {}
        exec(code, namespace, namespace)

        payload = SafeSerializer.dumps(
            {
                "count": 3,
                "items": (1, 2, 3),
                "raw": b"bytes",
            },
            allow_pickle=False,
        )
        result = namespace["_deserialize"](payload)
        assert result == {"count": 3, "items": (1, 2, 3), "raw": b"bytes"}

    def test_generated_deserializer_handles_pickle_prefix_when_enabled(self):
        code = SafeSerializer.get_deserializer_code(allow_pickle=True)
        namespace = {}
        exec(code, namespace, namespace)

        payload = "pickle:" + base64.b64encode(pickle.dumps({"hello": "world"})).decode()
        result = namespace["_deserialize"](payload)
        assert result == {"hello": "world"}


class TestConcurrency:
    """Test thread safety and concurrent access."""

    def test_concurrent_serialization(self):
        """Test concurrent serialization operations."""
        import threading

        results = []
        errors = []

        def serialize_data(data, index):
            try:
                serialized = SafeSerializer.dumps(data, allow_pickle=False)
                deserialized = SafeSerializer.loads(serialized, allow_pickle=False)
                results.append((index, deserialized == data))
            except Exception as e:
                errors.append((index, e))

        threads = []
        test_data = [{"thread": i, "data": list(range(100))} for i in range(10)]

        for i, data in enumerate(test_data):
            thread = threading.Thread(target=serialize_data, args=(data, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(success for _, success in results)
