"""Tests for transform and apply operations."""
import pytest
import pandas as pd
import numpy as np


class TestApply:
    """Tests for apply() operation"""

    @pytest.mark.skip(reason="Apply operations implementation needs verification")
    def test_basic_apply_axis0(self):
        """Test basic apply along axis 0 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        def simple_func(x):
            return x * 2

        unlockedpd.config.enabled = False
        expected = df.apply(simple_func, axis=0)

        unlockedpd.config.enabled = True
        result = df.apply(simple_func, axis=0)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Apply operations implementation needs verification")
    def test_basic_apply_axis1(self):
        """Test basic apply along axis 1 matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        def simple_func(x):
            return x * 2

        unlockedpd.config.enabled = False
        expected = df.apply(simple_func, axis=1)

        unlockedpd.config.enabled = True
        result = df.apply(simple_func, axis=1)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Apply operations implementation needs verification")
    def test_apply_numpy_function(self):
        """Test apply with numpy function."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.apply(np.sqrt, axis=0)

        unlockedpd.config.enabled = True
        result = df.apply(np.sqrt, axis=0)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Apply operations implementation needs verification")
    def test_apply_lambda(self):
        """Test apply with lambda function."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.apply(lambda x: x ** 2, axis=0)

        unlockedpd.config.enabled = True
        result = df.apply(lambda x: x ** 2, axis=0)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestTransform:
    """Tests for transform() operation"""

    @pytest.mark.skip(reason="Transform operations not yet implemented")
    def test_basic_transform(self):
        """Test basic transform matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.transform(lambda x: x * 2)

        unlockedpd.config.enabled = True
        result = df.transform(lambda x: x * 2)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Transform operations not yet implemented")
    def test_transform_multiple_functions(self):
        """Test transform with multiple functions."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.transform([np.sqrt, np.exp])

        unlockedpd.config.enabled = True
        result = df.transform([np.sqrt, np.exp])

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Transform operations not yet implemented")
    def test_transform_dict(self):
        """Test transform with dict of functions."""
        import unlockedpd

        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100)
        })

        unlockedpd.config.enabled = False
        expected = df.transform({'a': np.sqrt, 'b': np.exp, 'c': lambda x: x ** 2})

        unlockedpd.config.enabled = True
        result = df.transform({'a': np.sqrt, 'b': np.exp, 'c': lambda x: x ** 2})

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestPipe:
    """Tests for pipe() operation"""

    @pytest.mark.skip(reason="Pipe operations not yet implemented")
    def test_basic_pipe(self):
        """Test basic pipe matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        def add_one(data):
            return data + 1

        unlockedpd.config.enabled = False
        expected = df.pipe(add_one)

        unlockedpd.config.enabled = True
        result = df.pipe(add_one)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Pipe operations not yet implemented")
    def test_pipe_chaining(self):
        """Test pipe with function chaining."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        def add_one(data):
            return data + 1

        def multiply_two(data):
            return data * 2

        unlockedpd.config.enabled = False
        expected = df.pipe(add_one).pipe(multiply_two)

        unlockedpd.config.enabled = True
        result = df.pipe(add_one).pipe(multiply_two)

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)


class TestAgg:
    """Tests for agg/aggregate() operation"""

    @pytest.mark.skip(reason="Agg operations not yet implemented")
    def test_basic_agg(self):
        """Test basic agg matches pandas."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.agg('sum')

        unlockedpd.config.enabled = True
        result = df.agg('sum')

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Agg operations not yet implemented")
    def test_agg_multiple_functions(self):
        """Test agg with multiple functions."""
        import unlockedpd

        df = pd.DataFrame(np.random.randn(100, 10))

        unlockedpd.config.enabled = False
        expected = df.agg(['sum', 'mean', 'std'])

        unlockedpd.config.enabled = True
        result = df.agg(['sum', 'mean', 'std'])

        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    @pytest.mark.skip(reason="Agg operations not yet implemented")
    def test_agg_dict(self):
        """Test agg with dict of functions per column."""
        import unlockedpd

        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100)
        })

        unlockedpd.config.enabled = False
        expected = df.agg({'a': 'sum', 'b': 'mean', 'c': 'std'})

        unlockedpd.config.enabled = True
        result = df.agg({'a': 'sum', 'b': 'mean', 'c': 'std'})

        pd.testing.assert_series_equal(result, expected, rtol=1e-10)
