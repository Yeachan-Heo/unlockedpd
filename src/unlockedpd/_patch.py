"""Patch infrastructure for unlockedpd.

This module provides the mechanism for patching pandas methods with
optimized implementations while maintaining fallback to originals.
"""
from typing import Callable, Dict, Tuple, Any, Optional
from contextlib import contextmanager
import functools
import warnings


class _PatchRegistry:
    """Registry tracking all applied patches.

    This class manages the monkey-patching of pandas methods with
    optimized versions, while storing originals for fallback/restoration.
    """
    _patches: Dict[Tuple[type, str], Callable] = {}
    _originals: Dict[Tuple[type, str], Callable] = {}

    @classmethod
    def patch(
        cls,
        target_class: type,
        method_name: str,
        optimized_func: Callable,
        fallback: bool = True
    ) -> None:
        """Apply a patch to a method on a target class.

        Args:
            target_class: The class to patch (e.g., pd.core.window.rolling.Rolling)
            method_name: Name of the method to patch (e.g., 'mean')
            optimized_func: The optimized implementation
            fallback: If True, wrap with try/except to fall back to original on error
        """
        key = (target_class, method_name)

        # Store original if not already stored
        if key not in cls._originals:
            original = getattr(target_class, method_name)
            cls._originals[key] = original
        else:
            original = cls._originals[key]

        if fallback:
            @functools.wraps(original)
            def wrapper(self, *args, **kwargs):
                from ._config import config

                # If disabled, use original directly
                if not config.enabled:
                    return original(self, *args, **kwargs)

                try:
                    return optimized_func(self, *args, **kwargs)
                except Exception as e:
                    if config.warn_on_fallback:
                        warnings.warn(
                            f"unlockedpd: Falling back to pandas for {method_name}: {e}",
                            RuntimeWarning,
                            stacklevel=2
                        )
                    return original(self, *args, **kwargs)
            replacement = wrapper
        else:
            replacement = optimized_func

        setattr(target_class, method_name, replacement)
        cls._patches[key] = replacement

    @classmethod
    def unpatch(cls, target_class: type, method_name: str) -> None:
        """Remove a patch and restore the original method.

        Args:
            target_class: The class that was patched
            method_name: Name of the method to restore
        """
        key = (target_class, method_name)
        if key in cls._originals:
            setattr(target_class, method_name, cls._originals[key])
            cls._patches.pop(key, None)
            cls._originals.pop(key, None)

    @classmethod
    def unpatch_all(cls) -> None:
        """Remove all patches and restore all original methods."""
        for (target_class, method_name), original in list(cls._originals.items()):
            setattr(target_class, method_name, original)
        cls._patches.clear()
        cls._originals.clear()

    @classmethod
    def is_patched(cls, target_class: type, method_name: str) -> bool:
        """Check if a method is currently patched.

        Args:
            target_class: The class to check
            method_name: Name of the method

        Returns:
            True if the method is patched, False otherwise
        """
        return (target_class, method_name) in cls._patches

    @classmethod
    def get_original(cls, target_class: type, method_name: str) -> Optional[Callable]:
        """Get the original (unpatched) method.

        Args:
            target_class: The class
            method_name: Name of the method

        Returns:
            The original method, or None if not patched
        """
        return cls._originals.get((target_class, method_name))

    @classmethod
    @contextmanager
    def temporarily_unpatched(cls):
        """Context manager to temporarily disable all patches.

        Usage:
            with _PatchRegistry.temporarily_unpatched():
                # Original pandas methods are used here
                result = df.rolling(5).mean()
        """
        from ._config import config
        original_enabled = config.enabled
        config.enabled = False
        try:
            yield
        finally:
            config.enabled = original_enabled


# Convenience functions for module-level access
def patch(target_class: type, method_name: str, optimized_func: Callable, fallback: bool = True) -> None:
    """Apply a patch to a method. See _PatchRegistry.patch for details."""
    _PatchRegistry.patch(target_class, method_name, optimized_func, fallback)


def unpatch(target_class: type, method_name: str) -> None:
    """Remove a patch. See _PatchRegistry.unpatch for details."""
    _PatchRegistry.unpatch(target_class, method_name)


def unpatch_all() -> None:
    """Remove all patches. See _PatchRegistry.unpatch_all for details."""
    _PatchRegistry.unpatch_all()


def is_patched(target_class: type, method_name: str) -> bool:
    """Check if patched. See _PatchRegistry.is_patched for details."""
    return _PatchRegistry.is_patched(target_class, method_name)
