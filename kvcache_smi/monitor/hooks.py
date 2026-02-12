from __future__ import annotations

from collections.abc import Iterable


class HookRegistry:
    def __init__(self) -> None:
        self._handles = []

    def register_forward_hooks(self, modules: Iterable, hook_fn):
        for module in modules:
            handle = module.register_forward_hook(hook_fn)
            self._handles.append(handle)

    def unregister_all(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
