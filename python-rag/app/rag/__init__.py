"""Lightweight package marker for the medical RAG modules.

The active evaluation entrypoints import concrete submodules directly. Keeping the
package initializer empty avoids eagerly importing retired API-era modules that are
not part of the current evaluation call chain.
"""

__all__: list[str] = []
