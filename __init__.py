"""
LlamaAgent – Advanced Multi-Agent AI Framework with SPRE (Strategic Planning & Resourceful Execution)

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path

# ---------------------------------------------------------------------------
# Ensure that the *types* submodule is registered *before* we import the heavy
# implementation package.  This allows deeply nested modules (e.g. provider
# implementations) that import ``llamaagent.types`` at import-time to succeed
# without encountering a circular dependency.
# ---------------------------------------------------------------------------

try:
    _types_mod = _importlib.import_module("src.llamaagent.types")
    _sys.modules[__name__ + ".types"] = _types_mod
except ModuleNotFoundError:
    # In minimal builds the canonical implementation might be missing.  We
    # ignore the error because most unit-tests stub out the missing parts.
    pass

# ---------------------------------------------------------------------------
# Expose full implementation package under the current namespace             #
# ---------------------------------------------------------------------------

# Dynamically import the actual implementation which resides in the *src*
# folder so that external imports like `import llamaagent.benchmarks` work
# transparently.

_impl_pkg = _importlib.import_module("src.llamaagent")

# Re-export public attributes of the real package to the stub module.  We
# intentionally keep existing identifiers (like *llm*) that we add below.
for _attr in dir(_impl_pkg):
    if not _attr.startswith("__"):
        setattr(_sys.modules[__name__], _attr, getattr(_impl_pkg, _attr))

# Register submodules so that `llamaagent.<sub>` resolves via *sys.modules*.
for _sub_name, _sub_module in list(_sys.modules.items()):
    if _sub_name.startswith("src.llamaagent."):
        short_name = _sub_name.replace("src.llamaagent.", __name__ + ".")
        _sys.modules[short_name] = _sub_module

# Finally, make sure the *llm* convenience attribute is available.
_llm_mod = _importlib.import_module("llm")
llm = _llm_mod  # type: ignore
_sys.modules[__name__ + ".llm"] = _llm_mod

# ---------------------------------------------------------------------------
# Import and expose common classes at package level                          #
# ---------------------------------------------------------------------------

# Import agents and related classes
try:
    from src.llamaagent.agents import AgentConfig, AgentResponse, AgentRole, BaseAgent, ReactAgent
    # For backward compatibility, expose ReactAgent as Agent since it's the main implementation
    Agent = ReactAgent
except ImportError:
    # Fallback if import fails
    Agent = None
    ReactAgent = None
    BaseAgent = None
    AgentConfig = None
    AgentRole = None
    AgentResponse = None

# Import tools
try:
    from src.llamaagent.tools import ToolRegistry, get_all_tools
except ImportError:
    ToolRegistry = None
    get_all_tools = None

# Import LLM factory
try:
    from src.llamaagent.llm.factory import ProviderFactory
except ImportError:
    ProviderFactory = None

# ---------------------------------------------------------------------------
# Lazy attribute loader for submodules                                       #
# ---------------------------------------------------------------------------


def __getattr__(attr: str):  # noqa: D401 – simple passthrough
    """Dynamically load nested modules from the implementation package."""
    try:
        module = _importlib.import_module(f"src.llamaagent.{attr}")
        _sys.modules[f"{__name__}.{attr}"] = module
        return module
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise AttributeError(attr) from exc


# ---------------------------------------------------------------------------
# Eagerly register frequently accessed subpackages                           #
# ---------------------------------------------------------------------------

for _pkg in ("benchmarks", "agents", "config", "tools", "memory", "llm"):
    try:
        _loaded = _importlib.import_module(f"src.llamaagent.{_pkg}")
        _sys.modules[f"{__name__}.{_pkg}"] = _loaded
    except ModuleNotFoundError:
        # Some optional subpackages may be absent in stripped-down test builds
        continue

# ---------------------------------------------------------------------------
# Ensure local *llamaagent* sub-directory and *src/llamaagent* are on __path__
# so that traditional imports like ``import llamaagent.types`` work even though
# this stub package lives in the project root.
# ---------------------------------------------------------------------------

_root = _Path(__file__).resolve().parent
_candidate_paths = [
    _root / "llamaagent",          # project_root/llamaagent
    _root / "src" / "llamaagent",  # project_root/src/llamaagent
]

for _p in _candidate_paths:
    if _p.is_dir() and str(_p) not in __path__:
        __path__.append(str(_p))

# ---------------------------------------------------------------------------
# Explicitly register the *types* submodule so that eager imports like
# ``from llamaagent.types import ...`` succeed even if the module search path
# resolution occurs before lazy loading.
# ---------------------------------------------------------------------------

try:
    _types_mod = _importlib.import_module("src.llamaagent.types")
    _sys.modules[__name__ + ".types"] = _types_mod
except ModuleNotFoundError:
    # The canonical implementation may be absent in minimal builds – that's OK.
    pass

# Package metadata
__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

__all__ = [
    "Agent",
    "ReactAgent", 
    "BaseAgent",
    "AgentConfig",
    "AgentRole", 
    "AgentResponse",
    "ToolRegistry",
    "get_all_tools",
    "ProviderFactory",
    "llm"
]
