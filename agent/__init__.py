__all__ = ["CXRReActAgent"]


def __getattr__(name):
    if name == "CXRReActAgent":
        from agent.react_agent import CXRReActAgent
        return CXRReActAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
