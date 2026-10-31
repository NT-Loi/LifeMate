__all__ = ["LifeMateAgent"]


def __getattr__(name):
    if name == "LifeMateAgent":
        from .lifemate import LifeMateAgent

        return LifeMateAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
