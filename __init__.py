from .nodes import NODE_CLASS_MAPPINGS as _n1, NODE_DISPLAY_NAME_MAPPINGS as _d1

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = dict(_n1), dict(_d1)

try:                                    # PixCell nodes need diffusers; the rest works without it
    from .pixcell_nodes import NODE_CLASS_MAPPINGS as _n2, NODE_DISPLAY_NAME_MAPPINGS as _d2
    NODE_CLASS_MAPPINGS.update(_n2)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d2)
except ImportError as e:
    print(f"[SyncTweedies] PixCell nodes disabled: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
