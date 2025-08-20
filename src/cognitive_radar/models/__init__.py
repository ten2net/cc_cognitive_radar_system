TARGET_REGISTRY = {}  # 插件式架构设计 目标注册表
def register_target(name, cls):
    TARGET_REGISTRY[name] = cls

def create_target(name, *args, **kwargs):
    if name not in TARGET_REGISTRY:
        raise ValueError(f"未知目标类型: {name}")
    return TARGET_REGISTRY[name](*args, **kwargs)