import yaml
import re

class ScientificFloatLoader(yaml.SafeLoader):
    """优化版YAML加载器，优雅处理科学计数法"""
    def __init__(self, stream):
        super().__init__(stream)
        # 添加自定义类型解析
        self.add_implicit_resolver('!sci_float', re.compile(r'^\d*\.?\d+[eE][-+]?\d+$'), None)
        self.add_constructor('!sci_float', self.construct_sci_float)
    
    def construct_sci_float(self, loader, node):
        """科学计数法转换为浮点数"""
        return float(node.value)