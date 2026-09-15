# @author: ztwz
"""业务异常定义。"""


class BusinessError(Exception):
    """业务异常：message 可直接展示给用户，status_code 默认 400"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code