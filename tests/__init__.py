# 显式声明 tests 为常规包：环境里存在同名 site-packages/tests 包，
# 常规包优先级高于隐式命名空间包，缺此文件会导致 `tests.local_eval.*` 导入失败。
