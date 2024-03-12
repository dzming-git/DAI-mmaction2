import os
from contextlib import contextmanager

# 定义一个上下文管理器，用于临时改变当前工作目录
@contextmanager
def temporary_change_dir(path):
    current_dir = os.getcwd()  # 获取当前工作目录
    os.chdir(path)  # 改变当前工作目录到指定路径
    try:
        yield
    finally:
        os.chdir(current_dir)  # 恢复原工作目录
