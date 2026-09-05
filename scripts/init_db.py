"""
数据库初始化脚本
创建 MySQL 表结构
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.keyword_store import get_keyword_store


def init_database():
    """初始化数据库表结构"""
    store = get_keyword_store()
    store.connect()
    store._create_tables()
    store.close()
    print("Database initialized successfully")


if __name__ == "__main__":
    init_database()
