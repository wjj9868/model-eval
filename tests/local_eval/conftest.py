# @author: ztwz
"""local_eval 纯逻辑测试：覆盖上层 conftest 的数据库 autouse fixture，不依赖 MySQL。"""
import pytest

from tests.local_eval.fakes import MockEmbedder, make_prompt, make_teacher_output


@pytest.fixture(scope="session", autouse=True)
def _prepare_database():
    """同名覆盖上层 conftest 的建库逻辑：本目录纯逻辑测试无需数据库"""
    yield


@pytest.fixture(autouse=True)
def _clean_tables():
    """同名覆盖上层 conftest 的建表清表逻辑：本目录纯逻辑测试无需数据库"""
    yield


@pytest.fixture()
def mock_embedder():
    """无别名对的确定性 embedder"""
    return MockEmbedder()


@pytest.fixture()
def teacher_factory():
    """teacher 输出构造工厂"""
    return make_teacher_output


@pytest.fixture()
def prompt_factory():
    """v3 英文模板 prompt 构造工厂"""
    return make_prompt
