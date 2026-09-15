# @author: ztwz
"""Embedding 客户端：bge-small-zh-v1.5（中英混合，CPU 可跑）。

sentence-transformers 懒加载；模型优先 HF 本地缓存，缺失时 modelscope 兜底下载。
"""
from __future__ import annotations

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class EmbeddingClient:
    """encode(texts) 返回 L2 归一化后的向量矩阵 (n, dim)"""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model(self):
        """懒加载模型：HF 本地缓存优先，缺失时 modelscope 下载"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("需要安装 sentence-transformers：pip install sentence-transformers") from e
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception:
            # HF 下载失败兜底：modelscope 拉取后按本地路径加载
            from modelscope import snapshot_download
            local_path = snapshot_download(self.model_name)
            self._model = SentenceTransformer(local_path, device=self.device)

    def encode(self, texts: list[str]):
        """批量编码并 L2 归一化，返回 np.ndarray (n, dim)"""
        import numpy as np

        self._ensure_model()
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
