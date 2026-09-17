# @author: ztwz
"""Embedding 客户端：默认 bge-small-en-v1.5（英文语料，CPU 可跑）。

语料为纯英文，故默认改用英文优化模型；与旧的 bge-small-zh-v1.5 同族、同 384 维、
同 L2 归一化约定，是真正的 drop-in，下游无需任何维度适配。
中文语料可用环境变量 MODEL_EVAL_EMBEDDING_MODEL 切回 BAAI/bge-small-zh-v1.5；
中英混合且需兼顾则换多语言模型（如 BAAI/bge-m3）。
注意：换模型后 cosine 分布会变，embedding_scorer 的相似度阈值必须同步重标定。

sentence-transformers 懒加载；模型优先 HF 本地缓存，缺失时 modelscope 兜底下载。
"""
from __future__ import annotations

import os

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# 覆盖模型名的环境变量（沿用例：MODEL_EVAL_ 前缀）
EMBEDDING_MODEL_ENV = "MODEL_EVAL_EMBEDDING_MODEL"


class EmbeddingClient:
    """encode(texts) 返回 L2 归一化后的向量矩阵 (n, dim)"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        # 显式入参 > 环境变量 > 内置默认（en）
        self.model_name = model_name or os.environ.get(EMBEDDING_MODEL_ENV) or DEFAULT_EMBEDDING_MODEL
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
