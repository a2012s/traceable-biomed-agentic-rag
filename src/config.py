"""全局配置：环境变量 -> 强类型设置对象。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"
    strong_model: str = ""
    cheap_model: str = ""

    # Embedding / 向量库：hash(离线兜底) | tfidf(离线真向量) | api(真 embedding 模型)
    embedding_backend: str = "tfidf"
    embedding_model: str = "text-embedding-v3"
    # qdrant(本地 HNSW) | numpy(内存暴力检索)
    vector_store: str = "qdrant"
    # 默认:memory: —— 仍是真实 HNSW 索引，但无文件锁、无残留 sqlite 句柄；
    # 需要跨进程复用索引时设为 .qdrant_data 之类的路径（Qdrant 嵌入式持久化）
    qdrant_path: str = ":memory:"

    rerank_top_k: int = 5
    retrieve_top_k: int = 50
    # 送入重排的候选上限：Top-50 与 Top-20 重排延迟差数倍，接入真 cross-encoder 后尤其明显
    rerank_candidates: int = 20
    # 跨域融合后送入生成节点的证据条数：太少丢多源证据，太多稀释注意力
    evidence_top_k: int = 8
    # 融合截断：保留最佳分数的 fuse_margin 倍以上，但至少留 fuse_min_keep 条。
    # 缺保底时，最佳项是离群值会把其余高质量证据全部丢掉，只剩 1 条导致 Critic
    # 的「独立数据源 ≥2」必然不成立 → 跑满反思仍拒答。
    # 默认 2：消除"只剩 1 条"的结构性拒答，且不额外引入低分证据；
    # 调大能进一步兜底多源要求，但会把边缘证据送进 prompt，按语料实测取舍。
    fuse_margin: float = 0.5
    fuse_min_keep: int = 2
    # RRF 平滑常数：越大越弱化名次差异，60 是工业界常用默认值
    rrf_k: int = 60
    max_reflections: int = 2
    # 单请求 token 预算：超过即停止反思，避免证据枯竭的查询空转烧钱
    max_tokens_per_request: int = 20000
    # 召回优先：默认全量并行召回四个数据域，编排器只在明确无关时裁剪。
    # 设为 0 则完全由编排器按领域裁剪 —— 实测会掉召回（见 README 的消融数据）。
    wide_recall: int = 1
    # 1=额外打印决策链实时日志（stdout 已有一份表格，默认关）
    verbose: int = 0

    # ---- 生产化配置（默认值保持离线 Demo 行为不变）----
    # 检查点后端：memory(默认) | sqlite(需装 langgraph-checkpoint-sqlite)
    checkpointer: str = "memory"
    checkpoint_db: str = ".agentic_rag.db"
    # Web 层并发：同时执行的问答数与单请求超时（秒）
    max_concurrency: int = 4
    request_timeout_s: float = 120.0
    # 答案缓存 TTL（秒）；0 关闭。键含规范化问题 + 语料版本 + 检索配置
    answer_cache_ttl_s: int = 300
    # 外部 cross-encoder 重排端点；留空则用内置 IDF 加权余弦
    reranker_api_url: str = ""
    reranker_api_key: str = ""
    # 单次外部重排请求超时（秒）与熔断冷却（秒）
    reranker_timeout_s: float = 3.0
    reranker_breaker_cooldown_s: float = 60.0
    # api embedding 批大小：一次送全部文本会撞上单请求条数/字节上限
    embedding_batch_size: int = 64
    # 1=embedding API 失败直接报错，不静默降级（生产建议开，避免质量悄悄劣化）
    embedding_strict: int = 0
    # 1=允许向量集合维度变化时自动重建；默认 0，避免改一行配置就静默删索引
    allow_reindex: int = 0
    # 1=启用实体链接消歧（默认开，纯规则无网络）
    entity_linking: int = 1
    # 1=启用 LLM 自主工具选择（Agent 核心能力：模型看 schema 决定调哪个工具、传什么参数）
    tool_use_enabled: int = 1
    # 单轮最多接受几个工具调用：防止模型一次塞十几个调用把延迟和上下文打爆
    tool_use_max_calls: int = 3
    # 1=启用 Human-in-the-loop：Critic 判"证据不足且还能再检索"时中断，等人工决定后
    # 再续跑（继续生成 / 按补充说明再检索 / 中止拒答）。
    # 默认关：eval.py 与批处理是无人值守的，中断会让进程停在等人这一步。
    hitl_enabled: int = 0
    # 单次请求最多中断几次：>1 会让用户被反复弹窗打断，默认只问一次
    hitl_max_reviews: int = 1
    # 反馈闭环：默认关闭；开启后写本地 SQLite，生产多实例替换为共享 DB
    feedback_enabled: int = 0
    feedback_db: str = ".agentic_rag_feedback.db"

    @property
    def is_mock(self) -> bool:
        return self.llm_provider.lower() == "mock" or not self.active_api_key

    @property
    def active_api_key(self) -> str:
        return {
            "openai": self.openai_api_key,
            "deepseek": self.deepseek_api_key,
            "dashscope": self.dashscope_api_key,
        }.get(self.llm_provider.lower(), "")

    @property
    def active_base_url(self) -> str:
        return {
            "openai": self.openai_base_url,
            "deepseek": self.deepseek_base_url,
            "dashscope": self.dashscope_base_url,
        }.get(self.llm_provider.lower(), self.openai_base_url)

    @property
    def active_model(self) -> str:
        return {
            "openai": self.openai_model,
            "deepseek": self.deepseek_model,
            "dashscope": self.dashscope_model,
        }.get(self.llm_provider.lower(), self.openai_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
