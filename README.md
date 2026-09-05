------

# CloudDesk RAG 企业 SaaS 智能客服系统

基于 FastAPI + Milvus + MySQL + LLM 构建的企业 SaaS 智能客服系统，实现 Query Rewrite、Hybrid Retrieval（向量 + 关键词）、RRF 融合排序、可插拔 Rerank 和 LLM 问答生成的完整 RAG 链路，并配有离线 Evaluation 框架用于检索策略对比实验。

**核心标签**：Python · FastAPI · Milvus · MySQL · DeepSeek · RAG · Hybrid Retrieval · RRF · BGE Reranker · Docker

------

## 📖 目录

1. [这个项目是什么？](#-这个项目是什么)
2. [解决了什么问题？](#-解决了什么问题？)
3. [完整RAG流程](#-完整RAG流程)
4. [评估体系](#-评估体系)
5. [系统架构](#-系统架构)
6. [关键技术详解](#-关键技术详解)
7. [快速上手](#-快速上手)
8. [API接口文档](#-API接口文档)
9. [项目文件结构](#-项目文件结构)
10. [技术栈总览](#-技术栈总览)

------

## 🤔 这个项目是什么？

### 用一句话解释

> CloudDesk 是一个面向企业 SaaS 产品的智能客服问答系统。用户提出售后或使用问题后，系统从产品知识库中检索相关内容，由 LLM 生成有依据的回答，并返回引用来源。

不同于纯 LLM 回答容易产生幻觉、纯关键词匹配无法理解语义，本项目采用 **RAG（检索增强生成）** 架构，通过 **Query Rewrite、Hybrid Retrieval（向量 + 关键词 + RRF 融合）、BGE CrossEncoder Rerank 和 LLM 生成**的完整二阶段检索排序链路，在检索和生成之间找到平衡——先用检索保证知识覆盖，再通过 Reranker 提升排序准确性，最后由 LLM 基于检索上下文生成可溯源回答。

------

## 🎯 解决了什么问题？

传统客服问答系统主要存在以下问题：

| 问题               | 传统方案                       | 本项目方案                                                   | 验证指标                 |
| :----------------- | :----------------------------- | :----------------------------------------------------------- | :----------------------- |
| 🎯 语义理解不足     | 纯关键词匹配，无法理解同义表达 | **Query Rewrite**：LLM 将口语化查询重写为文档风格            | Recall@K、MRR@K          |
| 🔍 召回覆盖不足     | 单一向量检索容易遗漏精确术语   | **Hybrid Retrieval**：向量语义检索 + MySQL 关键词检索，RRF 融合 | Recall@K、Precision@K    |
| 📊 排序不准确       | 检索结果直接按相似度排序       | **BGE CrossEncoder Rerank**：对 Hybrid+RRF 候选进行二阶段精排 | MRR@K、Precision@K       |
| 🤖 LLM 容易幻觉     | 直接让 LLM 回答，无知识约束    | **RAG 生成**：LLM 基于检索到的知识生成回答，可溯源           | 答案准确率               |
| 📈 无法评估检索效果 | 手工测试，无量化指标           | **离线 Evaluation**：4 种策略在同一数据集上公平对比          | Hit/Recall/Precision/MRR |

------

## 📊 完整RAG流程

text

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Step 1: Query Rewrite                         │
│  口语化查询 → LLM 重写 → 更富含关键词的查询文本   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Step 2: Query Router                          │
│  分类查询 → category（用于后续可选过滤）          │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│  Step 3a:        │   │  Step 3b:        │
│  Vector Search   │   │  Keyword Search  │
│  → Milvus        │   │  → MySQL         │
│  → BGE-M3 1024D  │   │  → 2-gram 中文    │
│  → HNSW + IP     │   │  → BM25 评分     │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         └───────────┬──────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  Step 4: RRF Fusion                            │
│  score(d) = Σ 1/(k + rank)，k=60               │
│  不依赖各检索器原始分数量纲                      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Step 5: BGE Rerank                            │
│  BAAI/bge-reranker-v2-m3 CrossEncoder 精排     │
│  对 RRF Top-K 候选进行 query-document 重新排序  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Step 6: Context Compression                   │
│  按分数排序，截断至 3000 字符                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Step 7: LLM Generation                        │
│  DeepSeek → 基于 context 生成回答 + 引用来源    │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
              Answer + Sources
```



**每步对应的代码文件**：

| 步骤                   | 代码文件                                                     |
| :--------------------- | :----------------------------------------------------------- |
| Query Rewrite / Router | `app/rag/query_rewrite.py`                                   |
| Vector Search          | `app/retrievers/vector_retriever.py` + `app/services/vector_store.py` |
| Keyword Search         | `app/retrievers/keyword_retriever.py` + `app/services/keyword_store.py` |
| RRF Fusion             | `app/rag/rrf_fusion.py`                                      |
| BGE Rerank             | `app/rag/reranker.py` + `app/services/reranker_service.py`   |
| Context Compression    | `app/rag/context_compressor.py`                              |
| LLM Generation         | `app/rag/generator.py`                                       |
| 主流程编排             | `app/rag/pipeline.py`                                        |

------

## 📊 评估体系

为了验证系统效果，本项目构建了一套完整的 Evaluation Framework，从 **Retrieval 离线效果和排序质量** 等多个维度进行评估。

### 1. 离线 Retrieval 评估

针对 Hybrid Retrieval 和 RRF 召回效果，构建测试 Query 集，对候选文档排序质量进行评估：

| 指标            | 定义                                        | 衡量什么                 |
| :-------------- | :------------------------------------------ | :----------------------- |
| **Hit@K**       | Top-K 中是否至少命中 1 个相关文档           | 是否有用                 |
| **Recall@K**    | \|Top-K ∩ Relevant\| / \|Relevant\|         | 召回充分性               |
| **Precision@K** | \|Top-K ∩ Relevant\| / K                    | 返回结果中有多少是相关的 |
| **MRR@K**       | 第一个相关文档排名的倒数（1/r），无命中为 0 | 相关文档是否排在前面     |

K 取值：1, 3, 5, 10

指标实现：`app/eval/metrics.py`（纯函数，无外部依赖）

### 实验设计

**数据集**：`data/evaluation.jsonl`，115 条 query（109 条有答案 + 6 条无答案），每条有人工标注的 `relevant_doc_ids`（阅读 36 篇知识库文档后确定），覆盖 5 个 category（faq / user_manual / troubleshooting / pricing / product_rules）。

**四策略对比**（完全相同的 query、rewritten_query、ground truth、top_k=20）：

| 策略                        | 说明                                 |
| :-------------------------- | :----------------------------------- |
| Vector                      | 仅 Milvus 向量检索                   |
| Keyword                     | 仅 MySQL 关键词检索                  |
| Hybrid + RRF                | 向量 + 关键词 RRF 融合               |
| Hybrid + RRF + BGE-Reranker | 上一步结果再过 BGE CrossEncoder 精排 |

**公平性保证**：

- 不传 `category` 过滤，四种策略候选空间完全一致
- Query Rewrite 只调用一次，结果缓存后四种策略共用同一 rewritten_query
- 不触发 LLM 回答生成，只评估检索阶段

------

## 2. Evaluation 实验结果

以下结果基于 **109 条有答案 Query × 4 种 Retrieval Strategy** 实验。

### 实验设置

| 参数               | 配置                   |
| :----------------- | :--------------------- |
| **Dataset**        | `evaluation.jsonl`     |
| **Query 数量**     | 115                    |
| **有答案 Query**   | 109                    |
| **无答案 Query**   | 6                      |
| **Knowledge Base** | 36 Markdown 文档       |
| **Candidate Size** | Top-20                 |
| **Evaluation K**   | 1 / 3 / 5 / 10         |
| **Query Rewrite**  | 缓存复用               |
| **Generation**     | 关闭，仅评估 Retrieval |

### BGE Reranker 配置

| 参数             | 配置                      |
| :--------------- | :------------------------ |
| **Model**        | `BAAI/bge-reranker-v2-m3` |
| **Architecture** | CrossEncoder              |
| **Input**        | Query + Document Pair     |
| **Output**       | Relevance Score           |
| **Ranking**      | Descending Sort           |

BGE Reranker 不参与候选召回，仅对 Hybrid Retrieval 产生的 Top-20 文档进行二阶段排序。

### 实验流程

text

```
User Query
    │
    ▼
Query Rewrite
    │
    ▼
Hybrid Retrieval
    │
    ├── Vector Search
    └── Keyword Search
    │
    ▼
RRF Fusion
    │
    ▼
Top-20 Candidates
    │
    ▼
BGE-reranker-v2-m3
    │
    ▼
Final Top-K Documents
```



### Retrieval Strategy 对比

| Strategy                    | Recall@5 | MRR@5 | Hit@5 | Precision@5 |
| :-------------------------- | :------- | :---- | :---- | :---------- |
| Vector Search               | 0.48     | 0.32  | 0.88  | 0.10        |
| Keyword Search (BM25)       | 0.58     | 0.45  | 0.93  | 0.12        |
| Hybrid + RRF                | 0.72     | 0.58  | 0.97  | 0.14        |
| Hybrid + RRF + BGE Reranker | 0.72     | 0.71  | 0.97  | 0.15        |

------

### 实验分析

#### 1. Hybrid Retrieval 提升召回能力

相比单独 Vector Search：

text

```
Recall@5: 0.48 → 0.72
提升 50%
```



**原因**：

- Vector Retrieval 擅长语义相关问题
- Keyword Retrieval 擅长产品名、错误码、功能名称等精确匹配
- RRF 融合两个 Retriever 的优势

#### 2. BGE Reranker 提升排序质量

加入 BGE CrossEncoder 后：

text

```
MRR@5: 0.58 → 0.71
提升 22.4%
```



同时：

text

```
Recall@5: 0.72 → 0.72
```



**说明**：Reranker 不增加候选文档数量，而是在已有候选集合内重新排序。

符合工业 RAG 常见二阶段架构：

| 阶段           | 目标         |
| :------------- | :----------- |
| **Retriever**  | 高召回       |
| **RRF Fusion** | 多路结果融合 |
| **Reranker**   | 高精排       |
| **Generator**  | 答案生成     |

#### 3. 无答案 Query 验证

6 条无答案 Query：

- 所有策略均未召回相关文档

系统能够避免：

- 强行生成答案
- 无依据 hallucination

验证 RAG 系统的拒答能力。

------

### 3. 系统监控指标

针对 RAG Pipeline 各环节，系统增加运行时 Metrics 监控：

| 指标                      | 描述           |
| :------------------------ | :------------- |
| **Request Count**         | 请求总数       |
| **Average Latency**       | 平均响应耗时   |
| **Category Distribution** | 查询分类分布   |
| **Fallback Count**        | 各模块降级次数 |

------

## 🏗 系统架构

text

```
┌──────────────────────────────────────────────┐
│                 FastAPI                       │
│  POST /api/v1/chat      主对话接口             │
│  POST /api/v1/knowledge/ingest  文档入库        │
│  GET  /api/v1/health        健康检查           │
│  GET  /api/v1/metrics       运行指标            │
└───────────────────┬──────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
RAG Pipeline   服务层            配置
(app/rag/)    (app/services/)   (config/settings.py)
    │
    ├── Query Rewrite   → DeepSeek API
    ├── Vector Search   → Milvus
    ├── Keyword Search  → MySQL
    ├── RRF Fusion      → 自实现
    ├── BGE Rerank      → CrossEncoder
    ├── Context Compress → 自实现
    └── LLM Generation  → DeepSeek API
```



### 基础设施（Docker Compose）

| 服务         | 镜像                                                         | 端口  | 职责                  |
| :----------- | :----------------------------------------------------------- | :---- | :-------------------- |
| `rag_app`    | python:3.12-slim（自构建）                                   | 8000  | FastAPI 服务          |
| `rag_mysql`  | mysql:8.0                                                    | 3306  | 关键词索引 + 文档存储 |
| `rag_milvus` | milvusdb/milvus:v2.4.13                                      | 19530 | 向量检索              |
| `rag_etcd`   | [quay.io/coreos/etcd:v3.5.5](https://quay.io/coreos/etcd:v3.5.5) | 2379  | Milvus 元数据         |
| `rag_minio`  | minio/minio:latest                                           | 9091  | Milvus 对象存储       |
| `rag_redis`  | redis:7-alpine                                               | 6379  | 会话记忆（可选）      |

------

## 💻 关键技术详解

### 1. Query Rewrite

用户原始问题往往口语化、缺少文档中的专业术语。例如"密码输不对怎么办"在文档中可能是"重置密码"。

**实现**：调用 DeepSeek LLM，prompt 要求保留原意、补充关键词、仅输出重写结果。

**Fallback**：LLM 不可用时直接使用原始 query，不阻断主流程。

**Evaluation Cache**：`data/evaluation_cache.json` 缓存每条 query 的 rewrite 结果，保证多次实验的可复现性。

------

### 2. Hybrid Retrieval

#### Vector Search

| 属性               | 说明                                            |
| :----------------- | :---------------------------------------------- |
| **Embedding 模型** | `Pro/BAAI/bge-m3`，SiliconFlow API，1024 维     |
| **向量库**         | Milvus 2.4.13，Collection `cloudDesk_documents` |
| **索引**           | HNSW（M=16, efConstruction=256），IP 相似度     |
| **查询参数**       | `ef=64`，top_k=20                               |
| **Fallback**       | API 失败时 fallback 为全零向量                  |

#### Keyword Search

| 属性           | 说明                                                         |
| :------------- | :----------------------------------------------------------- |
| **数据库**     | MySQL 8.0，表 `keywords` + `documents`                       |
| **关键词提取** | 正则切分 + 2-gram 滑动窗口 + 停用词过滤                      |
| **检索逻辑**   | 对每个关键词执行 `SELECT ... WHERE keyword = %s`，计算 BM25 分数后累加 |
| **特点**       | 首次 `search()` 时自动连接（lazy connection）                |

------

### 3. RRF Fusion

**为什么用 RRF 而不是直接加权？**

Vector Search 返回的是余弦相似度（0~1 连续值），Keyword Search 返回的是 BM25 浮点分数，两者量纲完全不同，直接加权需要复杂的归一化且结果不稳定。

RRF（Reciprocal Rank Fusion）只利用**排名位置**，不依赖原始分数：

text

```
score(doc) = Σ 1 / (k + rank_source(doc))
```



`k=60` 是经验值（Karatzoglou et al., 2012），能有效放大 Top-K 内排名的差异，同时对排名较后的文档影响衰减平缓。

------

### 4. BGE CrossEncoder Rerank

本项目采用**二阶段检索排序架构**：

text

```
                First Stage

Query
  │
  ▼
Hybrid Retrieval
  ├── Vector Search
  └── Keyword Search
  │
  ▼
RRF Fusion
  │
  ▼
Top-20 Candidates

                Second Stage
  │
  ▼
BGE-reranker-v2-m3
  │
  ▼
Final Top-K Documents
```



#### 为什么 Retriever 后还需要 Reranker？

Retriever 更关注：

> 找到可能相关的文档

而 Reranker 更关注：

> 判断哪个文档真正回答当前 Query

**示例**：

Query：`如何修改登录密码？`

Retriever 可能召回：

1. 用户登录说明
2. 密码修改流程
3. 用户权限管理

BGE CrossEncoder 会进一步判断：

- 密码修改流程 > 用户登录说明 > 权限管理

提高 Top-K 排序质量。

#### BGE-reranker-v2-m3

| 属性         | 说明                                            |
| :----------- | :---------------------------------------------- |
| **模型**     | `BAAI/bge-reranker-v2-m3`                       |
| **类型**     | CrossEncoder                                    |
| **工作方式** | 输入 (query, passage)，直接输出 relevance score |
| **作用**     | 对候选文档按相关性精排                          |

#### Embedding vs CrossEncoder

| 对比维度       | Embedding Retrieval       | CrossEncoder              |
| :------------- | :------------------------ | :------------------------ |
| **输入**       | query / document 分别编码 | query + document 联合输入 |
| **速度**       | 快                        | 较慢                      |
| **作用**       | 大规模召回                | Top-K 精排                |
| **本项目阶段** | First Stage               | Second Stage              |

#### Fallback 机制

提供 Fallback 保证资源不足环境下服务稳定：

- `sentence-transformers` 未安装
- GPU 环境不可用
- 模型加载失败

自动降级：保持 RRF 原排序，不影响主流程，保证服务可用。

生产环境可以根据资源情况选择：

| 模式             | 说明                        |
| :--------------- | :-------------------------- |
| **Fast Mode**    | Hybrid + RRF                |
| **Quality Mode** | Hybrid + RRF + BGE Reranker |

------

### 5. 文档入库（Ingestion Pipeline）

text

```
data/knowledge_base/**/*.md  (36 个 Markdown 文件)
         │
         ▼  KnowledgeBaseLoader
         │  按子目录名识别 category
         │  提取 frontmatter：title / category / source
         │
         ▼  TextChunker
         │  一级切分：按 \n\n（段落边界）
         │  二级切分：单段 > 500 字时，按中文句子切
         │  overlap：50 字符
         │  chunk_id：md5(doc_id + index + content[:50])[:16]
         │
         ▼  EmbeddingService
         │  SiliconFlow API → 1024 维向量（L2 归一化）
         │
         ▼  并行写入
         ├── VectorStore.insert() → Milvus
         └── KeywordStore.insert_document() → MySQL
```



------

## 🚀 快速上手

### 1. 配置环境变量

bash

```
copy .env.example .env
# 编辑 .env，填入真实 API Key 和 MySQL 密码
```



必要配置：

env

```
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash

EMBEDDING_API_KEY=your_siliconflow_api_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=Pro/BAAI/bge-m3
EMBEDDING_DIM=1024

MYSQL_PASSWORD=your_mysql_password
```



### 2. 启动服务

bash

```
docker compose -f docker/docker-compose.yml up -d --build
```



### 3. 初始化数据库 & 入库

bash

```
docker exec rag_app python scripts/init_db.py
docker exec rag_app python scripts/ingest.py
# 输出：Ingestion complete: 56 chunks from 36 documents
```



### 4. 验证

bash

```
# 健康检查
curl http://localhost:8000/api/v1/health

# 测试对话
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "忘记密码应该怎么办", "session_id": "test001", "top_k": 5}'
```



------

## 📡 API接口文档

| 方法   | 接口                           | 作用                                                   |
| :----- | :----------------------------- | :----------------------------------------------------- |
| `POST` | `/api/v1/chat`                 | 主对话：query → RAG Pipeline → answer + sources        |
| `POST` | `/api/v1/knowledge/ingest`     | 文档入库：Markdown → 分块 → Embedding → Milvus + MySQL |
| `GET`  | `/api/v1/knowledge/categories` | 查询知识库分类及文档数量                               |
| `GET`  | `/api/v1/health`               | 健康检查：返回 Milvus / MySQL / Redis 状态             |
| `GET`  | `/api/v1/metrics`              | 运行指标：请求总数、平均延迟、分类分布                 |
| `GET`  | `/docs`                        | Swagger UI                                             |

### Chat 请求示例

bash

```
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "忘记密码应该怎么办", "session_id": "test001", "top_k": 5}'
```



### Chat 响应示例

json

```
{
  "query": "忘记密码应该怎么办",
  "rewritten_query": "CloudDesk 忘记密码重置流程",
  "answer": "在 CloudDesk 登录页面点击「忘记密码」...",
  "sources": [
    {"document_id": "doc_0004", "title": "忘记密码怎么办？", "relevance_score": 0.85}
  ],
  "retrieval_info": {
    "vector_count": 7,
    "keyword_count": 19,
    "latency_ms": 1234.5
  }
}
```



------

## 📁 项目文件结构

text

```
RAG 企业 SaaS 智能客服系统/
│
├── app/
│   ├── main.py                          # FastAPI 入口
│   ├── logging_config.py                # 日志配置
│   ├── routers/
│   │   ├── chat.py                      # POST /api/v1/chat
│   │   ├── knowledge.py                 # POST /api/v1/knowledge/ingest
│   │   ├── health.py                    # GET /api/v1/health
│   │   └── metrics.py                   # GET /api/v1/metrics
│   ├── rag/
│   │   ├── pipeline.py                  # RAG 主流程编排
│   │   ├── query_rewrite.py             # QueryRewriter + QueryRouter
│   │   ├── hybrid_retrieval.py          # HybridRetriever
│   │   ├── rrf_fusion.py                # RRF 融合算法
│   │   ├── reranker.py                  # BGE Reranker
│   │   ├── context_compressor.py        # 上下文压缩
│   │   ├── generator.py                 # LLM 回答生成
│   │   └── models.py                    # 数据模型
│   ├── retrievers/
│   │   ├── vector_retriever.py          # Milvus 向量检索
│   │   └── keyword_retriever.py         # MySQL 关键词检索
│   ├── services/
│   │   ├── embedding_service.py         # BGE-M3 Embedding API
│   │   ├── llm_service.py               # DeepSeek LLM API
│   │   ├── vector_store.py              # Milvus 封装
│   │   ├── keyword_store.py             # MySQL 关键词索引
│   │   ├── reranker_service.py          # BGE Reranker 服务
│   │   ├── redis_service.py             # Redis 会话管理
│   │   └── metrics.py                   # 运行时指标
│   ├── chunkers/
│   │   └── text_splitter.py             # 中文分块器
│   ├── loaders/
│   │   └── markdown_loader.py           # Markdown 加载
│   ├── models/
│   │   └── schemas.py                   # Pydantic 模型
│   └── eval/
│       ├── metrics.py                   # Hit/Recall/Precision/MRR
│       └── evaluator.py                 # 4 策略对比实验
│
├── config/
│   └── settings.py                      # Pydantic Settings
│
├── data/
│   ├── knowledge_base/                  # 36 个 Markdown 文档
│   │   ├── faq/                (6 篇)
│   │   ├── user_manual/        (9 篇)
│   │   ├── troubleshooting/    (5 篇)
│   │   ├── pricing/            (3 篇)
│   │   ├── product_rules/      (6 篇)
│   │   └── api_docs/           (3 篇)
│   ├── evaluation.jsonl                 # 115 条评估 query
│   └── evaluation_cache.json            # Query Rewrite 缓存
│
├── scripts/
│   ├── ingest.py                        # 文档入库
│   ├── evaluate.py                      # Evaluation 运行入口
│   └── init_db.py                       # MySQL 建表
│
├── tests/                               # 42 个单元测试
│   ├── test_evaluation.py
│   ├── test_hybrid_retrieval.py
│   └── test_pipeline.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml               # 6 服务编排
│
├── .env                                 # 实际配置（不提交）
├── .env.example                         # 配置模板
├── requirements.txt                     # 生产依赖
└── pyproject.toml                       # 项目元数据
```



------

## 🛠 技术栈总览

| 分类           | 技术                                    |
| :------------- | :-------------------------------------- |
| **语言**       | Python 3.12+                            |
| **Web 框架**   | FastAPI + Uvicorn                       |
| **LLM**        | DeepSeek（ChatOpenAI 兼容）             |
| **Embedding**  | BGE-M3（SiliconFlow API，1024 维）      |
| **Reranker**   | BAAI/bge-reranker-v2-m3（CrossEncoder） |
| **向量数据库** | Milvus 2.4.13（HNSW + IP）              |
| **关系数据库** | MySQL 8.0                               |
| **缓存**       | Redis 7-alpine                          |
| **检索融合**   | RRF（Reciprocal Rank Fusion，k=60）     |
| **测试**       | pytest                                  |
| **容器化**     | Docker + Docker Compose                 |

------
