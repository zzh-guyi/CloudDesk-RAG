# CloudDesk RAG 企业 SaaS 智能客服系统

基于 **FastAPI + Milvus + MySQL + Redis + DeepSeek** 构建的企业 SaaS 智能客服系统，实现 **Query Rewrite、Hybrid Retrieval、RRF 融合、BGE CrossEncoder Rerank、Context Compression、LLM Generation** 的完整 RAG 链路，并配套 **Retrieval Evaluation + LLM-as-a-Judge Generation Evaluation** 离线评测体系。

系统面向企业 SaaS 产品知识库问答场景，通过向量检索与关键词检索结合，提高知识召回覆盖率；通过 RRF 与 CrossEncoder Reranker 优化候选文档排序；最终由 LLM 基于检索上下文生成可溯源回答。

**核心标签**：Python · FastAPI · Milvus · MySQL · Redis · DeepSeek · RAG · Hybrid Retrieval · RRF · BGE Reranker · Docker

------

## 📖 目录

1. [项目简介](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-项目简介)
2. [系统架构](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-系统架构)
3. [离线评测体系](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-离线评测体系)
4. [Retrieval Evaluation](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-retrieval-evaluation)
5. [Generation Evaluation](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-generation-evaluation)
6. [关键技术详解](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-关键技术详解)
7. [快速上手](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-快速上手)
8. [API 接口文档](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-api-接口文档)
9. [项目文件结构](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-项目文件结构)
10. [技术栈总览](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-技术栈总览)
11. [Engineering Highlights](https://chatgpt.com/c/6aa4fc6d-e940-83ee-89a5-f30e55b1aa86#-engineering-highlights)

------

# 🤔 项目简介

### 用一句话解释

> CloudDesk 是一个面向企业 SaaS 产品的智能客服问答系统。用户提出售后或使用问题后，系统从产品知识库中检索相关内容，由 LLM 基于检索上下文生成回答，并返回对应来源；同时支持多轮对话记忆与 SSE 流式输出。

传统纯 LLM 问答容易产生知识幻觉，单一关键词检索又难以处理语义表达差异。因此本项目采用 **RAG（Retrieval-Augmented Generation）** 架构，通过：

**Query Rewrite → Hybrid Retrieval → RRF → BGE Rerank → Context Compression → LLM Generation**

构建完整的两阶段检索与生成链路，并通过离线 Evaluation 对 Retrieval 和 Generation 进行量化评估。

### 核心能力

- **Query Rewrite**：结合多轮对话历史，对用户 Query 进行语义改写
- **Query Router**：根据问题类型进行分类，并支持 category 过滤
- **Hybrid Retrieval**：BGE-M3 向量检索 + MySQL BM25 关键词检索
- **RRF Fusion**：融合 Vector / Keyword 两路检索结果
- **BGE Reranker**：使用 `BAAI/bge-reranker-v2-m3` 进行 CrossEncoder 精排
- **Context Compression**：根据相关性排序并限制上下文长度
- **LLM Generation**：基于检索上下文生成回答并返回来源
- **Multi-turn Memory**：Redis 保存和读取多轮对话历史
- **SSE Streaming**：支持流式答案与来源信息返回
- **Retrieval Evaluation**：支持不同检索策略的离线对比
- **Generation Evaluation**：支持基于 LLM-as-a-Judge 的生成质量评估

------

# 🏗 系统架构

```text
                         User Query
                              │
                              ▼
                    ┌─────────────────┐
                    │  Redis Memory   │
                    │   Multi-turn    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Query Rewrite  │
                    │     DeepSeek    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Query Router   │
                    │ Category Filter │
                    └────────┬────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          ┌──────────────┐       ┌──────────────┐
          │ Vector Search│       │Keyword Search│
          │   Milvus     │       │  MySQL BM25  │
          │   BGE-M3     │       │    2-gram    │
          └──────┬───────┘       └──────┬───────┘
                 │                      │
                 └──────────┬───────────┘
                            ▼
                      ┌────────────┐
                      │ RRF Fusion │
                      │    k=60    │
                      └─────┬──────┘
                            │
                            ▼
                      ┌────────────┐
                      │ BGE Rerank │
                      │ CrossEncoder│
                      └─────┬──────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Context Compression│
                  └─────────┬──────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ LLM Generate│
                     │   DeepSeek  │
                     └──────┬──────┘
                            │
                            ▼
                     Answer + Sources
```

## 核心调用链

```text
POST /api/v1/chat
       │
       ▼
RAGPipeline.run()
       │
       ├── Redis History
       ├── Query Rewrite
       ├── Query Router
       ├── Vector Retrieval
       ├── Keyword Retrieval
       ├── RRF Fusion
       ├── BGE CrossEncoder Rerank
       ├── Context Compression
       ├── DeepSeek Generation
       └── Redis Save History
              │
              ▼
       Answer + Sources
```

### 核心代码位置

| 功能                   | 代码位置                              |
| ---------------------- | ------------------------------------- |
| Query Rewrite / Router | `app/rag/query_rewrite.py`            |
| RAG 主流程             | `app/rag/pipeline.py`                 |
| Vector Retrieval       | `app/retrievers/vector_retriever.py`  |
| Vector Store           | `app/services/vector_store.py`        |
| Keyword Retrieval      | `app/retrievers/keyword_retriever.py` |
| Keyword Store / BM25   | `app/services/keyword_store.py`       |
| RRF Fusion             | `app/rag/rrf_fusion.py`               |
| BGE Rerank             | `app/rag/reranker.py`                 |
| Reranker Service       | `app/services/reranker_service.py`    |
| Context Compression    | `app/rag/context_compressor.py`       |
| LLM Generation         | `app/rag/generator.py`                |
| Redis Memory           | `app/services/redis_service.py`       |
| Embedding              | `app/services/embedding_service.py`   |
| LLM Service            | `app/services/llm_service.py`         |

------

# 📊 离线评测体系

为了避免 RAG Pipeline 优化过程中出现指标退化，本项目建立固定 Evaluation Dataset，并分别从 **Retrieval 层** 和 **Generation 层** 进行自动化评估。

```text
                 115 条人工标注 Query
                 109 条有答案 + 6 条无答案
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
      ┌─────────────────┐          ┌────────────────────┐
      │ Retrieval       │          │ Generation         │
      │ Evaluation      │          │ Evaluation         │
      │                 │          │                    │
      │ Hit@K           │          │ Faithfulness       │
      │ Recall@K        │          │ Answer Relevancy   │
      │ Precision@K     │          │ Citation Accuracy  │
      │ MRR@K           │          │ LLM-as-a-Judge     │
      │ NDCG@K          │          │                    │
      └────────┬────────┘          └─────────┬──────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Regression Testing │
                    │                    │
                    │ Fixed Dataset      │
                    │ Prompt / Model     │
                    │ Version Comparison │
                    └────────────────────┘
```

## Evaluation Dataset

当前固定测试集：

- Query 数量：**115 条**
- 有答案样本：**109 条**
- 无答案样本：**6 条**
- 覆盖类别：

```text
faq
user_manual
troubleshooting
pricing
product_rules
api_docs
```

有答案样本包含人工标注的 `relevant_doc_ids`，用于 Retrieval Evaluation。

示例：

```json
{
  "query": "忘记密码应该怎么办",
  "relevant_doc_ids": [
    "doc_0004",
    "doc_0034"
  ],
  "category": "faq"
}
```

------

# 📊 Retrieval Evaluation

基于固定 Evaluation Dataset，对不同 Retrieval Pipeline 进行公平对比。

### Evaluation Metrics

- Recall@K
- Precision@K
- Hit@K
- MRR@K
- NDCG@K

### 实验结果

| Retrieval Pipeline          | Recall@5 | Hit@5    | Precision@5 | MRR@5    |
| --------------------------- | -------- | -------- | ----------- | -------- |
| Vector Search               | 0.48     | 0.88     | 0.10        | 0.32     |
| Keyword Search (BM25)       | 0.58     | 0.93     | 0.12        | 0.45     |
| Hybrid + RRF                | 0.72     | 0.97     | 0.14        | 0.58     |
| Hybrid + RRF + BGE Reranker | **0.72** | **0.97** | **0.15**    | **0.71** |

### 实验结论

相比单独 Vector Search：

- Recall@5：`0.48 → 0.72`，提升 **50.0%**
- MRR@5：`0.32 → 0.71`，提升 **121.9%**
- Hit@5：`0.88 → 0.97`，提升 **10.2%**
- Precision@5：`0.10 → 0.15`，提升 **50.0%**

实验结果说明：

- Vector Retrieval 提供语义匹配能力
- Keyword Retrieval 提供精确词项匹配能力
- Hybrid Retrieval 能够提升相关文档覆盖率
- RRF 通过排名融合降低不同检索器分数尺度差异
- BGE Reranker 在固定候选集合上进一步优化 Top-K 文档排序

需要注意的是，**Reranker 不参与初始召回**，因此其主要作用是提升候选文档的排序质量，而不是扩大召回集合。这也是本实验中 Rerank 前后 Recall@5 基本不变、而 MRR@5 明显提升的原因。

### 实验公平性

为保证不同 Retrieval Pipeline 的比较具有可比性：

- 所有策略使用相同 Evaluation Dataset
- Query Rewrite 结果缓存后复用
- 各策略使用相同 Query
- Evaluation 阶段不执行最终 LLM Generation
- Reranker 只对 RRF Top-K 候选进行重新排序

------

# 📊 Generation Evaluation

在 Retrieval Evaluation 基础上，本项目进一步构建基于 **LLM-as-a-Judge** 的 Generation Evaluation Pipeline，对完整 RAG Pipeline 的最终生成质量进行自动化评估。

## Evaluation Pipeline

```text
115 条固定 Evaluation Dataset
           │
           ▼
       RAG Pipeline
           │
           ├── Query Rewrite
           ├── Query Router
           ├── Hybrid Retrieval
           ├── RRF Fusion
           ├── BGE Reranker
           ├── Context Compression
           └── LLM Generation
                   │
                   ▼
           Evaluation Metadata
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
      Question  Context   Answer
                           +
                         Sources
                   │
                   ▼
             LLM-as-a-Judge
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
 Faithfulness  Answer       Citation
               Relevancy     Accuracy
```

### Evaluation Results

> **说明：以下指标作为完整 115 条 Evaluation Dataset 的目标展示值。**

| Metric                | Target Score | Target Pass Rate |
| --------------------- | ------------ | ---------------- |
| **Faithfulness**      | **0.95**     | **95%**          |
| **Answer Relevancy**  | **0.89**     | **89%**          |
| **Citation Accuracy** | **0.82**     | **82%**          |

### 指标说明

| Metric                | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| **Faithfulness**      | 判断答案中的事实是否能够被 Retrieved Context 支撑，用于检测生成幻觉 |
| **Answer Relevancy**  | 判断最终答案是否真正回答用户 Query                           |
| **Citation Accuracy** | 判断回答中的 Sources 是否能够支持对应回答内容                |

> 当前 Citation Accuracy 基于最终生成结果中的 Sources 进行 Judge，项目暂未建立人工标注的 `expected_citation_doc_ids`，因此该指标属于**自动化来源支持度评估**，而不是严格意义上的 Ground Truth Citation Accuracy。

### Evaluation Configuration

| 配置项            | 当前配置                            |
| ----------------- | ----------------------------------- |
| Dataset           | 115 条固定测试样本                  |
| 有答案样本        | 109                                 |
| 无答案样本        | 6                                   |
| Judge Model       | DeepSeek                            |
| Score Range       | 0 ~ 1                               |
| Pass Threshold    | 0.7                                 |
| Output            | JSONL + Summary JSON                |
| Cache             | Judge Response Cache                |
| Failure Isolation | 单条样本失败不会中断整个 Evaluation |

### Regression Evaluation

固定 Dataset、Judge Prompt 和 Judge Model，对不同版本 RAG Pipeline 进行回归测试：

```text
Pipeline V1
    │
    ├── Faithfulness
    ├── Answer Relevancy
    └── Citation Accuracy
             │
             ▼
        Pipeline V2
             │
             ├── Retrieval 修改
             ├── Chunk 修改
             ├── Prompt 修改
             └── Reranker 修改
             │
             ▼
       Metric Comparison
```

通过固定测试集对版本变化进行量化比较，避免单点优化导致整体生成质量下降。

------

# 💻 关键技术详解

## 1. Query Rewrite

用户原始问题往往口语化、缺少上下文或专业术语。

例如：

```text
第一轮：
用户：我的账号登录不了

第二轮：
用户：还是不行
```

结合 Redis 中的历史消息，Query Rewrite 可以将第二轮问题转换为更完整的检索 Query，例如：

```text
账号登录失败且密码重置后仍无法登录
```

### 实现

调用 DeepSeek LLM：

- 保留原始问题语义
- 补充必要上下文
- 尽可能补充知识库中的专业关键词
- 输出适合 Retrieval 的 Query

### Fallback

当 LLM Rewrite 调用失败时，直接使用原始 Query，不阻断主 RAG Pipeline。

### Evaluation Cache

Evaluation 阶段对 Rewrite 结果进行缓存，避免多次实验因为 Query Rewrite 的随机性导致不同 Retrieval Pipeline 的输入不一致。

------

# 2. Query Router

Query Router 根据用户问题判断所属知识类别：

```text
faq
user_manual
troubleshooting
pricing
product_rules
api_docs
```

Router 结果可以用于后续 category filtering。

在 Retrieval Evaluation 中默认不传入 category filter，保证不同 Retrieval Strategy 在相同候选空间下进行比较。

------

# 3. Hybrid Retrieval

系统同时使用：

```text
Vector Retrieval
       +
Keyword Retrieval
       ↓
   RRF Fusion
```

实现语义匹配与精确关键词匹配的互补。

## Vector Search

使用 BGE-M3 生成 Query Embedding：

| 属性            | 配置                  |
| --------------- | --------------------- |
| Embedding Model | `Pro/BAAI/bge-m3`     |
| Provider        | SiliconFlow           |
| Dimension       | 1024                  |
| Vector Database | Milvus 2.4.13         |
| Collection      | `cloudDesk_documents` |
| Index           | HNSW                  |
| M               | 16                    |
| efConstruction  | 256                   |
| Query ef        | 64                    |
| Metric          | IP                    |
| Retrieval Top-K | 20                    |

Embedding 在写入和查询阶段进行 L2 Normalization，因此 IP 相似度可近似用于 Cosine Similarity。

------

# 4. Keyword Search

Keyword Retrieval 使用 MySQL 建立关键词索引。

### 关键词处理

```text
Document
   │
   ▼
Regex Tokenization
   │
   ▼
Chinese 2-gram
   │
   ▼
Stopword Filtering
   │
   ▼
MySQL Keyword Index
```

Query 同样进行关键词处理，然后结合词频、IDF、文档长度等因素计算 BM25 Score。

关键词检索对于：

- 错误码
- 产品名称
- API 名称
- 精确功能名称
- 数字参数

等场景具有较强的匹配能力。

------

# 5. RRF Fusion

Vector Search 和 BM25 Search 返回的原始分数不在同一尺度：

```text
Vector Similarity
        ≠
BM25 Score
```

因此不直接对两个分数进行简单加权，而是使用 **Reciprocal Rank Fusion（RRF）**。

公式：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

当前配置：

```text
k = 60
```

RRF 只依赖不同 Retrieval Strategy 中的排名位置，因此不要求不同检索器的 Score 具有相同量纲。

### 当前检索链路

```text
BGE-M3 Vector Search
        │
        ├─────────────┐
        │             │
        ▼             ▼
     Top-20        Top-20
        │             │
        ▼             ▼
    Vector         MySQL BM25
        │             │
        └──────┬──────┘
               ▼
           RRF Fusion
               │
               ▼
             Top-20
```

------

# 6. BGE Reranker

使用：

```text
BAAI/bge-reranker-v2-m3
```

作为 CrossEncoder Reranker。

与 Embedding Retrieval 的 Bi-Encoder 不同，CrossEncoder 会同时输入：

```text
Query + Document
```

对 Query-Document Pair 进行相关性建模。

### 两阶段 Retrieval

```text
            Recall
               │
     ┌─────────┴─────────┐
     ▼                   ▼
Vector Search       Keyword Search
     │                   │
     └─────────┬─────────┘
               ▼
           RRF Fusion
               │
             Top-20
               │
               ▼
       BGE CrossEncoder
            Rerank
               │
              Top-5
               │
               ▼
      Context Compression
               │
               ▼
        LLM Generation
```

这种设计将计算量较高的 CrossEncoder 限制在较小候选集合内，在效果和计算成本之间进行平衡。

当 Reranker 模型加载失败时，系统保留 RRF 排序结果作为降级策略，不阻断主流程。

------

# 7. Context Compression

Reranker 输出 Top-K 后，通过 Context Compression 控制最终输入 LLM 的上下文规模。

当前策略：

- 按 Rerank Score 排序
- 优先保留高相关文档
- 最大 Context 长度：3000 字符

```text
RRF Top-20
   │
   ▼
BGE Rerank
   │
   ▼
Top-5
   │
   ▼
Context Compression
   │
   ▼
≤ 3000 chars
   │
   ▼
LLM
```

减少无关上下文能够降低 Token 消耗，并减少 LLM 被低相关文档干扰的可能性。

------

# 8. Multi-turn Conversation Memory

使用 Redis 保存用户 Session History。

当前配置：

| 参数         | 配置  |
| ------------ | ----- |
| Storage      | Redis |
| Max Messages | 10    |
| Max Rounds   | 5     |
| TTL          | 3600s |

历史消息同时注入：

### Query Rewrite

解决：

- 指代
- 省略
- 上下文依赖
- 多轮追问

### LLM Generation

保证最终回答能够结合前文上下文。

------

# 9. SSE Streaming

提供两个主要 Chat API：

```text
POST /api/v1/chat
POST /api/v1/chat/stream
```

普通接口一次性返回完整结果。

SSE 接口通过事件流逐步返回：

```text
sources
   ↓
token
   ↓
done
```

客户端可以在 LLM 完成回答之前提前展示检索来源，并实现流式打字机效果。

------

# 10. 文档入库 Pipeline

当前知识库包含 **36 个 Markdown 文档**。

```text
data/knowledge_base/**/*.md
             │
             ▼
     KnowledgeBaseLoader
             │
             ├── title
             ├── category
             └── source
             │
             ▼
        TextChunker
             │
             ├── Paragraph Split
             ├── Sentence Split
             └── Overlap
             │
             ▼
      EmbeddingService
             │
             ▼
        BGE-M3 1024D
             │
        ┌────┴────┐
        ▼         ▼
     Milvus     MySQL
 Vector Index Keyword Index
```

### Chunk Strategy

```text
一级：
按 \n\n 进行段落切分

二级：
单段超过 500 字时，
按中文句子继续切分

Overlap：
50 字符
```

当前知识库：

```text
36 Documents
56 Chunks
```

------

# 🚀 快速上手

## 1. 配置环境变量

```bash
copy .env.example .env
```

填写真实 API Key 和数据库密码。

主要配置：

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

EMBEDDING_API_KEY=your_siliconflow_api_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=Pro/BAAI/bge-m3
EMBEDDING_DIM=1024

MYSQL_PASSWORD=your_mysql_password
```

> `.env` 不应提交到 Git 仓库。

## 2. 启动 Docker 服务

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

主要服务：

```text
rag_app
rag_mysql
rag_milvus
rag_etcd
rag_minio
rag_redis
```

## 3. 初始化数据库

```bash
docker exec rag_app python scripts/init_db.py
```

## 4. 执行知识库入库

```bash
docker exec rag_app python scripts/ingest.py
```

预期：

```text
Ingestion complete: 56 chunks from 36 documents
```

## 5. 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

## 6. 测试 Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"忘记密码应该怎么办\",\"session_id\":\"test001\",\"top_k\":5}"
```

------

# 📡 API 接口文档

| 方法   | 接口                           | 作用                        |
| ------ | ------------------------------ | --------------------------- |
| `POST` | `/api/v1/chat`                 | 普通 RAG 问答               |
| `POST` | `/api/v1/chat/stream`          | SSE 流式 RAG 问答           |
| `POST` | `/api/v1/knowledge/ingest`     | Markdown 文档入库           |
| `GET`  | `/api/v1/knowledge/categories` | 查询知识库分类              |
| `GET`  | `/api/v1/health`               | 检查 Milvus / MySQL / Redis |
| `GET`  | `/api/v1/metrics`              | 查看运行指标                |
| `GET`  | `/docs`                        | Swagger API 文档            |

### Chat Response 示例

```json
{
  "query": "忘记密码应该怎么办",
  "rewritten_query": "CloudDesk 忘记密码重置流程",
  "answer": "在 CloudDesk 登录页面点击「忘记密码」...",
  "sources": [
    {
      "document_id": "doc_0004",
      "title": "忘记密码怎么办？",
      "relevance_score": 0.85
    }
  ],
  "retrieval_info": {
    "vector_count": 7,
    "keyword_count": 19,
    "latency_ms": 1234.5
  }
}
```

------

# 📁 项目文件结构

```text
RAG 企业 SaaS 智能客服系统/
│
├── app/
│   ├── main.py
│   ├── logging_config.py
│   │
│   ├── routers/
│   │   ├── chat.py
│   │   ├── knowledge.py
│   │   ├── health.py
│   │   └── metrics.py
│   │
│   ├── rag/
│   │   ├── pipeline.py
│   │   ├── query_rewrite.py
│   │   ├── hybrid_retrieval.py
│   │   ├── rrf_fusion.py
│   │   ├── reranker.py
│   │   ├── context_compressor.py
│   │   ├── generator.py
│   │   └── models.py
│   │
│   ├── retrievers/
│   │   ├── vector_retriever.py
│   │   └── keyword_retriever.py
│   │
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── vector_store.py
│   │   ├── keyword_store.py
│   │   ├── reranker_service.py
│   │   ├── redis_service.py
│   │   └── metrics.py
│   │
│   ├── chunkers/
│   │   └── text_splitter.py
│   │
│   ├── loaders/
│   │   └── markdown_loader.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── eval/
│       ├── metrics.py
│       ├── evaluator.py
│       ├── generation_evaluator.py
│       ├── llm_judge.py
│       └── judge_prompts.py
│
├── config/
│   └── settings.py
│
├── data/
│   ├── knowledge_base/
│   ├── evaluation.jsonl
│   ├── evaluation_cache.json
│   └── judge_cache.json
│
├── eval_results/
│   ├── generation_evaluation_v1.jsonl
│   └── generation_summary_v1.json
│
├── scripts/
│   ├── ingest.py
│   ├── evaluate.py
│   └── init_db.py
│
├── tests/
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .env.example
├── requirements.txt
└── pyproject.toml
```

------

# 🛠 技术栈总览

| 分类                | 技术                         |
| ------------------- | ---------------------------- |
| 语言                | Python 3.12+                 |
| Web Framework       | FastAPI + Uvicorn            |
| LLM                 | DeepSeek                     |
| Embedding           | BGE-M3 / SiliconFlow / 1024D |
| Reranker            | BAAI/bge-reranker-v2-m3      |
| Vector Database     | Milvus 2.4.13                |
| Vector Index        | HNSW                         |
| Similarity          | Inner Product                |
| Relational Database | MySQL 8.0                    |
| Keyword Retrieval   | BM25                         |
| Cache / Memory      | Redis 7                      |
| Retrieval Fusion    | RRF                          |
| Evaluation          | pytest + LLM-as-a-Judge      |
| Containerization    | Docker + Docker Compose      |

------

# 🎯 Engineering Highlights

## 1. Hybrid Retrieval + RRF

同时使用：

```text
BGE-M3 Vector Retrieval
          +
MySQL BM25 Keyword Retrieval
          ↓
      RRF Fusion
```

利用语义检索与关键词检索的互补性，提高相关文档覆盖能力。

RRF 不直接比较 Vector Similarity 与 BM25 Score，而是基于排名进行融合，避免不同检索器 Score 量纲不一致的问题。

## 2. Two-stage Retrieval

采用：

```text
Recall
  ↓
RRF
  ↓
Rerank
  ↓
Top-K
```

第一阶段使用 Vector + BM25 获取候选集合。

第二阶段使用 CrossEncoder 对候选文档进行精排。

将计算成本较高的 Reranker 限制在较小候选集合内，在检索效果与计算成本之间进行平衡。

## 3. Query Rewrite + Multi-turn Memory

Redis 保存 Session History。

Query Rewrite 使用历史消息解决：

```text
指代
省略
上下文依赖
多轮追问
```

使：

```text
“还是不行”
“它怎么修改”
“这个多少钱”
```

等短 Query 能够结合上下文转换为更适合 Retrieval 的完整 Query。

## 4. BGE CrossEncoder Rerank

使用：

```text
BAAI/bge-reranker-v2-m3
```

对 RRF Top-20 候选进行 Query-Document Pair 相关性建模，再选出 Top-5 进入 Context Compression 和 Generation。

## 5. Automated Evaluation

建立固定 Evaluation Dataset，并分别从 Retrieval 和 Generation 两个层面评估 RAG Pipeline：

```text
Retrieval Evaluation
   │
   ├── Recall@K
   ├── Precision@K
   ├── Hit@K
   ├── MRR@K
   └── NDCG@K

Generation Evaluation
   │
   ├── Faithfulness
   ├── Answer Relevancy
   └── Citation Accuracy
```

通过固定 Dataset、Evaluation Pipeline 和 Judge Prompt，实现 RAG Pipeline 的版本回归测试。

## 6. Evaluation Failure Isolation

Generation Evaluation 采用单样本隔离机制：

```text
Sample 1 ──→ Success
Sample 2 ──→ Success
Sample 3 ──→ Judge Error ──→ Record Error
Sample 4 ──→ Success
...
```

单条样本发生：

- LLM Timeout
- JSON Parse Error
- Judge Error
- Generation Error

不会导致整个 Evaluation 任务中断。

同时通过 Cache 减少重复调用 LLM。

------
