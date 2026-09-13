"""
LLM-as-a-Judge Prompt 定义。

每个 Prompt 都要求模型只返回 JSON，并提供明确的字段、评分标准和范围。
"""

import json
from typing import Any, Iterable


_JSON_ONLY_RULE = """
## 输出约束
1. 只输出一个合法 JSON 对象。
2. 禁止输出 JSON 对象之外的任何内容，包括解释、Markdown 代码块、前后缀文字。
3. `score` 必须是 0.0 到 1.0 之间的数值（包含边界）。
4. 所有文本字段必须使用简洁、可验证的中文说明。
""".strip()


def _json_text(value: Any) -> str:
    """将输入稳定地序列化到 Prompt，避免 Prompt 中出现 Python repr。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _format_sources(sources: Iterable[Any]) -> str:
    return _json_text(list(sources))


def build_faithfulness_prompt(
    question: str,
    context: str,
    answer: str,
) -> str:
    """构建 Faithfulness Judge Prompt。"""

    return f"""
你是 RAG 回答忠实度（Faithfulness）评审员。请判断回答中的事实性陈述是否由给定 context 支持。

## 输入字段
- `question`: 用户原始问题
- `context`: 提供给回答生成模型的检索上下文或知识库原文
- `answer`: 待评价的模型回答

## 输入内容
question:
{_json_text(question)}

context:
{_json_text(context)}

answer:
{_json_text(answer)}

## 评价标准
1. 只评价 answer 中事实性内容是否可从 context 得到支持，不评价相关性、流畅度、语气或完整性。
2. 不得使用 context 之外的外部知识补充事实依据。
3. answer 中每一条可验证的事实性陈述都必须在 context 中有直接或合理明确的支持，才能被认为是忠实的。
4. 如果 answer 对 context 进行了无依据扩展、人物/数字/时间/政策改写或因果关系臆测，应扣分并列入 `unsupported_claims`。
5. 明确区分“context 未说明”和“answer 与 context 冲突”；两者都属于不支持。
6. 如果 context 为空，而 answer 包含任何事实性断言，则 score 应为 0.0。

## 0.0~1.0 评分规则
- 1.0：所有事实性陈述均被 context 完整支持，无未支持声明。
- 0.8~0.99：绝大多数内容有支持，仅有轻微、不影响核心事实的扩展。
- 0.5~0.79：核心方向正确，但存在若干未支持或过度推断的重要陈述。
- 0.2~0.49：只有少部分事实得到支持，或存在明显冲突。
- 0.0~0.19：几乎全部事实无支持，或回答主要由编造内容构成。

## JSON 输出格式
{{
  "score": 0.0,
  "reason": "简洁说明评分依据",
  "unsupported_claims": ["逐条列出不被 context 支持的事实性陈述"]
}}

如果没有未支持声明，`unsupported_claims` 必须为空数组。

{_JSON_ONLY_RULE}
""".strip()


def build_answer_relevancy_prompt(
    question: str,
    answer: str,
) -> str:
    """构建 Answer Relevancy Judge Prompt。"""

    return f"""
你是 RAG 回答相关性（Answer Relevancy）评审员。请判断 answer 是否直接、完整地回应 question。

## 输入字段
- `question`: 用户原始问题
- `answer`: 待评价的模型回答

## 输入内容
question:
{_json_text(question)}

answer:
{_json_text(answer)}

## 评价标准
1. 评价重点是回答是否切题，以及是否覆盖问题要求的关键方面。
2. 不评价事实正确性，也不使用外部知识判断答案真假。
3. 冗余背景、无关信息、答非所问、只回答部分子问题都应扣分并列入 `missing_aspects`。
4. 如果问题包含多个明确要求，必须检查每个要求是否被回应。
5. 如果 answer 表示无法回答，但问题其实要求给出具体信息，应视为未完整回应并相应扣分。

## 0.0~1.0 评分规则
- 1.0：直接、完整地回答了问题的全部明确意图。
- 0.8~0.99：回答切题，覆盖主要内容，仅有非关键细节缺失。
- 0.5~0.79：部分相关，但遗漏一个重要方面或包含明显偏题内容。
- 0.2~0.49：仅与问题弱相关，主要问题未被回答。
- 0.0~0.19：完全无关、无法理解，或没有提供任何有用回应。

## JSON 输出格式
{{
  "score": 0.0,
  "reason": "简洁说明评分依据",
  "missing_aspects": ["列出问题中未被回答或回答不完整的方面"]
}}

如果没有缺失方面，`missing_aspects` 必须为空数组。

{_JSON_ONLY_RULE}
""".strip()


def build_citation_accuracy_prompt(
    question: str,
    context: str,
    answer: str,
    sources: Iterable[Any],
) -> str:
    """构建 Citation Accuracy Judge Prompt。"""

    return f"""
你是 RAG 引用准确性（Citation Accuracy）评审员。请判断 answer 中的引用是否真实存在，并且是否正确支持其关联陈述。

## 输入字段
- `question`: 用户原始问题
- `context`: 提供给回答生成模型的检索上下文或知识库原文
- `answer`: 包含引用或来源标注的待评价回答
- `sources`: 当前最终 Retrieval Results 提供的来源信息

## 输入内容
question:
{_json_text(question)}

context:
{_json_text(context)}

answer:
{_json_text(answer)}

sources:
{_format_sources(sources)}

## 重要事实约束
1. `sources` 是当前最终 Retrieval Results 提供的来源信息，是本次评价可使用的引用依据。
2. 当前不存在人工 `expected_citation_doc_ids`，因此不得声称存在人工 Citation Ground Truth，也不得把某个人工标准答案当作引用标签。
3. 只能依据 question、context、answer 和 sources 评价，不得补充外部来源。

## 评价标准
1. 检查 answer 中出现的每个引用标识、来源标题或来源名称是否能在 `sources` 中找到。
2. 检查引用是否支持 answer 中与其关联的具体陈述，而不是仅检查来源名称是否出现。
3. 引用不存在于 sources、引用与陈述不匹配、引用无法由对应来源支持，均属于无效引用。
4. 引用列表、标题或文档 ID 模糊到无法判断对应来源时，应列入 `invalid_citations`。
5. 如果 answer 没有任何引用，即使正文看起来合理，Citation Accuracy 也应为 0.0。
6. 不得因为 sources 中文件很多，就把所有 source 自动视为已被回答引用。

## 0.0~1.0 评分规则
- 1.0：所有引用都存在于 sources 中，并且准确支持其关联陈述。
- 0.8~0.99：绝大多数引用有效，仅有个别轻微标注问题。
- 0.5~0.79：部分引用有效，但存在明显无效或错配引用。
- 0.2~0.49：多数引用无效，或核心结论使用了错误来源。
- 0.0~0.19：没有引用，或引用整体不可验证、与 sources 不匹配。

## JSON 输出格式
{{
  "score": 0.0,
  "reason": "简洁说明评分依据",
  "valid_citations": ["列出能够由 sources 验证并正确支持陈述的引用"],
  "invalid_citations": ["列出不存在、无法验证或与陈述不匹配的引用"]
}}

`valid_citations` 和 `invalid_citations` 必须始终是数组；没有相应内容时使用空数组。

{_JSON_ONLY_RULE}
""".strip()
