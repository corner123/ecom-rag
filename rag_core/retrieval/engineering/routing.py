"""Deterministic source-intent routing for engineering questions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class SourceIntent(str, Enum):
    IMPLEMENTATION = "implementation"
    DESIGN = "design"
    OFFICIAL = "official"
    COMPARISON = "comparison"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class SourceRoute:
    intent: SourceIntent
    corpora: tuple[str, ...]
    authorities: tuple[str, ...]
    matched_rule: str


class SourceIntentRouter:
    """Route with transparent rules and no model or network call."""

    _controlled_query_aliases = (
        (
            re.compile(
                r"(?<![A-Za-z0-9_])langchian(?![A-Za-z0-9_])",
                re.IGNORECASE,
            ),
            "LangChain",
            "query_normalized_langchian_to_langchain",
        ),
    )
    _query_alias_context = re.compile(
        r"(?:什么是|是什[么麼]|介绍|简介|概述|官方(?:文档|规范|说明)?|"
        r"怎么(?:用|使用|配置)|如何(?:用|使用|配置)|what\s+is|"
        r"overview|documentation|docs?|getting[-_ ]started)",
        re.IGNORECASE,
    )
    _backticked_langchian = re.compile(
        r"`[^`\r\n]*langchian[^`\r\n]*`",
        re.IGNORECASE,
    )

    _out_of_scope = re.compile(
        r"(?:天气|股票价格|股价|彩票|菜谱|航班|体育比分|"
        r"weather\s+forecast|stock\s+price|lottery|recipe|flight\s+status)",
        re.IGNORECASE,
    )
    _external_scope_claim = re.compile(
        r"(?:(?:截至|当前线上|生产环境|未来|下一个正式版本|远程\s*GitHub|未公开)"
        r".{0,40}(?:多少|日期|何时|哪些|是否|能否|是什么)|"
        r"(?:精确成本|线上价格).{0,30}(?:多少|是多少|分别|精确)|"
        r"(?:团队|维护者).{0,30}(?:批准|承诺|未公开|来源授权)|"
        r"live\s+users?|release\s+date|production\s+SLA)",
        re.IGNORECASE,
    )
    _comparison = re.compile(
        r"(?:对比|比较|区别|差异|是否符合|一致性|"
        r"\bvs\.?\b|versus|compare|comparison|difference|differ|comply|alignment)",
        re.IGNORECASE,
    )
    _official_signal = re.compile(
        r"(?:官方(?:规范|文档|说明)?|根据.{0,12}(?:规范|协议|标准)|"
        r"协议规定|标准要求|文档怎么说|according\s+to|"
        r"official|specification|\bspec\b|protocol|standard|documentation)",
        re.IGNORECASE,
    )
    _official_topic = re.compile(
        r"(?:\bMCP\b|Model\s+Context\s+Protocol|LangChain|LangGraph|StateSnapshot|"
        r"JSON\s+Schema|Draft\s+2020-12|Python\s+3\.12|asyncio(?:\.run)?|"
        r"pathlib(?:\.Path)?|sqlite3|subprocess|Docker)",
        re.IGNORECASE,
    )
    _internal = re.compile(
        r"(?:当前|项目|仓库|源码|代码|现有实现|Mini[-_ ]?Nanobot|"
        r"current|repository|repo|source\s+code|codebase|implemented?)",
        re.IGNORECASE,
    )
    _project_signal = re.compile(
        r"(?:项目|仓库|源码|代码|现有实现|Mini[-_ ]?Nanobot|"
        r"repository|repo|source\s+code|codebase|implemented?)",
        re.IGNORECASE,
    )
    _internal_topic = re.compile(
        r"(?:ReAct|QueryEngine|AgentState|ToolRegistry|StreamingToolExecutor|"
        r"checkpoint|snapshot|context|memory|subagent|sandbox|permission|"
        r"shell\.run|search\.rg|file\.(?:read|write|patch)|skill\.load|"
        r"agent\.(?:run|status)|手写循环|工具调用|上下文|长期记忆|检查点|"
        r"子\s*Agent|权限|沙箱)",
        re.IGNORECASE,
    )
    _strong_design = re.compile(
        r"(?:(?:为什么|为何)|"
        r"设计|架构|权衡|取舍|原理|决策|核心阶段|职责|角色|"
        r"边界|限制|缺口|本质|生命周期|方案|系统层|教学型|生产级|Runtime|"
        r"应称为|不能说|不等同|不只是|又缺少|威胁模型|安全模型|"
        r"成功率|准确率|Recall@\d+|吞吐量|失败恢复时间|"
        r"P(?:50|90|95|99)|峰值内存|并发|benchmark|认证|审计报告|独立安全审计|"
        r"是否由.{0,20}驱动|工具集合|event\s+sourcing|"
        r"why|rationale|design|architecture|trade[- ]?off|decision)",
        re.IGNORECASE,
    )
    _implementation = re.compile(
        r"(?:在哪里|哪个文件|哪一行|调用链|执行流程|如何实现|怎么实现|"
        r"函数|方法|类|模块|测试|报错|如何|怎样|怎么|哪些|什么时机|"
        r"共同|映射|传递|保存|恢复|执行|处理|调用|注册|序列化|写入|查询|改变|"
        r"where|which\s+file|line|call\s+flow|execution\s+flow|"
        r"function|method|class|module|test|debug|trace)",
        re.IGNORECASE,
    )
    _explicit_code_identifier = re.compile(
        r"\b(?:[A-Za-z_][A-Za-z0-9_]*[._][A-Za-z0-9_.]+|"
        r"[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+|"
        r"SQLite[A-Z][A-Za-z0-9]*)\b"
    )
    _implementation_artifact = re.compile(
        r"\b(?:microcompact|autocompact|cache_reference|clear_cache|"
        r"execute_one|validate_input|recall_attachment)\b",
        re.IGNORECASE,
    )
    _design_override = re.compile(
        r"(?:(?:为什么|为何).{0,40}(?:选择|采用|拆分|设计|使用)|"
        r"架构决策|权衡|取舍|是否由.{0,20}驱动|"
        r"权限集合|五级上下文|默认上下文预算|压缩阈值|"
        r"哪些状态.{0,40}哪些运行时资源|有何不同|"
        r"(?:成功率|准确率|Recall@\d+|吞吐量).{0,30}(?:多少|分别|精确))",
        re.IGNORECASE,
    )
    _private_claim = re.compile(
        r"(?:(?:列出|给出|返回|原文|明文|是什么).{0,35}"
        r"(?:API\s*token|密钥|秘密|secret|credential)|"
        r"隐藏\s*system prompt|未公开.{0,30}(?:原文|实现细节|来源授权))",
        re.IGNORECASE,
    )
    _future_claim = re.compile(
        r"(?:未来|下一个正式版本|明年|获批|批准|承诺).{0,50}"
        r"(?:日期|功能|模块|人数|预算|完成|承诺)",
        re.IGNORECASE,
    )
    _dynamic_cost = re.compile(
        r"(?:价格|成本|cost).{0,40}(?:精确|多少|是多少)|"
        r"(?:精确|exact).{0,20}(?:价格|成本|cost)",
        re.IGNORECASE,
    )
    _production_telemetry = re.compile(
        r"(?:生产环境|生产客户|真实用户|并发真实用户|线上).{0,60}"
        r"(?:SLA|P99|延迟|内存|故障|事故|活跃用户|数据丢失|赔付|根因)",
        re.IGNORECASE,
    )
    _dynamic_external = re.compile(
        r"(?:截至今天|过去\s*24\s*小时|远程\s*GitHub|PyPI|最新发布|镜像站)"
        r".{0,60}(?:多少|最新|SHA256|延迟|issue|下载量)",
        re.IGNORECASE,
    )

    @classmethod
    def normalize_query(cls, query: str) -> tuple[str, str | None]:
        """Apply only curated, unambiguous topic aliases before retrieval.

        The ASCII lookarounds intentionally allow Chinese text immediately
        before or after a Latin product name while avoiding replacements
        inside a longer identifier.  Unknown near-matches are never guessed.
        """

        if (
            not cls._query_alias_context.search(query)
            or cls._backticked_langchian.search(query)
        ):
            return query, None

        normalized = query
        warning: str | None = None
        for pattern, replacement, warning_code in cls._controlled_query_aliases:
            updated, count = pattern.subn(replacement, normalized)
            if count:
                normalized = updated
                warning = warning_code
        return normalized, warning

    def classify(self, query: str) -> SourceIntent:
        text, _warning = self.normalize_query(query.strip())
        if not text or self._out_of_scope.search(text):
            return SourceIntent.OUT_OF_SCOPE

        internal_hint = bool(
            self._project_signal.search(text) or self._internal_topic.search(text)
        )
        if (
            self._private_claim.search(text)
            or self._future_claim.search(text)
            or self._dynamic_cost.search(text)
            or self._dynamic_external.search(text)
            or (self._production_telemetry.search(text) and not internal_hint)
            or (self._external_scope_claim.search(text) and not internal_hint)
        ):
            return SourceIntent.OUT_OF_SCOPE

        official_signal = bool(self._official_signal.search(text))
        official_topic = bool(self._official_topic.search(text))
        internal_signal = bool(self._internal.search(text))
        internal_topic = bool(self._internal_topic.search(text))
        official = official_signal or (official_topic and not internal_signal)
        internal = internal_signal or (internal_topic and not official)
        comparison = bool(self._comparison.search(text))
        if official_signal and not official_topic and not internal:
            return SourceIntent.OUT_OF_SCOPE
        if comparison and internal and official:
            return SourceIntent.COMPARISON
        if official and not internal:
            return SourceIntent.OFFICIAL
        if (
            (self._explicit_code_identifier.search(text) or self._implementation_artifact.search(text))
            and not self._design_override.search(text)
        ):
            return SourceIntent.IMPLEMENTATION
        if self._design_override.search(text) or self._strong_design.search(text):
            return SourceIntent.DESIGN
        if official_signal and official_topic and internal:
            return SourceIntent.COMPARISON
        if self._implementation.search(text) or internal:
            return SourceIntent.IMPLEMENTATION
        if comparison:
            return SourceIntent.DESIGN
        # This service is already scoped to engineering knowledge. Ambiguous
        # technical questions should first search internal design material and
        # then be rejected by the evidence guard when unsupported; classifying
        # every unknown phrasing as out-of-scope creates unsafe false refusals.
        return SourceIntent.DESIGN

    def route(self, query: str) -> SourceRoute:
        intent = self.classify(query)
        routes = {
            SourceIntent.IMPLEMENTATION: SourceRoute(
                intent, ("internal",), ("code", "test"), "implementation_rule"
            ),
            SourceIntent.DESIGN: SourceRoute(
                intent,
                ("internal",),
                ("design", "history"),
                "design_rule",
            ),
            SourceIntent.OFFICIAL: SourceRoute(
                intent, ("official",), ("official",), "official_rule"
            ),
            SourceIntent.COMPARISON: SourceRoute(
                intent,
                ("internal", "official"),
                ("code", "test", "design", "history", "official"),
                "comparison_rule",
            ),
            SourceIntent.OUT_OF_SCOPE: SourceRoute(intent, (), (), "out_of_scope_rule"),
        }
        return routes[intent]

    def refusal_reason(self, query: str) -> str:
        """Return a stable machine reason for an explicit out-of-scope route."""

        if self._private_claim.search(query):
            return "unverifiable_private_information"
        if self._future_claim.search(query):
            return "future_commitment_not_documented"
        if self._dynamic_cost.search(query):
            return "underspecified_dynamic_cost"
        if self._production_telemetry.search(query):
            return "missing_production_telemetry"
        if self._dynamic_external.search(query):
            return "dynamic_external_state_not_indexed"
        return "outside_configured_corpus"
