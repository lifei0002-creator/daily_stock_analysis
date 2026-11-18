# -*- coding: utf-8 -*-
"""
===================================
HithinkFetcher - 同花顺 Financial-API 兜底源
===================================

数据来源：同花顺官方金融数据服务（hithink-finance），
https://fuyao.aicubes.cn，GitHub HiThink-Tech/Financial-API（MIT）。
一个统一 API Key 通 REST / MCP / CLI / Python / Agent Skill。

定位：东财 / AKShare / 新浪三条免费链路在 GitHub runner 上失灵时的兜底，
覆盖三处（全部 fail-open，永不抛错到调用方）：
1) 实时行情快照（+ 估值快照补 PE/PB/中文名）→ manager 实时行情循环的
   `hithink` 分支；
2) 基本面 bundle（财务指标 + 利润表 + 现金流量表）→ AKShare bundle 为空时
   由 `DataFetcherManager.get_fundamental_context` 按块填空；
3) 龙虎榜 flag → AKShare 龙虎榜完全答不上来时，由
   `DataFetcherManager.get_dragon_tiger_context` 补充。

形态：专用补充源，照抄 SinaFundFlowFetcher——不提供日线行情
（`_fetch_raw_data` / `_normalize_data` 抛错），
`is_available_for_request("daily_data")` 返回 False，绝不会被日线路由选中。

Key 只走环境变量 HITHINK_API_KEY（工作流 Secret 注入），永不记日志、
永不 hardcode、永不写文件。无 Key 时 fetcher 不注册（见 base.py），
即使被直接实例化，三个公开方法也全部返回空结果、不发请求。

实测记录（2026-11-18，本机云电脑 + live API，600170.SH）：
- snapshot：last_price 2.07 等 4 只全通；字段见 _SNAPSHOT_*；无中文名。
- valuations/snapshot：pe_ttm 21.499 / pb_mrq 0.582，带中文名。
- indicators?report=2026-3：营收同比 -10.31%、归母净利同比 -31.17%、
  ROE 1.39%、毛利率 9.60%。注意实测 index_id 与文档表不一致：
  文档写 operating_income_yoy_growth_ratio，实测返回
  calculate_operating_income_yoy_growth_ratio；文档 net_profit_yoy_growth_ratio，
  实测 calculate_parent_holder_net_profit_yoy_growth_ratio。
  映射必须同时接受两套 id（见 _INDICATOR_ALIASES）。
- income-statements：period 取值就是字面量 annual / quarterly
  （2026-3、2026Q3、2026-09-30、20260930 均报 code=1002）。
  period=quarterly 返回最近多期列表，首项即最新期（2026 Q3）：
  operating_income 1417.82亿、parent_holder_net_profit 8.34亿。
  中式季报是累计值（Q3 = 1-9 月累计），与 AKShare 财务摘要口径一致。
- cash-flow-statements：首项 act_cash_flow_net -147.46亿（经营活动现金流量净额）。
- balance-sheets：已实测可用（assets_total / total_debt / holder_equity_total），
  但当前 bundle 无消费字段，本 fetcher 不调用，留待以后。
- dragon-tiger-list：2026-11-17 全榜 75 只，4 只自选股均未上榜。
- capital-flow/* 端点是端内专用，公开 Key 无权调用，本 fetcher 不碰资金流。

通用约定（官方契约 docs/api/）：HTTP 状态恒为 200，业务成败看 code 字段；
code=0 成功，1001 缺必填参数，1002 参数格式无效，2001 未认证，
2003 Key 无效，4001 限流，5xxx 服务端异常；时间戳毫秒、时区 Asia/Shanghai；
价格 CNY；null 表示未披露，不得补零。
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import (
    BaseFetcher,
    DataFetchError,
    normalize_stock_code,
)
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote

logger = logging.getLogger(__name__)

_HITHINK_BASE_URL = "https://fuyao.aicubes.cn"
_HITHINK_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# 单次请求超时（秒）。snapshot/估值走实时链路，用更短的预算；
# 财报/龙虎榜走基本面补充链路，另有 manager 侧剩余预算硬约束。
_HITHINK_SNAPSHOT_TIMEOUT_SECONDS = 8
_HITHINK_VALUATION_TIMEOUT_SECONDS = 5
_HITHINK_FINANCIAL_TIMEOUT_SECONDS = 10
_HITHINK_DRAGON_TIGER_TIMEOUT_SECONDS = 10

# 财务指标 index_id 映射：文档 id 在前，实测返回的 calculate_* 变体在后。
# 2026-11-18 实测：文档表与 live 返回不一致，两套都要认。
_INDICATOR_ALIASES = {
    "revenue_yoy": (
        "operating_income_yoy_growth_ratio",
        "calculate_operating_income_yoy_growth_ratio",
    ),
    "net_profit_yoy": (
        "net_profit_yoy_growth_ratio",
        "calculate_parent_holder_net_profit_yoy_growth_ratio",
    ),
    "roe": ("index_weighted_avg_roe",),
    "gross_margin": ("sale_gross_margin",),
}

# 利润表 / 现金流量表 period 取值：字面量 annual / quarterly。
# 2026-11-18 实测：2026-3、2026Q3、2026-09-30、20260930 均报 code=1002。
_STATEMENT_PERIOD_QUARTERLY = "quarterly"

_SHANGHAI_TZ = timezone(timedelta(hours=8))


def _to_thscode(code: str) -> Optional[str]:
    """6 位 A 股代码 → 同花顺 thscode（带交易所后缀）。

    官方文档只给 .SH/.SZ 示例；北交所（43/92/8 开头）暂不支持，返回 None。
    """
    if not code.isdigit() or len(code) != 6:
        return None
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return None


def _coerce_float(value: Any) -> Optional[float]:
    """转 float，转不了返回 None（不抛错；null 表示未披露，不得补零）。"""
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN guard
        return None
    return result


def _ms_to_date(ms: Any) -> Optional[str]:
    """毫秒时间戳（Asia/Shanghai）→ ISO 日期；注意不能用 UTC 口径换算。"""
    value = _coerce_float(ms)
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=_SHANGHAI_TZ).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _recent_report_periods(now: Optional[datetime] = None) -> List[str]:
    """最近两个已完结季度的 report 串（如 ["2026-3", "2026-2"]）。

    与 fundamental_adapter._recent_report_dates 同一口径（季度结束日倒推），
    只是输出 indicators 接口要的 yyyy-N（1=一季报、2=中报、3=三季报、4=年报）。
    """
    now = now or datetime.now()
    quarter_start = datetime(now.year, (now.month - 1) // 3 * 3 + 1, 1)
    periods = []
    for _ in range(2):
        end = quarter_start - timedelta(days=1)
        periods.append(f"{end.year}-{end.month // 3}")
        quarter_start = datetime(end.year, (end.month - 1) // 3 * 3 + 1, 1)
    return periods


def _latest_statement_item(items: Any) -> Optional[Dict[str, Any]]:
    """利润表/现金流量表返回最近多期列表，取 period_end_ms 最晚的一项。"""
    if not isinstance(items, list):
        return None
    rows = [row for row in items if isinstance(row, dict)]
    if not rows:
        return None
    rows.sort(key=lambda row: _coerce_float(row.get("period_end_ms")) or 0.0, reverse=True)
    return rows[0]


class HithinkFetcher(BaseFetcher):
    """
    同花顺 Financial-API 兜底源（专用补充数据源）

    优先级：8（默认，排在新浪 7 之后；免费源最后）
    数据来源：同花顺官方 hithink-finance（fuyao.aicubes.cn）
    """

    name = "HithinkFetcher"
    # HITHINK_PRIORITY 为空字符串时回落到 8：工作流里 vars 未设置会展开成
    # 空字符串传进来，int('') 会在导入时直接崩掉整次运行（run #14 实测）。
    _priority_raw = (os.getenv("HITHINK_PRIORITY") or "8").strip()
    priority = int(_priority_raw or "8")

    def __init__(self, api_key: str = "", priority: Optional[int] = None) -> None:
        """初始化。api_key 为空时本 fetcher 全程禁用（不发任何请求）。"""
        self._api_key = (api_key or "").strip()
        if priority is not None:
            self.priority = int(priority)

    def is_available_for_request(self, capability: str = "") -> bool:
        """Keep this supplement-only provider out of every daily-data route,
        and disable it entirely when no API key is configured."""
        if not self._api_key:
            return False
        return capability != "daily_data"

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise DataFetchError("HithinkFetcher 不提供日线行情数据（专用补充数据源）")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        raise DataFetchError("HithinkFetcher 不提供日线行情数据（专用补充数据源）")

    def _get_json(self, path: str, params: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """调 Hithink REST 端点，返回 data（code != 0 时抛错，由调用方转空结果）。

        Key 只出现在请求头里，永不记日志。
        """
        if not self._api_key:
            raise DataFetchError("HithinkFetcher 未配置 HITHINK_API_KEY")
        url = _HITHINK_BASE_URL + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _HITHINK_UA, "X-api-key": self._api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as exc:
            raise DataFetchError(f"Hithink HTTP {exc.code} for {path}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DataFetchError(f"Hithink request failed for {path}: {exc}") from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise DataFetchError(f"Hithink bad JSON for {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise DataFetchError(f"Hithink unexpected envelope for {path}")
        code = payload.get("code")
        if code != 0:
            raise DataFetchError(
                f"Hithink {path} code={code}: {payload.get('message')}"
            )
        data = payload.get("data")
        if data is None:
            raise DataFetchError(f"Hithink {path} returned null data")
        return data

    def get_realtime_quote(self, stock_code: str, source: str = "hithink") -> Optional[UnifiedRealtimeQuote]:
        """实时行情快照（+ 估值快照补 PE/PB/中文名）。

        失败返回 None（manager 继续下一源）。snapshot 失败时不调估值。
        """
        try:
            code = normalize_stock_code(stock_code)
        except Exception:
            return None
        thscode = _to_thscode(code)
        if thscode is None:
            logger.debug(f"Hithink 实时行情跳过非沪深代码 {code}")
            return None
        try:
            data = self._get_json(
                "/api/a-share/prices/snapshot",
                {"thscodes": thscode},
                _HITHINK_SNAPSHOT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(f"Hithink 实时行情请求失败 {code}: {exc}")
            return None
        items = data.get("item") if isinstance(data, dict) else None
        item = items[0] if isinstance(items, list) and items else None
        if not isinstance(item, dict):
            logger.warning(f"Hithink 实时行情无数据 {code}")
            return None
        price = _coerce_float(item.get("last_price"))
        if price is None or price <= 0:
            logger.warning(f"Hithink 实时行情价格无效 {code}")
            return None
        volume_raw = _coerce_float(item.get("volume"))
        quote = UnifiedRealtimeQuote(
            code=code,
            name="",
            source=RealtimeSource.HITHINK,
            price=price,
            change_pct=_coerce_float(item.get("price_change_ratio_pct")),
            change_amount=_coerce_float(item.get("price_change")),
            volume=int(volume_raw) if volume_raw is not None else None,
            amount=_coerce_float(item.get("turnover")),
            open_price=_coerce_float(item.get("open_price")),
            high=_coerce_float(item.get("high_price")),
            low=_coerce_float(item.get("low_price")),
            pre_close=_coerce_float(item.get("prev_price")),
            market="cn",
            currency="CNY",
        )
        # 估值快照补 PE/PB/中文名（snapshot 本身不带）；失败不影响 quote 返回。
        try:
            valuation = self._get_json(
                "/api/a-share/valuations/snapshot",
                {"thscodes": thscode},
                _HITHINK_VALUATION_TIMEOUT_SECONDS,
            )
            vitems = valuation.get("item") if isinstance(valuation, dict) else None
            vitem = vitems[0] if isinstance(vitems, list) and vitems else None
            if isinstance(vitem, dict):
                quote.pe_ratio = _coerce_float(vitem.get("pe_ttm"))
                quote.pb_ratio = _coerce_float(vitem.get("pb_mrq"))
                name = vitem.get("name")
                if isinstance(name, str) and name.strip():
                    quote.name = name.strip()
        except Exception as exc:
            logger.debug(f"Hithink 估值快照失败 {code}: {exc}")
        logger.info(f"Hithink 实时行情 {code}: 价格={quote.price} (来源: hithink)")
        return quote

    def _fetch_indicators(self, thscode: str) -> Dict[str, Optional[float]]:
        """财务指标：按最近两个已完结季度依次试，拿到非空 abilities 即停。"""
        for report in _recent_report_periods():
            try:
                data = self._get_json(
                    "/api/a-share/financials/indicators",
                    {"thscode": thscode, "report": report},
                    _HITHINK_FINANCIAL_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.debug(f"Hithink 财务指标请求失败 {thscode} report={report}: {exc}")
                continue
            abilities = data.get("abilities") if isinstance(data, dict) else None
            if not isinstance(abilities, list) or not abilities:
                continue
            flat: Dict[str, Optional[float]] = {}
            for block in abilities:
                if not isinstance(block, dict):
                    continue
                for indicator in block.get("indicators") or []:
                    if not isinstance(indicator, dict):
                        continue
                    index_id = indicator.get("index_id")
                    if isinstance(index_id, str) and index_id not in flat:
                        flat[index_id] = _coerce_float(indicator.get("value"))
            if any(v is not None for v in flat.values()):
                return flat
        return {}

    def _fetch_statement(self, endpoint: str, thscode: str) -> Optional[Dict[str, Any]]:
        """利润表 / 现金流量表：period=quarterly，取最新一期。"""
        try:
            data = self._get_json(
                f"/api/a-share/financials/{endpoint}",
                {"thscode": thscode, "period": _STATEMENT_PERIOD_QUARTERLY},
                _HITHINK_FINANCIAL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.debug(f"Hithink {endpoint} 请求失败 {thscode}: {exc}")
            return None
        items = data.get("item") if isinstance(data, dict) else None
        return _latest_statement_item(items)

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """基本面 bundle（与 AkshareFundamentalAdapter.get_fundamental_bundle 同构）。

        增长块来自财务指标，财报块来自利润表 + 现金流量表；
        institution 块恒为空（Hithink 公开端点无机构持股变动口径）。
        永不抛错：失败返回 not_supported，manager 按块填空。
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }
        try:
            code = normalize_stock_code(stock_code)
        except Exception as exc:
            result["errors"].append(f"Hithink bundle 代码规范化失败 {stock_code}: {exc}")
            return result
        thscode = _to_thscode(code)
        if thscode is None:
            result["errors"].append(f"Hithink bundle 仅支持沪深代码，跳过 {code}")
            return result

        indicators = self._fetch_indicators(thscode)
        growth: Dict[str, Any] = {}
        for field, aliases in _INDICATOR_ALIASES.items():
            for alias in aliases:
                value = indicators.get(alias)
                if value is not None:
                    growth[field] = value
                    break
        if growth:
            result["growth"] = growth
            result["source_chain"].append("growth:hithink")

        income = self._fetch_statement("income-statements", thscode)
        cashflow = self._fetch_statement("cash-flow-statements", thscode)
        financial_report: Dict[str, Any] = {}
        if isinstance(income, dict):
            financial_report["report_date"] = _ms_to_date(income.get("period_end_ms"))
            financial_report["revenue"] = _coerce_float(income.get("operating_income"))
            financial_report["net_profit_parent"] = _coerce_float(
                income.get("parent_holder_net_profit")
            )
        if isinstance(cashflow, dict):
            financial_report["operating_cash_flow"] = _coerce_float(
                cashflow.get("act_cash_flow_net")
            )
        roe = growth.get("roe")
        if roe is not None:
            financial_report["roe"] = roe
        financial_report = {
            key: value
            for key, value in financial_report.items()
            if value is not None
        }
        if financial_report:
            result["earnings"]["financial_report"] = financial_report
            result["source_chain"].append("earnings.financial_report:hithink")

        if result["growth"] or result["earnings"]:
            result["status"] = "partial"
        else:
            result["errors"].append(f"Hithink bundle 未拿到可用数据 {code}")
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """龙虎榜 flag（与 AkshareFundamentalAdapter.get_dragon_tiger_flag 同构）。

        只查最新一期榜单（端点按单日取）；lookback_days 仅为签名兼容而保留，
        不伪装成 20 天回看——Hithink 只在 AKShare 完全答不上来时才被调用。
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }
        try:
            code = normalize_stock_code(stock_code)
        except Exception as exc:
            result["errors"].append(f"Hithink dragon_tiger 代码规范化失败 {stock_code}: {exc}")
            return result
        thscode = _to_thscode(code)
        if thscode is None:
            result["errors"].append(f"Hithink dragon_tiger 仅支持沪深代码，跳过 {code}")
            return result
        try:
            data = self._get_json(
                "/api/a-share/special-data/dragon-tiger-list",
                {"board_type": "all"},
                _HITHINK_DRAGON_TIGER_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(f"Hithink 龙虎榜请求失败 {code}: {exc}")
            result["errors"].append(f"Hithink dragon_tiger request failed for {code}: {exc}")
            return result
        trade_date = data.get("trade_date") if isinstance(data, dict) else None
        items = data.get("stock_items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            result["errors"].append(f"Hithink dragon_tiger 返回结构异常 {code}")
            return result
        matched = any(
            isinstance(row, dict) and str(row.get("ticker")) == code for row in items
        )
        result["status"] = "ok"
        result["is_on_list"] = bool(matched)
        result["recent_count"] = 1 if matched else 0
        result["latest_date"] = trade_date if matched else None
        result["source_chain"].append("dragon_tiger:hithink")
        logger.info(
            f"Hithink 龙虎榜补充 {code}: {'上榜' if matched else '未上榜'}"
            + (f" {trade_date}" if trade_date else "")
        )
        return result


if __name__ == "__main__":
    # 手动测试代码（不联网的单元测试见 tests/test_hithink_port_fallbacks.py）
    logging.basicConfig(level=logging.DEBUG)

    fetcher = HithinkFetcher(api_key=os.getenv("HITHINK_API_KEY", ""))
    print(f"可用: {fetcher.is_available_for_request('realtime_quote')}")
    if fetcher.is_available_for_request("realtime_quote"):
        quote = fetcher.get_realtime_quote("600170")
        print(f"实时行情: {quote.to_dict() if quote else None}")
        bundle = fetcher.get_fundamental_bundle("600170")
        print(f"基本面: {json.dumps(bundle, ensure_ascii=False)}")
        flag = fetcher.get_dragon_tiger_flag("600170")
        print(f"龙虎榜: {json.dumps(flag, ensure_ascii=False)}")
