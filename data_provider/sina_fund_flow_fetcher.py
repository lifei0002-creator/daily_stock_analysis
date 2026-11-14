# -*- coding: utf-8 -*-
"""
===================================
SinaFundFlowFetcher - 资金流补充源（免费）
===================================

数据来源：新浪财经 MoneyFlow.ssl_qsfx_zjlrqs（日度资金流）
特点：免费、无需 key、无需注册、与东财不同风控面

定位：fundamental_adapter 的 AKShare 东财资金流（stock_individual_fund_flow）
在 GitHub 云服务器上常被东财限流返回空时，由 DataFetcherManager 的
_supplement_capital_flow_from_fetchers 兜底调用。本 fetcher 是“专用补充源”，
照抄 MiaoxiangFetcher 的形态：不提供日线行情（_fetch_raw_data /
_normalize_data 抛错），is_available_for_request("daily_data") 返回 False，
绝不会被日线路由选中。

移植来源：simonlin1212/a-stock-data SKILL.md「备用源速查」fund_flow_backup()
（Apache 2.0，原作者 simonlin1212）；函数体逐字移植，仅做两处适配：
1) 返回值改成与 MiaoxiangFetcher.get_capital_flow 同构的 payload
   （manager 的补充循环只认这个形状）；
2) 把 skill 的 raise 改成返回 {"status": "not_supported", "errors": [...]}，
   沿用项目的 fail-open 补充源约定。

单位（2026-11-14 实测验证）：netamount 单位是元，无需换算。
600170 当日 netamount=-77821815.39，ratioamount=-0.381052，
netamount / ratioamount ≈ 2.04 亿，与日报同日成交额 2.07 亿吻合。
"""

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import (
    BaseFetcher,
    DataFetchError,
    normalize_stock_code,
)

logger = logging.getLogger(__name__)

# 与 a-stock-data SKILL.md 共用的 UA（新浪接口同样认浏览器 UA）
_SINA_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_SINA_REFERER = "https://finance.sina.com.cn/"
_SINA_FUND_FLOW_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "MoneyFlow.ssl_qsfx_zjlrqs"
)


def _sina_market_prefix(code: str) -> str:
    """新浪 daima 前缀（skill 原样：92 先判北交所，误判成 sh/sz 时新浪返回空数组）。"""
    if code.startswith(("92", "8")):
        return "bj"
    if code.startswith(("6", "9")):
        return "sh"
    return "sz"


def _coerce_float(value: Any) -> Optional[float]:
    """把新浪返回的字符串/数字转 float，转不了返回 None（不抛错）。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SinaFundFlowFetcher(BaseFetcher):
    """
    新浪资金流补充源（专用补充数据源）

    优先级：7（默认，排在妙想 6 之后；付费源优先、免费源最后）
    数据来源：新浪财经 MoneyFlow
    """

    name = "SinaFundFlowFetcher"
    priority = int(os.getenv("SINA_FUND_FLOW_PRIORITY", "7"))
    capital_flow_markets = {"cn"}

    # 取数行数：10 日窗口 + 3~5 行冗余（防个别日期 netamount 为空）
    SINA_FUND_FLOW_DAYS = 15
    SINA_REQUEST_TIMEOUT_SECONDS = 10

    def __init__(self) -> None:
        """初始化（无 key、无配置）。"""

    def is_available_for_request(self, capability: str = "") -> bool:
        """Keep this supplement-only provider out of every daily-data route."""
        return capability != "daily_data"

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise DataFetchError("SinaFundFlowFetcher 不提供日线行情数据（专用补充数据源）")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        raise DataFetchError("SinaFundFlowFetcher 不提供日线行情数据（专用补充数据源）")

    @staticmethod
    def _window_sum(values: List[Optional[float]], window: int) -> Optional[float]:
        """窗口字段必须拿到完整交易日数据才输出（与 MiaoxiangFetcher 同一纪律）。"""
        if len(values) < window:
            return None
        recent = values[-window:]
        if any(v is None for v in recent):
            return None
        return float(sum(recent))

    def _fetch_sina_fund_flow(self, code: str) -> List[Dict[str, Any]]:
        """调新浪接口取最近 SINA_FUND_FLOW_DAYS 日资金流（skill fund_flow_backup 逐字移植）。

        Returns:
            按日期升序的 [{date, close, net_amount, turnover}]，net_amount 单位：元。
            失败抛异常，由调用方转成 not_supported 结果（不向上传播）。
        """
        pre = _sina_market_prefix(code) + code
        url = (
            f"{_SINA_FUND_FLOW_URL}?page=1&num={self.SINA_FUND_FLOW_DAYS}"
            f"&sort=opendate&asc=0&daima={pre}"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": _SINA_UA, "Referer": _SINA_REFERER}
        )
        with urllib.request.urlopen(req, timeout=self.SINA_REQUEST_TIMEOUT_SECONDS) as r:
            text = r.read().decode("utf-8", "ignore")
        arr = json.loads(text[text.index("["):text.rindex("]") + 1])
        rows = [
            {
                "date": x.get("opendate"),
                "close": _coerce_float(x.get("trade")),
                "net_amount": _coerce_float(x.get("netamount")),
                "turnover": _coerce_float(x.get("turnover")),
            }
            for x in arr
        ]
        rows = [row for row in rows if row.get("date")]
        rows.sort(key=lambda row: str(row["date"]))
        return rows

    def get_capital_flow(
        self, stock_code: str, top_n: int = 5
    ) -> Dict[str, Any]:
        """获取个股资金流（新浪补充源）。

        Args:
            stock_code: 股票代码，如 '600170'（manager 传入的已规范化代码）
            top_n: 行业排名返回条数（本源无行业排名数据，接受参数但不使用，
                仅为与 manager 的补充循环调用签名一致）

        Returns:
            与 MiaoxiangFetcher.get_capital_flow 同构的 payload：
            {
                "status": "ok" | "not_supported",
                "stock_flow": {"main_net_inflow": 元, "inflow_5d": 元|None,
                               "inflow_10d": 元|None},
                "sector_rankings": {"top": [], "bottom": []},
                "source_chain": ["capital_stock:sina_fund_flow"],
                "errors": [...],
            }
            注意：stock_flow 为 {} 时 manager 判定补充失败并继续尝试下一源。
        """
        errors: List[str] = []
        source_chain = ["capital_stock:sina_fund_flow"]
        try:
            code = normalize_stock_code(stock_code)
        except Exception as e:
            errors.append(f"SinaFundFlowFetcher 代码规范化失败 {stock_code}: {e}")
            return {
                "status": "not_supported",
                "stock_flow": {},
                "sector_rankings": {"top": [], "bottom": []},
                "source_chain": source_chain,
                "errors": errors,
            }
        # 仅 A 股 6 位数字代码；美股/港股/北交所不走本源
        if not code.isdigit() or len(code) != 6:
            errors.append(f"SinaFundFlowFetcher 仅支持 A 股代码，跳过 {code}")
            return {
                "status": "not_supported",
                "stock_flow": {},
                "sector_rankings": {"top": [], "bottom": []},
                "source_chain": source_chain,
                "errors": errors,
            }
        try:
            rows = self._fetch_sina_fund_flow(code)
        except Exception as e:
            logger.warning(f"Sina 资金流补充请求失败 {code}: {e}")
            errors.append(f"Sina fund flow request failed for {code}: {e}")
            return {
                "status": "not_supported",
                "stock_flow": {},
                "sector_rankings": {"top": [], "bottom": []},
                "source_chain": source_chain,
                "errors": errors,
            }

        net_values = [row.get("net_amount") for row in rows]
        usable = [v for v in net_values if v is not None]
        if not usable:
            errors.append(f"Sina fund flow returned no usable rows for {code}")
            return {
                "status": "not_supported",
                "stock_flow": {},
                "sector_rankings": {"top": [], "bottom": []},
                "source_chain": source_chain,
                "errors": errors,
            }

        # main_net_inflow 必须是最新一日的净额；若最新一日为空，不拿旧值顶替
        # （旧值会错标最新一日资金流向），直接判补充失败、fail-open。
        main_net_inflow = net_values[-1] if net_values else None
        if main_net_inflow is None:
            errors.append(f"Sina fund flow latest-day net_amount is null for {code}")
            return {
                "status": "not_supported",
                "stock_flow": {},
                "sector_rankings": {"top": [], "bottom": []},
                "source_chain": source_chain,
                "errors": errors,
            }
        inflow_5d = self._window_sum(net_values, 5)
        inflow_10d = self._window_sum(net_values, 10)
        stock_flow = {
            "main_net_inflow": main_net_inflow,
            "inflow_5d": inflow_5d,
            "inflow_10d": inflow_10d,
        }
        logger.info(
            f"Sina 资金流补充 {code}: 当日主力净流入={main_net_inflow / 1e8:.2f}亿"
            + (f", 5日={inflow_5d / 1e8:.2f}亿" if inflow_5d is not None else ", 5日=N/A")
            + (f", 10日={inflow_10d / 1e8:.2f}亿" if inflow_10d is not None else ", 10日=N/A")
        )
        return {
            "status": "ok",
            "stock_flow": stock_flow,
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": source_chain,
            "errors": errors,
        }


if __name__ == "__main__":
    # 手动测试代码（不联网的单元测试见 tests/test_asd_port_fallbacks.py）
    logging.basicConfig(level=logging.DEBUG)

    fetcher = SinaFundFlowFetcher()

    try:
        result = fetcher.get_capital_flow("600170")
        print(f"状态: {result['status']}")
        print(f"资金流: {result['stock_flow']}")
        print(f"来源链: {result['source_chain']}")
        print(f"错误: {result['errors']}")
    except Exception as e:
        print(f"获取失败: {e}")
