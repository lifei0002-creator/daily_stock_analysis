# -*- coding: utf-8 -*-
"""a-stock-data 三段移植的离线单测（tests/test_asd_port_fallbacks.py）

不联网：所有外部请求都用 unittest.mock 挡掉。
运行：python -m pytest tests/test_asd_port_fallbacks.py -v
  或 python -m unittest tests.test_asd_port_fallbacks -v

覆盖：
- 筹码本地推演 _chip_distribution_local：硬约束（获利比例∈[0,1]、成本区间包含、
  集中度 90%≥70%）、缺列/空表抛错、输入倒序结果不变。
- BaostockFetcher.get_chip_distribution：守卫（非 A 股返回 None、有效交易日不足
  返回 None）与正常映射（source="baostock_local"、前复权查询字段齐全）。
- 新浪资金流 SinaFundFlowFetcher.get_capital_flow：主/5日/10日窗口、窗口不满为 None、
  空结果 not_supported、市场前缀 sh/sz/bj、最新一日为空不拿旧值顶替。
- 东财个股新闻：JSONP 解析 + 去 HTML 标签 + 截断、请求失败返回 []、格式化、
  run_stock_news_fallback 的适用性守卫（仅 A 股非指数）。
"""

import json
import sys
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# 允许直接 python tests/test_asd_port_fallbacks.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.baostock_fetcher import BaostockFetcher, _chip_distribution_local
from data_provider.sina_fund_flow_fetcher import SinaFundFlowFetcher
from src.services import stock_news_fallback
from src.services.stock_news_fallback import (
    fetch_eastmoney_stock_news,
    format_stock_news_context,
    run_stock_news_fallback,
)


def _make_kline_df(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """合成 K 线：收盘价 10→13 缓涨，turn 0.5%~3%，日期升序。"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2026-05-01", periods=n_days, freq="B")
    close = np.linspace(10.0, 13.0, n_days) + rng.normal(0, 0.05, n_days)
    close = np.maximum(close, 1.0)
    high = close * 1.01
    low = close * 0.99
    turn = rng.uniform(0.5, 3.0, n_days)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * 0.995,
        "high": high,
        "low": low,
        "close": close,
        "turn": turn,
    })


class ChipDistributionLocalTest(unittest.TestCase):
    def test_hard_constraints(self):
        df = _make_kline_df()
        r = _chip_distribution_local(df)
        self.assertGreaterEqual(r["profit_ratio"], 0.0)
        self.assertLessEqual(r["profit_ratio"], 1.0)
        self.assertGreater(r["avg_cost"], 0.0)
        self.assertGreaterEqual(r["avg_cost"], float(df["low"].min()))
        self.assertLessEqual(r["avg_cost"], float(df["high"].max()))
        # 90% 成本区间必须包含 70% 成本区间
        self.assertLessEqual(r["cost_90"][0], r["cost_70"][0])
        self.assertGreaterEqual(r["cost_90"][1], r["cost_70"][1])
        # 集中度：90% 必大于等于 70%（skill 自述硬约束）
        self.assertGreaterEqual(r["concentration_90"], r["concentration_70"])
        self.assertGreater(r["peak_price"], 0.0)

    def test_descending_input_gives_same_result(self):
        df = _make_kline_df()
        asc = _chip_distribution_local(df)
        desc = _chip_distribution_local(df.iloc[::-1].reset_index(drop=True))
        self.assertAlmostEqual(asc["profit_ratio"], desc["profit_ratio"], places=9)
        self.assertAlmostEqual(asc["avg_cost"], desc["avg_cost"], places=9)

    def test_missing_column_raises(self):
        df = _make_kline_df().drop(columns=["turn"])
        with self.assertRaises(ValueError):
            _chip_distribution_local(df)

    def test_empty_raises(self):
        df = _make_kline_df().iloc[0:0]
        with self.assertRaises(ValueError):
            _chip_distribution_local(df)


class _FakeRS:
    """模拟 baostock query_history_k_data_plus 的返回对象。"""

    def __init__(self, rows, fields):
        self.error_code = "0"
        self.error_msg = ""
        self.fields = fields
        self._rows = rows
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


class _FakeBS:
    """模拟 baostock 模块（只实现筹码查询用到的一个方法）。"""

    def __init__(self, rs):
        self._rs = rs
        self.last_kwargs = None

    def query_history_k_data_plus(self, code, fields, start_date, end_date,
                                  frequency, adjustflag):
        self.last_kwargs = dict(
            code=code, fields=fields, start_date=start_date,
            end_date=end_date, frequency=frequency, adjustflag=adjustflag,
        )
        return self._rs


def _make_bs_rows(n_days: int = 120, n_suspended: int = 5) -> list:
    """模拟 baostock 返回的行（字符串值，含停牌日 tradestatus=0）。"""
    df = _make_kline_df(n_days)
    rows = []
    for _, row in df.iterrows():
        rows.append([
            row["date"], f"{row['open']:.2f}", f"{row['high']:.2f}",
            f"{row['low']:.2f}", f"{row['close']:.2f}", f"{row['turn']:.2f}", "1",
        ])
    for i in range(n_suspended):
        rows.append(["2026-05-0%d" % (i + 1), "10.00", "10.00", "10.00", "10.00", "0.00", "0"])
    return rows


class BaostockChipFetcherTest(unittest.TestCase):
    def _fetcher_with_fake_bs(self, rows):
        fetcher = BaostockFetcher()
        fields = ["date", "open", "high", "low", "close", "turn", "tradestatus"]
        fake_bs = _FakeBS(_FakeRS(rows, fields))

        @contextmanager
        def _fake_session():
            yield fake_bs

        fetcher._baostock_session = _fake_session
        return fetcher, fake_bs

    def test_normal_mapping(self):
        fetcher, fake_bs = self._fetcher_with_fake_bs(_make_bs_rows())
        result = fetcher.get_chip_distribution("600170")
        self.assertIsNotNone(result)
        self.assertEqual(result.code, "600170")
        self.assertEqual(result.source, "baostock_local")
        self.assertEqual(result.date, "2026-10-15")  # 120 个交易日的最后一天
        self.assertGreaterEqual(result.profit_ratio, 0.0)
        self.assertLessEqual(result.profit_ratio, 1.0)
        self.assertGreater(result.avg_cost, 0.0)
        # 查询必须是前复权 + 含 turn/tradestatus 字段
        self.assertEqual(fake_bs.last_kwargs["adjustflag"], "2")
        self.assertIn("turn", fake_bs.last_kwargs["fields"])
        self.assertIn("tradestatus", fake_bs.last_kwargs["fields"])
        self.assertEqual(fake_bs.last_kwargs["code"], "sh.600170")

    def test_non_a_share_returns_none(self):
        fetcher, _ = self._fetcher_with_fake_bs(_make_bs_rows())
        self.assertIsNone(fetcher.get_chip_distribution("AAPL"))

    def test_too_few_trading_days_returns_none(self):
        fetcher, _ = self._fetcher_with_fake_bs(_make_bs_rows(n_days=30))
        self.assertIsNone(fetcher.get_chip_distribution("600170"))


class _FakeHTTPResponse:
    """模拟 urllib.response：with 语句 + read() 返回 bytes。"""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_sina_rows(n_days: int = 12, latest_null: bool = False) -> list:
    """模拟新浪返回：按 opendate 倒序，netamount 为 1e6 的倍数（便于验算窗口和）。"""
    dates = pd.date_range("2026-11-13", periods=n_days, freq="B")[::-1]
    rows = []
    for i, d in enumerate(dates):
        # dates 倒序：i=0 是最新一日，对应 net_amount = n_days * 1e6
        net = (n_days - i) * 1_000_000.0
        if latest_null and i == 0:
            net = None
        rows.append({
            "opendate": d.strftime("%Y-%m-%d"),
            "trade": "2.09",
            "netamount": None if net is None else str(net),
            "turnover": "109.471",
            "ratioamount": "-0.381052",
        })
    return rows


class SinaFundFlowTest(unittest.TestCase):
    def _run_with_rows(self, code, rows):
        payload = json.dumps(rows).encode("utf-8")
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeHTTPResponse(payload)) as urlopen:
            fetcher = SinaFundFlowFetcher()
            result = fetcher.get_capital_flow(code)
            called_url = (urlopen.call_args[0][0].full_url
                          if urlopen.call_args is not None else None)
        return result, called_url

    def test_windows(self):
        result, url = self._run_with_rows("600170", _make_sina_rows(12))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source_chain"], ["capital_stock:sina_fund_flow"])
        self.assertIn("daima=sh600170", url)
        flow = result["stock_flow"]
        self.assertEqual(flow["main_net_inflow"], 12_000_000.0)   # 最新一日
        self.assertEqual(flow["inflow_5d"], 50_000_000.0)         # 8+9+10+11+12
        self.assertEqual(flow["inflow_10d"], 75_000_000.0)        # 3+...+12

    def test_short_window_returns_none_windows(self):
        result, _ = self._run_with_rows("600170", _make_sina_rows(3))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stock_flow"]["main_net_inflow"], 3_000_000.0)
        self.assertIsNone(result["stock_flow"]["inflow_5d"])
        self.assertIsNone(result["stock_flow"]["inflow_10d"])

    def test_latest_null_is_not_supported(self):
        # 最新一日 netamount 为空：不拿旧值顶替，直接判补充失败
        result, _ = self._run_with_rows("600170", _make_sina_rows(12, latest_null=True))
        self.assertEqual(result["status"], "not_supported")
        self.assertEqual(result["stock_flow"], {})

    def test_empty_is_not_supported(self):
        result, _ = self._run_with_rows("600170", [])
        self.assertEqual(result["status"], "not_supported")
        self.assertEqual(result["stock_flow"], {})

    def test_market_prefix(self):
        _, url_sz = self._run_with_rows("000001", _make_sina_rows(12))
        self.assertIn("daima=sz000001", url_sz)
        _, url_bj = self._run_with_rows("920002", _make_sina_rows(12))
        self.assertIn("daima=bj920002", url_bj)

    def test_non_a_share_is_not_supported(self):
        result, _ = self._run_with_rows("AAPL", _make_sina_rows(12))
        self.assertEqual(result["status"], "not_supported")


class _FakeEMResponse:
    def __init__(self, text):
        self.text = text


def _make_em_jsonp(n_items: int = 2) -> str:
    articles = []
    for i in range(n_items):
        articles.append({
            "title": f"<b>测试标题{i}</b>",
            "content": f"<p>正文内容{i}，{'x' * 300}</p>",
            "date": "2026-11-14 10:00:00",
            "mediaName": "证券时报",
            "url": f"https://example.com/{i}",
        })
    return "jQuery_news(" + json.dumps(
        {"result": {"cmsArticleWebOld": articles}}, ensure_ascii=False) + ")"


class EastmoneyNewsTest(unittest.TestCase):
    def setUp(self):
        # 关掉节流，避免单测等待
        self._interval_patch = mock.patch.object(stock_news_fallback, "EM_MIN_INTERVAL", 0)
        self._interval_patch.start()
        self.addCleanup(self._interval_patch.stop)

    def test_parse_and_strip_html(self):
        with mock.patch.object(stock_news_fallback._EM_SESSION, "get",
                               return_value=_FakeEMResponse(_make_em_jsonp(2))):
            rows = fetch_eastmoney_stock_news("600170")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "测试标题0")
        self.assertNotIn("<", rows[0]["content"])
        self.assertLessEqual(len(rows[0]["content"]), 200)  # 截断
        self.assertEqual(rows[0]["source"], "证券时报")
        self.assertEqual(rows[0]["time"], "2026-11-14 10:00:00")

    def test_request_failure_returns_empty(self):
        with mock.patch.object(stock_news_fallback._EM_SESSION, "get",
                               side_effect=Exception("boom")):
            self.assertEqual(fetch_eastmoney_stock_news("600170"), [])

    def test_empty_articles_returns_empty(self):
        payload = "jQuery_news(" + json.dumps({"result": {}}) + ")"
        with mock.patch.object(stock_news_fallback._EM_SESSION, "get",
                               return_value=_FakeEMResponse(payload)):
            self.assertEqual(fetch_eastmoney_stock_news("600170"), [])

    def test_format_empty(self):
        self.assertEqual(format_stock_news_context([], "上海建工"), "")

    def test_format_shows_up_to_five(self):
        news = [
            {"title": f"标题{i}", "content": f"内容{i}",
             "time": "2026-11-14", "source": "媒体", "url": ""}
            for i in range(6)
        ]
        text = format_stock_news_context(news, "上海建工")
        self.assertIn("上海建工", text)
        self.assertIn("东财个股新闻", text)
        self.assertIn("标题0", text)
        self.assertIn("标题4", text)
        self.assertNotIn("标题5", text)  # 最多展示 5 条

    def test_fallback_gate_non_cn(self):
        with mock.patch.object(stock_news_fallback, "fetch_eastmoney_stock_news",
                               side_effect=AssertionError("不应调用")):
            self.assertEqual(
                run_stock_news_fallback("AAPL", "苹果", "us", False), (None, None))

    def test_fallback_gate_index(self):
        with mock.patch.object(stock_news_fallback, "fetch_eastmoney_stock_news",
                               side_effect=AssertionError("不应调用")):
            self.assertEqual(
                run_stock_news_fallback("000001", "上证指数", "cn", True), (None, None))

    def test_fallback_success_and_zero(self):
        news = [{"title": "标题0", "content": "内容0",
                 "time": "2026-11-14", "source": "媒体", "url": ""}]
        with mock.patch.object(stock_news_fallback, "fetch_eastmoney_stock_news",
                               return_value=news):
            ctx, count = run_stock_news_fallback("600170", "上海建工", "cn", False)
        self.assertEqual(count, 1)
        self.assertIn("标题0", ctx)
        with mock.patch.object(stock_news_fallback, "fetch_eastmoney_stock_news",
                               return_value=[]):
            self.assertEqual(
                run_stock_news_fallback("600170", "上海建工", "cn", False), (None, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
