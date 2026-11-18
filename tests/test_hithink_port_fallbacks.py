# -*- coding: utf-8 -*-
"""同花顺 Financial-API（Hithink）兜底接入的离线单测（tests/test_hithink_port_fallbacks.py）

不联网：所有外部请求都用 unittest.mock 挡掉（mock HithinkFetcher._get_json，
或在 urlopen 层挡掉）。
运行：python -m pytest tests/test_hithink_port_fallbacks.py -v
  或 python -m unittest tests.test_hithink_port_fallbacks -v
注意：导入 data_provider.base 需要装好 requirements.txt（akshare、
fake-useragent、python-dotenv 等），与 tests/test_asd_port_fallbacks.py
的依赖面一致；本机裸 python 跑不起来时先建 venv 装 requirements。

覆盖：
- 无 Key 时禁用：三个公开方法全部返回空结果，且不发任何 HTTP 请求。
- 实时行情快照映射（live 形状）：价格/涨跌/量额/开高低/昨收 + 估值补 PE/PB/中文名；
  估值失败时 quote 仍返回；快照失败/空数据/无效价格/非沪深代码返回 None。
- 基本面 bundle 映射（live 形状）：abilities 分组指标（含实测 calculate_* id
  与文档 id 双认）、利润表 + 现金流量表、report_date 由 period_end_ms（上海时区）换算；
  报告期推导（2026-11-18 → ["2026-3", "2026-2"]）。
- 龙虎榜 flag 映射：上榜/未上榜/请求失败/结构异常。
- fail-open：HTTP 错误/超时/坏 JSON 时三个方法都不抛错，返回空结果。
- _merge_fundamental_bundle 按块填空：只填空块，不覆盖已有非空块。
- 注册：有 Key 时 manager 能按名找到 HithinkFetcher（priority 8），无 Key 时不注册；
  默认实时源优先级末位追加 hithink。
"""

import json
import os
import sys
import unittest
import urllib.error
from datetime import datetime
from pathlib import Path
from unittest import mock

# 允许直接 python tests/test_hithink_port_fallbacks.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.base import DataFetcherManager
from data_provider.hithink_fetcher import (
    HithinkFetcher,
    _recent_report_periods,
)
from src.config import Config

TEST_KEY = "test-key-123"


# ---- live 形状的 fixtures（2026-11-18 实测 600170.SH）-----------------------

SNAPSHOT_DATA = {
    "timestamp": 1794983429000,
    "total": 1,
    "item": [
        {
            "thscode": "600170.SH",
            "ticker": "600170",
            "volume": 55279915,
            "turnover": 115099534,
            "last_price": 2.08,
            "price_change": 0,
            "price_change_ratio_pct": 0,
            "open_price": 2.09,
            "high_price": 2.1,
            "low_price": 2.07,
            "prev_price": 2.08,
        }
    ],
}

VALUATION_DATA = {
    "timestamp": 1794983433000,
    "total": 1,
    "item": [
        {
            "thscode": "600170.SH",
            "ticker": "600170",
            "name": "上海建工",
            "pe_ttm": 21.499223,
            "pe_mrq": 16.627814,
            "pb_mrq": 0.582065,
            "ps_ttm": 0.097431,
            "pcf_ttm": 2.439284,
        }
    ],
}

INDICATORS_DATA = {
    "thscode": "600170.SH",
    "report": "2026-3",
    "abilities": [
        {
            "ability": "growth",
            "indicators": [
                {"index_id": "calculate_operating_income_yoy_growth_ratio", "value": "-10.30873800"},
                {"index_id": "calculate_operating_profit_yoy_growth_ratio", "value": "-13.47378200"},
                {"index_id": "total_assets_growth_ratio", "value": "-10.5715"},
                {"index_id": "calculate_parent_holder_net_profit_yoy_growth_ratio", "value": "-31.16517400"},
            ],
        },
        {
            "ability": "profitability",
            "indicators": [
                {"index_id": "total_assets_net_ratio", "value": "0.1609"},
                {"index_id": "index_deduct_weighted_avg_roe", "value": "-1.1800"},
                {"index_id": "sale_gross_margin", "value": "9.6048"},
                {"index_id": "sale_net_interest_ratio", "value": "0.3978"},
                {"index_id": "index_weighted_avg_roe", "value": "1.3900"},
            ],
        },
    ],
}

INCOME_DATA = {
    "item": [
        {
            "thscode": "600170.SH",
            "ticker": "600170",
            "period": "quarterly",
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
            "report_date_ms": 1793376000000,
            "period_end_ms": 1790697600000,
            "currency": "CNY",
            "operating_income": 141782354185.9,
            "parent_holder_net_profit": 833667361.82,
            "net_profit": 564042985.2,
        },
        {
            "thscode": "600170.SH",
            "ticker": "600170",
            "period": "quarterly",
            "fiscal_year": 2026,
            "fiscal_period": "Q2",
            "period_end_ms": 1782864000000,
            "operating_income": 90000000000.0,
            "parent_holder_net_profit": 500000000.0,
        },
    ]
}

CASHFLOW_DATA = {
    "item": [
        {
            "thscode": "600170.SH",
            "ticker": "600170",
            "period": "quarterly",
            "fiscal_year": 2026,
            "fiscal_period": "Q3",
            "period_end_ms": 1790697600000,
            "act_cash_flow_net": -14746413486.7,
            "invest_cash_flow_net": -2427487470.34,
        }
    ]
}

LHB_ON_LIST = {
    "trade_date": "2026-11-17",
    "stock_items": [
        {"thscode": "002242.SZ", "ticker": "002242", "name": "九阳股份"},
        {"thscode": "600170.SH", "ticker": "600170", "name": "上海建工"},
    ],
}

LHB_NOT_ON_LIST = {
    "trade_date": "2026-11-17",
    "stock_items": [
        {"thscode": "002242.SZ", "ticker": "002242", "name": "九阳股份"},
    ],
}


def _fetcher_with(routes):
    """返回一个 _get_json 被 routes 字典挡掉的 fetcher。

    routes: path -> 返回的 data（dict），或要抛出的异常。
    未在 routes 里的 path 会 AssertionError，防漏测。
    """
    fetcher = HithinkFetcher(api_key=TEST_KEY)

    def fake_get_json(path, params, timeout):
        if path in routes:
            result = routes[path]
            if isinstance(result, Exception):
                raise result
            return result
        raise AssertionError(f"unexpected Hithink path in test: {path}")

    fetcher._get_json = fake_get_json
    return fetcher


class DisabledWithoutKeyTest(unittest.TestCase):
    def test_all_methods_empty_and_no_http(self):
        fetcher = HithinkFetcher(api_key="")
        with mock.patch(
            "urllib.request.urlopen", side_effect=AssertionError("must not call HTTP")
        ):
            self.assertIsNone(fetcher.get_realtime_quote("600170"))
            bundle = fetcher.get_fundamental_bundle("600170")
            self.assertEqual(bundle["status"], "not_supported")
            self.assertEqual(bundle["growth"], {})
            self.assertEqual(bundle["earnings"], {})
            flag = fetcher.get_dragon_tiger_flag("600170")
            self.assertEqual(flag["status"], "not_supported")
            self.assertFalse(flag["is_on_list"])
        self.assertFalse(fetcher.is_available_for_request("realtime_quote"))
        self.assertFalse(fetcher.is_available_for_request("daily_data"))

    def test_available_with_key_except_daily_data(self):
        fetcher = HithinkFetcher(api_key=TEST_KEY)
        self.assertTrue(fetcher.is_available_for_request("realtime_quote"))
        self.assertTrue(fetcher.is_available_for_request())
        # 专用补充源：绝不进日线路由
        self.assertFalse(fetcher.is_available_for_request("daily_data"))


class QuoteMappingTest(unittest.TestCase):
    def test_snapshot_mapping_with_valuation(self):
        fetcher = _fetcher_with({
            "/api/a-share/prices/snapshot": SNAPSHOT_DATA,
            "/api/a-share/valuations/snapshot": VALUATION_DATA,
        })
        quote = fetcher.get_realtime_quote("600170")
        self.assertIsNotNone(quote)
        self.assertEqual(quote.code, "600170")
        self.assertEqual(quote.name, "上海建工")
        self.assertEqual(quote.price, 2.08)
        self.assertEqual(quote.change_pct, 0)
        self.assertEqual(quote.change_amount, 0)
        self.assertEqual(quote.volume, 55279915)
        self.assertIsInstance(quote.volume, int)
        self.assertEqual(quote.amount, 115099534)
        self.assertEqual(quote.open_price, 2.09)
        self.assertEqual(quote.high, 2.1)
        self.assertEqual(quote.low, 2.07)
        self.assertEqual(quote.pre_close, 2.08)
        self.assertAlmostEqual(quote.pe_ratio, 21.499223)
        self.assertAlmostEqual(quote.pb_ratio, 0.582065)
        from data_provider.realtime_types import RealtimeSource
        self.assertIs(quote.source, RealtimeSource.HITHINK)

    def test_valuation_failure_keeps_quote(self):
        fetcher = HithinkFetcher(api_key=TEST_KEY)

        def fake_get_json(path, params, timeout):
            if path.endswith("/prices/snapshot"):
                return SNAPSHOT_DATA
            raise ValueError("boom")

        fetcher._get_json = fake_get_json
        quote = fetcher.get_realtime_quote("600170")
        self.assertIsNotNone(quote)
        self.assertEqual(quote.price, 2.08)
        self.assertEqual(quote.name, "")
        self.assertIsNone(quote.pe_ratio)
        self.assertIsNone(quote.pb_ratio)

    def test_snapshot_failure_returns_none(self):
        fetcher = _fetcher_with({
            "/api/a-share/prices/snapshot": ValueError("boom"),
        })
        self.assertIsNone(fetcher.get_realtime_quote("600170"))

    def test_snapshot_empty_item_returns_none(self):
        fetcher = _fetcher_with({
            "/api/a-share/prices/snapshot": {"item": []},
        })
        self.assertIsNone(fetcher.get_realtime_quote("600170"))

    def test_invalid_price_returns_none(self):
        bad = {"item": [dict(SNAPSHOT_DATA["item"][0], last_price=0)]}
        fetcher = _fetcher_with({"/api/a-share/prices/snapshot": bad})
        self.assertIsNone(fetcher.get_realtime_quote("600170"))

    def test_non_cn_code_returns_none(self):
        fetcher = _fetcher_with({})
        self.assertIsNone(fetcher.get_realtime_quote("AAPL"))
        # 北交所暂不支持
        self.assertIsNone(fetcher.get_realtime_quote("430047"))


class FundamentalBundleTest(unittest.TestCase):
    def _bundle_fetcher(self):
        return _fetcher_with({
            "/api/a-share/financials/indicators": INDICATORS_DATA,
            "/api/a-share/financials/income-statements": INCOME_DATA,
            "/api/a-share/financials/cash-flow-statements": CASHFLOW_DATA,
        })

    def test_bundle_mapping_live_shape(self):
        bundle = self._bundle_fetcher().get_fundamental_bundle("600170")
        self.assertEqual(bundle["status"], "partial")
        growth = bundle["growth"]
        self.assertAlmostEqual(growth["revenue_yoy"], -10.308738, places=4)
        self.assertAlmostEqual(growth["net_profit_yoy"], -31.165174, places=4)
        self.assertAlmostEqual(growth["roe"], 1.39)
        self.assertAlmostEqual(growth["gross_margin"], 9.6048)
        report = bundle["earnings"]["financial_report"]
        self.assertAlmostEqual(report["revenue"], 141782354185.9)
        self.assertAlmostEqual(report["net_profit_parent"], 833667361.82)
        # period_end_ms 1790697600000 = 2026-09-30 00:00 Asia/Shanghai
        self.assertEqual(report["report_date"], "2026-09-30")
        self.assertAlmostEqual(report["operating_cash_flow"], -14746413486.7)
        self.assertAlmostEqual(report["roe"], 1.39)
        self.assertEqual(bundle["institution"], {})
        self.assertIn("growth:hithink", bundle["source_chain"])
        self.assertIn("earnings.financial_report:hithink", bundle["source_chain"])

    def test_documented_index_ids_also_accepted(self):
        documented = {
            "abilities": [
                {"ability": "growth", "indicators": [
                    {"index_id": "operating_income_yoy_growth_ratio", "value": "5.5"},
                    {"index_id": "net_profit_yoy_growth_ratio", "value": "7.7"},
                ]},
                {"ability": "profitability", "indicators": [
                    # roe / 毛利率只有实测这一套 id（文档表与 live 一致），不参与双认
                    {"index_id": "index_weighted_avg_roe", "value": "3.3"},
                    {"index_id": "sale_gross_margin", "value": "20.0"},
                ]},
            ]
        }
        fetcher = _fetcher_with({
            "/api/a-share/financials/indicators": documented,
            "/api/a-share/financials/income-statements": INCOME_DATA,
            "/api/a-share/financials/cash-flow-statements": CASHFLOW_DATA,
        })
        growth = fetcher.get_fundamental_bundle("600170")["growth"]
        self.assertAlmostEqual(growth["revenue_yoy"], 5.5)
        self.assertAlmostEqual(growth["net_profit_yoy"], 7.7)
        self.assertAlmostEqual(growth["roe"], 3.3)
        self.assertAlmostEqual(growth["gross_margin"], 20.0)

    def test_indicators_empty_returns_not_supported(self):
        fetcher = _fetcher_with({
            "/api/a-share/financials/indicators": {"abilities": []},
            "/api/a-share/financials/income-statements": {"item": []},
            "/api/a-share/financials/cash-flow-statements": {"item": []},
        })
        bundle = fetcher.get_fundamental_bundle("600170")
        self.assertEqual(bundle["status"], "not_supported")
        self.assertEqual(bundle["growth"], {})
        self.assertEqual(bundle["earnings"], {})

    def test_all_requests_fail_returns_not_supported(self):
        fetcher = _fetcher_with({
            "/api/a-share/financials/indicators": ValueError("boom"),
            "/api/a-share/financials/income-statements": ValueError("boom"),
            "/api/a-share/financials/cash-flow-statements": ValueError("boom"),
        })
        bundle = fetcher.get_fundamental_bundle("600170")
        self.assertEqual(bundle["status"], "not_supported")
        self.assertTrue(bundle["errors"])

    def test_recent_report_periods(self):
        # 2026-11-18（Q4）→ 最近两个已完结季度是 2026-3（三季报）、2026-2（中报）
        self.assertEqual(
            _recent_report_periods(datetime(2026, 11, 18)), ["2026-3", "2026-2"]
        )
        self.assertEqual(
            _recent_report_periods(datetime(2026, 2, 10)), ["2025-4", "2025-3"]
        )


class DragonTigerFlagTest(unittest.TestCase):
    def test_on_list(self):
        fetcher = _fetcher_with({
            "/api/a-share/special-data/dragon-tiger-list": LHB_ON_LIST,
        })
        flag = fetcher.get_dragon_tiger_flag("600170")
        self.assertEqual(flag["status"], "ok")
        self.assertTrue(flag["is_on_list"])
        self.assertEqual(flag["recent_count"], 1)
        self.assertEqual(flag["latest_date"], "2026-11-17")
        self.assertIn("dragon_tiger:hithink", flag["source_chain"])

    def test_not_on_list_is_valid_negative(self):
        fetcher = _fetcher_with({
            "/api/a-share/special-data/dragon-tiger-list": LHB_NOT_ON_LIST,
        })
        flag = fetcher.get_dragon_tiger_flag("600170")
        self.assertEqual(flag["status"], "ok")
        self.assertFalse(flag["is_on_list"])
        self.assertEqual(flag["recent_count"], 0)
        self.assertIsNone(flag["latest_date"])

    def test_request_failure_returns_not_supported(self):
        fetcher = _fetcher_with({
            "/api/a-share/special-data/dragon-tiger-list": ValueError("boom"),
        })
        flag = fetcher.get_dragon_tiger_flag("600170")
        self.assertEqual(flag["status"], "not_supported")
        self.assertFalse(flag["is_on_list"])
        self.assertTrue(flag["errors"])

    def test_malformed_payload_returns_not_supported(self):
        fetcher = _fetcher_with({
            "/api/a-share/special-data/dragon-tiger-list": {"trade_date": "2026-11-17"},
        })
        flag = fetcher.get_dragon_tiger_flag("600170")
        self.assertEqual(flag["status"], "not_supported")


class FailOpenHttpTest(unittest.TestCase):
    """_get_json 层：HTTP 错误 / 超时 / 坏 JSON 一律转 DataFetchError，
    调用方再转空结果；三个公开方法都不应把异常抛给 pipeline。"""

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_urlopen_http_error(self):
        fetcher = HithinkFetcher(api_key=TEST_KEY)
        err = urllib.error.HTTPError("http://x", 500, "err", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=err):
            self.assertIsNone(fetcher.get_realtime_quote("600170"))
            self.assertEqual(
                fetcher.get_fundamental_bundle("600170")["status"], "not_supported"
            )
            self.assertEqual(
                fetcher.get_dragon_tiger_flag("600170")["status"], "not_supported"
            )

    def test_urlopen_url_error_timeout(self):
        fetcher = HithinkFetcher(api_key=TEST_KEY)
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timed out"),
        ):
            self.assertIsNone(fetcher.get_realtime_quote("600170"))

    def test_bad_json_and_business_code(self):
        fetcher = HithinkFetcher(api_key=TEST_KEY)
        with mock.patch(
            "urllib.request.urlopen",
            return_value=self._FakeResponse(b"not json at all"),
        ):
            self.assertIsNone(fetcher.get_realtime_quote("600170"))
        bad_code = self._FakeResponse(
            json.dumps({"code": 2001, "message": "bad key", "data": None}).encode()
        )
        with mock.patch("urllib.request.urlopen", return_value=bad_code):
            self.assertIsNone(fetcher.get_realtime_quote("600170"))

    def test_api_key_never_in_logs_or_errors(self):
        fetcher = HithinkFetcher(api_key=TEST_KEY)
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("boom"),
        ):
            with self.assertLogs("data_provider.hithink_fetcher", level="WARNING") as logs:
                fetcher.get_realtime_quote("600170")
        self.assertNotIn(TEST_KEY, "\n".join(logs.output))


class MergeFundamentalBundleTest(unittest.TestCase):
    def test_fills_empty_blocks(self):
        supplement = {
            "growth": {"revenue_yoy": -10.3, "roe": 1.39},
            "earnings": {"financial_report": {"revenue": 100.0}, "roe": 1.39},
            "institution": {},
        }
        growth, earnings, institution, filled = DataFetcherManager._merge_fundamental_bundle(
            {}, {}, {}, supplement
        )
        self.assertEqual(filled, ["growth", "earnings"])
        self.assertEqual(growth["revenue_yoy"], -10.3)
        self.assertEqual(earnings["financial_report"]["revenue"], 100.0)

    def test_keeps_existing_nonempty_blocks(self):
        existing_growth = {"revenue_yoy": 5.0}
        supplement = {
            "growth": {"revenue_yoy": -10.3},
            "earnings": {"financial_report": {"revenue": 100.0}, "roe": 1.39},
            "institution": {},
        }
        growth, earnings, institution, filled = DataFetcherManager._merge_fundamental_bundle(
            existing_growth, {}, {}, supplement
        )
        self.assertEqual(filled, ["earnings"])
        self.assertEqual(growth, {"revenue_yoy": 5.0})

    def test_empty_supplement_fills_nothing(self):
        growth, earnings, institution, filled = DataFetcherManager._merge_fundamental_bundle(
            {}, {}, {}, {"growth": {}, "earnings": {}, "institution": {}}
        )
        self.assertEqual(filled, [])
        self.assertEqual(growth, {})


class RegistrationTest(unittest.TestCase):
    def _manager_with_env(self, env):
        """按 env 构造一个真实 DataFetcherManager（_init_default_fetchers
        走 function-local 的 src.config.get_config，用 patch 挡掉）。"""
        merged = dict(os.environ)
        merged.pop("HITHINK_API_KEY", None)
        merged.pop("HITHINK_PRIORITY", None)
        merged.update(env)
        with mock.patch.dict(os.environ, merged, clear=True):
            config = Config._load_from_env()
            with mock.patch("src.config.get_config", return_value=config):
                return DataFetcherManager()

    def test_registered_with_key(self):
        manager = self._manager_with_env({"HITHINK_API_KEY": TEST_KEY})
        fetcher = manager._get_fetcher_by_name("HithinkFetcher")
        self.assertIsNotNone(fetcher)
        self.assertEqual(fetcher.priority, 8)
        self.assertTrue(fetcher.is_available_for_request("realtime_quote"))
        self.assertFalse(fetcher.is_available_for_request("daily_data"))

    def test_not_registered_without_key(self):
        manager = self._manager_with_env({})
        self.assertIsNone(manager._get_fetcher_by_name("HithinkFetcher"))

    def test_default_realtime_priority_ends_with_hithink(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("REALTIME_SOURCE_PRIORITY", None)
            config = Config._load_from_env()
        self.assertTrue(config.realtime_source_priority.endswith(",hithink"))
        self.assertEqual(
            config.realtime_source_priority,
            "tencent,akshare_sina,efinance,akshare_em,hithink",
        )


class DragonTigerWiringTest(unittest.TestCase):
    """base.py 接线：adapter 答不上来时才调 Hithink；adapter 明确答复
    “未上榜”时不调 Hithink。"""

    def _manager(self):
        manager = DataFetcherManager(fetchers=[HithinkFetcher(api_key=TEST_KEY)])
        return manager

    def _patch_config(self):
        return mock.patch(
            "src.config.get_config",
            return_value=_FakeConfig(),
        )

    def test_supplement_used_when_adapter_not_supported(self):
        manager = self._manager()
        hithink = manager._get_fetcher_by_name("HithinkFetcher")
        with mock.patch.object(
            hithink, "get_dragon_tiger_flag",
            return_value={
                "status": "ok", "is_on_list": True, "recent_count": 1,
                "latest_date": "2026-11-17",
                "source_chain": ["dragon_tiger:hithink"], "errors": [],
            },
        ) as hithink_mock, mock.patch.object(
            manager._fundamental_adapter, "get_dragon_tiger_flag",
            return_value={
                "status": "not_supported", "is_on_list": False, "recent_count": 0,
                "latest_date": None, "source_chain": [], "errors": ["akshare: empty"],
            },
        ), self._patch_config():
            ctx = manager.get_dragon_tiger_context("600170", budget_seconds=30)
        hithink_mock.assert_called_once()
        self.assertEqual(ctx["status"], "ok")
        self.assertTrue(ctx["data"]["is_on_list"])
        self.assertEqual(ctx["data"]["latest_date"], "2026-11-17")

    def test_supplement_skipped_when_adapter_answers_not_on_list(self):
        manager = self._manager()
        hithink = manager._get_fetcher_by_name("HithinkFetcher")
        with mock.patch.object(
            hithink, "get_dragon_tiger_flag",
            return_value={
                "status": "ok", "is_on_list": True, "recent_count": 1,
                "latest_date": "2026-11-17",
                "source_chain": ["dragon_tiger:hithink"], "errors": [],
            },
        ) as hithink_mock, mock.patch.object(
            manager._fundamental_adapter, "get_dragon_tiger_flag",
            return_value={
                "status": "ok", "is_on_list": False, "recent_count": 0,
                "latest_date": "2026-11-17",
                "source_chain": ["dragon_tiger:akshare"], "errors": [],
            },
        ), self._patch_config():
            ctx = manager.get_dragon_tiger_context("600170", budget_seconds=30)
        hithink_mock.assert_not_called()
        self.assertEqual(ctx["status"], "ok")
        self.assertFalse(ctx["data"]["is_on_list"])


class _FakeConfig:
    fundamental_fetch_timeout_seconds = 30.0
    fundamental_retry_max = 1


if __name__ == "__main__":
    unittest.main()
