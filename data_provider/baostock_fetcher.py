# -*- coding: utf-8 -*-
"""
===================================
BaostockFetcher - 备用数据源 2 (Priority 3)
===================================

数据来源：证券宝（Baostock）
特点：免费、无需 Token、需要登录管理
优点：稳定、无配额限制

关键策略：
1. 管理 bs.login() 和 bs.logout() 生命周期
2. 使用上下文管理器防止连接泄露
3. 失败后指数退避重试
"""

import logging
import re
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Generator

import numpy as np
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import (
    BaseFetcher,
    DataFetchError,
    STANDARD_COLUMNS,
    is_bse_code,
    normalize_stock_code,
    _is_hk_market,
)
import os
from .realtime_types import ChipDistribution

logger = logging.getLogger(__name__)


def _is_us_code(stock_code: str) -> bool:
    """
    判断代码是否为美股
    
    美股代码规则：
    - 1-5个大写字母，如 'AAPL', 'TSLA'
    - 可能包含 '.'，如 'BRK.B'
    """
    code = stock_code.strip().upper()
    return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', code))


# ----------------------------------------------------------------------
# 筹码分布本地推演（移植自 simonlin1212/a-stock-data SKILL.md §4.6，
# Apache 2.0，原作者 simonlin1212；函数体逐字移植，仅改名）
# 东财没有公开 CYQ 接口（作者实测 push2 cyq/get 均 404），故本地推演。
# ----------------------------------------------------------------------


def _triangular_weights(grid: np.ndarray, low: float, high: float, avg: float) -> np.ndarray:
    """当日筹码在价格网格上的三角分布权重（峰值在均价，面积归一）"""
    w = np.zeros_like(grid)
    if not np.isfinite([low, high, avg]).all() or high < low:
        return w
    if high - low < 1e-9:                       # 一字板：全部堆在一个价位
        w[np.argmin(np.abs(grid - low))] = 1.0
        return w
    avg = min(max(avg, low), high)              # 均价必须落在当日区间内
    left = (grid >= low) & (grid <= avg)
    right = (grid > avg) & (grid <= high)
    if avg - low > 1e-9:
        w[left] = (grid[left] - low) / (avg - low)
    else:
        w[left] = 1.0
    if high - avg > 1e-9:
        w[right] = (high - grid[right]) / (high - avg)
    else:
        w[right] = 1.0
    total = w.sum()
    if total > 0:
        return w / total
    # 兜底：当日振幅窄于网格步长时，可能一个网格点都没落进 [low, high]，
    #    权重会全为 0。若就此跳过该日，连它的换手衰减也会一并丢失 ——
    #    低波动标的（银行股等）+ 长窗口下这会累积成很大的偏差。映射到最近网格点。
    w[np.argmin(np.abs(grid - avg))] = 1.0
    return w


def _chip_distribution_local(df: pd.DataFrame, grid_size: int = 300, decay: float = 1.0) -> dict:
    """筹码分布 — df 需含 date/high/low/close/turn（turn 为百分数，0.31 表示 0.31%）

    decay: 换手衰减系数。1.0=按真实换手率换手；同花顺口径常用 1.5~2.0 加快历史筹码消散。
    """
    # 必须带 date 并按时间升序：换手衰减是有方向的时序递推，
    # 若传入常见的「最新在前」倒序，衰减会反向推、且 close.iloc[-1] 会把最老的
    # 收盘价当成现价 —— 结果完全错却不会报错。这里强制要求 date 并自行排序。
    need = {"date", "high", "low", "close", "turn"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"chip_distribution 缺少列: {sorted(missing)}（date 用于强制时间升序）")
    d = df.dropna(subset=["high", "low", "close", "turn"]).copy()
    d = d[d["high"] > 0]
    if d.empty:
        raise ValueError("chip_distribution: 有效行数为 0（检查是否全是停牌日，或字段类型不对）")
    d = d.sort_values("date").reset_index(drop=True)

    lo, hi = float(d["low"].min()), float(d["high"].max())
    pad = (hi - lo) * 0.02 or max(lo * 0.02, 0.01)
    grid = np.linspace(lo - pad, hi + pad, grid_size)

    # 初始筹码必须播种成「首日全部流通盘」，不能从全零开始。
    # 从零起步等于假设窗口之前没有任何持仓，再把窗口内的少量换手归一化成 100%：
    # 两个 1% 换手日（价 10 和 100）会被算成约 50/50，而真实情况是约 99% 仍在 10 附近。
    chips = None
    for row in d.itertuples(index=False):
        t = float(row.turn) / 100.0 * decay
        t = min(max(t, 0.0), 1.0)               # 换手率兜到 [0,1]，防异常值把筹码一次清零
        avg = (float(row.high) + float(row.low) + float(row.close)) / 3.0
        w = _triangular_weights(grid, float(row.low), float(row.high), avg)
        if w.sum() <= 0:
            continue
        if chips is None:
            chips = w.copy()                    # 首日分布 = 期初全部流通筹码
            continue
        chips = chips * (1.0 - t) + w * t
    if chips is None:
        raise RuntimeError("chip_distribution: 所有交易日的价格区间都无效，无法构建分布")

    total = chips.sum()
    if total <= 0:
        raise RuntimeError("chip_distribution: 筹码总量为 0，无法计算指标")
    chips = chips / total

    price = float(d["close"].iloc[-1])
    cum = np.cumsum(chips)

    def price_at(q: float) -> float:
        return float(np.interp(q, cum, grid))

    p05, p15, p85, p95 = (price_at(q) for q in (0.05, 0.15, 0.85, 0.95))
    peak_i = int(np.argmax(chips))
    return {
        "price": price,
        "profit_ratio": float(chips[grid <= price].sum()),      # 获利比例
        "avg_cost": float((grid * chips).sum()),                # 平均成本
        "cost_90": (p05, p95),
        "cost_70": (p15, p85),
        "concentration_90": float((p95 - p05) / (p95 + p05)) if p95 + p05 else None,
        "concentration_70": float((p85 - p15) / (p85 + p15)) if p85 + p15 else None,
        "peak_price": float(grid[peak_i]),                      # 筹码峰
        "histogram": [(float(pp), float(cc)) for pp, cc in zip(grid, chips) if cc > 1e-6],
    }



class BaostockFetcher(BaseFetcher):
    """
    Baostock 数据源实现
    
    优先级：3
    数据来源：证券宝 Baostock API
    
    关键策略：
    - 使用上下文管理器管理连接生命周期
    - 每次请求都重新登录/登出，防止连接泄露
    - 失败后指数退避重试
    
    Baostock 特点：
    - 免费、无需注册
    - 需要显式登录/登出
    - 数据更新略有延迟（T+1）
    """
    
    name = "BaostockFetcher"
    priority = int(os.getenv("BAOSTOCK_PRIORITY", "3"))
    
    def __init__(self):
        """初始化 BaostockFetcher"""
        self._bs_module = None
    
    def _get_baostock(self):
        """
        延迟加载 baostock 模块
        
        只在首次使用时导入，避免未安装时报错
        """
        if self._bs_module is None:
            import baostock as bs
            self._bs_module = bs
        return self._bs_module
    
    @contextmanager
    def _baostock_session(self) -> Generator:
        """
        Baostock 连接上下文管理器
        
        确保：
        1. 进入上下文时自动登录
        2. 退出上下文时自动登出
        3. 异常时也能正确登出
        
        使用示例：
            with self._baostock_session():
                # 在这里执行数据查询
        """
        bs = self._get_baostock()
        login_result = None
        
        try:
            # 登录 Baostock
            login_result = bs.login()
            
            if login_result.error_code != '0':
                raise DataFetchError(f"Baostock 登录失败: {login_result.error_msg}")
            
            logger.debug("Baostock 登录成功")
            
            yield bs
            
        finally:
            # 确保登出，防止连接泄露
            try:
                logout_result = bs.logout()
                if logout_result.error_code == '0':
                    logger.debug("Baostock 登出成功")
                else:
                    logger.warning(f"Baostock 登出异常: {logout_result.error_msg}")
            except Exception as e:
                logger.warning(f"Baostock 登出时发生错误: {e}")
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Baostock 格式
        
        Baostock 要求的格式：
        - 沪市：sh.600519
        - 深市：sz.000001
        
        Args:
            stock_code: 原始代码，如 '600519', '000001'
            
        Returns:
            Baostock 格式代码，如 'sh.600519', 'sz.000001'
        """
        raw_code = stock_code.strip()
        upper = raw_code.upper()

        # HK stocks are not supported by Baostock
        if _is_hk_market(raw_code):
            raise DataFetchError(f"BaostockFetcher 不支持港股 {raw_code}，请使用 AkshareFetcher")

        # 保留既有小写 baostock 格式输入的内部容错，但用户配置仍推荐 6 位裸代码。
        if raw_code.startswith(('sh.', 'sz.')):
            return raw_code.lower()

        exchange_hint = None
        if upper.startswith(('SH', 'SS')) or upper.endswith(('.SH', '.SS')):
            exchange_hint = 'sh'
        elif upper.startswith('SZ') or upper.endswith('.SZ'):
            exchange_hint = 'sz'

        code = normalize_stock_code(raw_code)

        if exchange_hint in ('sh', 'sz') and code.isdigit() and len(code) == 6:
            return f"{exchange_hint}.{code}"
        
        # ETF: Shanghai ETF (51xx, 52xx, 56xx, 58xx) -> sh; Shenzhen ETF (15xx, 16xx, 18xx) -> sz
        if len(code) == 6:
            if code.startswith(('51', '52', '56', '58')):
                return f"sh.{code}"
            if code.startswith(('15', '16', '18')):
                return f"sz.{code}"

        # 根据代码前缀判断市场
        if code.startswith(('600', '601', '603', '605', '688')):
            return f"sh.{code}"
        elif code.startswith(('000', '001', '002', '003', '300', '301')):
            return f"sz.{code}"
        else:
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市")
            return f"sz.{code}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Baostock 获取原始数据
        
        使用 query_history_k_data_plus() 获取日线数据
        
        流程：
        1. 检查是否为美股（不支持）
        2. 使用上下文管理器管理连接
        3. 转换股票代码格式
        4. 调用 API 查询数据
        5. 将结果转换为 DataFrame
        """
        # 美股不支持，抛出异常让 DataFetcherManager 切换到其他数据源
        if _is_us_code(stock_code):
            raise DataFetchError(f"BaostockFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        # 港股不支持，抛出异常让 DataFetcherManager 切换到其他数据源
        if _is_hk_market(stock_code):
            raise DataFetchError(f"BaostockFetcher 不支持港股 {stock_code}，请使用 AkshareFetcher")

        # 北交所不支持，抛出异常让 DataFetcherManager 切换到其他数据源
        if is_bse_code(stock_code):
            raise DataFetchError(
                f"BaostockFetcher 不支持北交所 {stock_code}，将自动切换其他数据源"
            )
        
        # 转换代码格式
        bs_code = self._convert_stock_code(stock_code)
        
        logger.debug(f"调用 Baostock query_history_k_data_plus({bs_code}, {start_date}, {end_date})")
        
        with self._baostock_session() as bs:
            try:
                # 查询日线数据
                # adjustflag: 1-后复权，2-前复权，3-不复权
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    fields="date,open,high,low,close,volume,amount,pctChg",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",  # 日线
                    adjustflag="2"  # 前复权
                )
                
                if rs.error_code != '0':
                    raise DataFetchError(f"Baostock 查询失败: {rs.error_msg}")
                
                # 转换为 DataFrame
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    raise DataFetchError(f"Baostock 未查询到 {stock_code} 的数据")
                
                df = pd.DataFrame(data_list, columns=rs.fields)
                
                return df
                
            except Exception as e:
                if isinstance(e, DataFetchError):
                    raise
                raise DataFetchError(f"Baostock 获取数据失败: {e}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Baostock 数据
        
        Baostock 返回的列名：
        date, open, high, low, close, volume, amount, pctChg
        
        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()
        
        # 列名映射（只需要处理 pctChg）
        column_mapping = {
            'pctChg': 'pct_chg',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 数值类型转换（Baostock 返回的都是字符串）
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 添加股票代码列
        df['code'] = stock_code
        
        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """
        获取股票名称
        
        使用 Baostock 的 query_stock_basic 接口获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票名称，失败返回 None
        """
        # 检查缓存
        if hasattr(self, '_stock_name_cache') and stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]
        
        # 初始化缓存
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        
        try:
            bs_code = self._convert_stock_code(stock_code)
            
            with self._baostock_session() as bs:
                # 查询股票基本信息
                rs = bs.query_stock_basic(code=bs_code)
                
                if rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if data_list:
                        # Baostock 返回的字段：code, code_name, ipoDate, outDate, type, status
                        fields = rs.fields
                        name_idx = fields.index('code_name') if 'code_name' in fields else None
                        if name_idx is not None and len(data_list[0]) > name_idx:
                            name = data_list[0][name_idx]
                            self._stock_name_cache[stock_code] = name
                            logger.debug(f"Baostock 获取股票名称成功: {stock_code} -> {name}")
                            return name
                
        except Exception as e:
            logger.warning(f"Baostock 获取股票名称失败 {stock_code}: {e}")
        
        return None
    
    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """
        获取股票列表
        
        使用 Baostock 的 query_stock_basic 接口获取全部股票列表
        
        Returns:
            包含 code, name 列的 DataFrame，失败返回 None
        """
        try:
            with self._baostock_session() as bs:
                # 查询所有股票基本信息
                rs = bs.query_stock_basic()
                
                if rs.error_code == '0':
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if data_list:
                        df = pd.DataFrame(data_list, columns=rs.fields)
                        
                        # 转换代码格式（去除 sh. 或 sz. 前缀）
                        df['code'] = df['code'].apply(lambda x: x.split('.')[1] if '.' in x else x)
                        df = df.rename(columns={'code_name': 'name'})
                        
                        # 更新缓存
                        if not hasattr(self, '_stock_name_cache'):
                            self._stock_name_cache = {}
                        for _, row in df.iterrows():
                            self._stock_name_cache[row['code']] = row['name']
                        
                        logger.info(f"Baostock 获取股票列表成功: {len(df)} 条")
                        return df[['code', 'name']]
                
        except Exception as e:
            logger.warning(f"Baostock 获取股票列表失败: {e}")
        
        return None

    # ------------------------------------------------------------------
    # 筹码分布（a-stock-data §4.6 移植，Apache 2.0，原作者 simonlin1212）
    # ------------------------------------------------------------------
    # 定位：DataFetcherManager.get_chip_distribution 按 priority 遍历 fetcher，
    # akshare（priority 1，stock_cyq_em）失败/返回空时自动落到本方法（priority 3）。
    # 本方法是“兜底”而非替换：Variables 里 ENABLE_CHIP_DISTRIBUTION=true 时，
    # akshare 路径仍优先；只有它拿不到时才走 Baostock 本地计算。

    CHIP_HISTORY_CALENDAR_DAYS = 400  # 约 250 个交易日，见 PLAN.md §2.3
    CHIP_MIN_TRADING_DAYS = 60        # 有效交易日低于此数，播种占主导，结果不可信
    CHIP_GRID_SIZE = 300
    CHIP_DECAY = 1.0                  # 1.0=按真实换手率换手

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """Baostock 本地筹码分布（CYQ 推演）。

        取最近约 400 日历日（约 250 交易日）前复权 K 线 + 换手率，
        按换手率衰减逐日递推、当日成交按三角分布撒进 [low, high]，
        算出获利比例 / 平均成本 / 90% 与 70% 成本区间及集中度。

        Returns:
            ChipDistribution（source="baostock_local"），失败返回 None
            （manager 会继续尝试下一数据源；本 fetcher 已是链尾，
            返回 None 后 pipeline 记录“筹码分布获取失败或已禁用”并继续主流程）。
        """
        try:
            code = normalize_stock_code(stock_code)
        except Exception as e:
            logger.warning(f"Baostock 筹码分布: 代码规范化失败 {stock_code}: {e}")
            return None
        # 仅 A 股：美股/港股/北交所不走本路径（与 _fetch_raw_data 的守卫一致）
        if _is_us_code(code) or _is_hk_market(code) or is_bse_code(code):
            return None
        try:
            bs_code = self._convert_stock_code(code)
            end = datetime.now()
            start = end - timedelta(days=self.CHIP_HISTORY_CALENDAR_DAYS)
            with self._baostock_session() as bs:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,turn,tradestatus",
                    start_date=start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"),
                    frequency="d",
                    adjustflag="2",  # 2=前复权，筹码成本必须用复权价
                )
                if rs.error_code != "0":
                    logger.warning(f"Baostock 筹码分布查询失败 {code}: {rs.error_msg}")
                    return None
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    logger.warning(f"Baostock 筹码分布无数据 {code}")
                    return None
                df = pd.DataFrame(rows, columns=rs.fields)
            for col in ("open", "high", "low", "close", "turn"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df["tradestatus"] == "1"]  # 停牌日不参与换手衰减
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) < self.CHIP_MIN_TRADING_DAYS:
                logger.warning(
                    f"Baostock 筹码分布有效交易日不足 {code}: {len(df)} < {self.CHIP_MIN_TRADING_DAYS}"
                )
                return None
            result = _chip_distribution_local(
                df, grid_size=self.CHIP_GRID_SIZE, decay=self.CHIP_DECAY
            )
        except Exception as e:
            logger.warning(f"Baostock 筹码分布失败 {code}: {e}")
            return None

        latest_date = str(df["date"].iloc[-1])
        logger.info(
            f"Baostock 筹码分布(本地推演) {code} {latest_date}: "
            f"获利比例={result['profit_ratio']:.1%}, 平均成本={result['avg_cost']:.2f}, "
            f"筹码峰={result['peak_price']:.2f}"
        )
        return ChipDistribution(
            code=code,
            date=latest_date,
            source="baostock_local",
            profit_ratio=result["profit_ratio"],
            avg_cost=result["avg_cost"],
            cost_90_low=result["cost_90"][0],
            cost_90_high=result["cost_90"][1],
            concentration_90=result["concentration_90"],
            cost_70_low=result["cost_70"][0],
            cost_70_high=result["cost_70"][1],
            concentration_70=result["concentration_70"],
        )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    fetcher = BaostockFetcher()
    
    try:
        # 测试历史数据
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"获取成功，共 {len(df)} 条数据")
        print(df.tail())
        
        # 测试股票名称
        name = fetcher.get_stock_name('600519')
        print(f"股票名称: {name}")
        
    except Exception as e:
        print(f"获取失败: {e}")
