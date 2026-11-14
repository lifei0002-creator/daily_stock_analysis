# -*- coding: utf-8 -*-
"""
东财个股新闻兜底(移植自 simonlin1212/a-stock-data SKILL.md 5.1 节 eastmoney_stock_news,
Apache 2.0,原作者 simonlin1212)。

用途:搜索服务不可用(未配任何搜索 key)或搜索零命中时,
用免费的东财 search-api-web JSONP 接口补最近的个股新闻,
拼成与 SearchService 情报报告(format_intel_report)同形的 news_context,
供 analyzer 使用。

注意:
- 东财该接口对部分 IP 间歇性只返回 passportWeb 而无文章列表(上游风控),
  此时安全返回 [],pipeline 会如实记录"执行了但零命中",不阻断主流程。
- 请求自带节流(EM_MIN_INTERVAL=1.0s 加随机抖动,复用 session),与 skill 的
  em_get 同口径;pipeline 每只股最多调一次。
- 仅 A 股非指数(market == "cn" and not is_index);美股/港股/指数不调用。
"""

import json
import logging
import random
import re
import time

import requests

logger = logging.getLogger(__name__)

# 与 a-stock-data SKILL.md 共用的 UA
_EM_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_EM_STOCK_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"
_EM_STOCK_NEWS_REFERER = "https://so.eastmoney.com/"

EM_MIN_INTERVAL = 1.0  # 两次东财请求最小间隔(秒)
_EM_SESSION = requests.Session()
_em_last_call = [0.0]


def _em_get(url, params=None, headers=None, timeout=15):
    """东财统一请求入口:自动节流、复用 session、默认 UA。"""
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return _EM_SESSION.get(url, params=params, headers=headers, timeout=timeout)
    finally:
        _em_last_call[0] = time.time()


def _strip_html(text):
    """去掉东财标题/正文里的 HTML 标签。"""
    return re.sub(r"<[^>]+>", "", text or "")


def fetch_eastmoney_stock_news(code, page_size=10, timeout=15):
    """东财个股新闻(JSONP 接口,skill 5.1 节逐字移植)。

    Args:
        code: 6 位 A 股代码,如 "600170"
        page_size: 取最近几条(默认 10;格式化时最多展示 5 条)
        timeout: 单次请求超时秒数

    Returns:
        [{title, content(去标签,截 200 字), time, source, url}];
        任何失败(请求异常、JSONP 解析失败、接口间歇性返回空)都返回 [],不抛错。
    """
    try:
        # 构造 JSONP 参数
        cb = "jQuery_news"
        inner_params = json.dumps({
            "uid": "",
            "keyword": code,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                      "pageIndex": 1, "pageSize": page_size, "preTag": "", "postTag": ""}},
        }, separators=(",", ":"))
        params = {"cb": cb, "param": inner_params}
        headers = {"User-Agent": _EM_UA, "Referer": _EM_STOCK_NEWS_REFERER}
        r = _em_get(_EM_STOCK_NEWS_URL, params=params, headers=headers, timeout=timeout)
        # 解析 JSONP
        text = r.text
        json_str = text[text.index("(") + 1:text.rindex(")")]
        d = json.loads(json_str)
        rows = []
        # 东财实际返回里 result.cmsArticleWebOld 直接就是文章列表
        articles = d.get("result", {}).get("cmsArticleWebOld", []) or []
        for a in articles:
            rows.append({
                "title": _strip_html(a.get("title", "")),
                "content": _strip_html(a.get("content", ""))[:200],
                "time": a.get("date", ""),
                "source": a.get("mediaName", ""),
                "url": a.get("url", ""),
            })
        return rows
    except Exception as e:
        # 间歇性返回空与请求异常都在此消化:返回 [] 即"零命中"
        logger.warning(f"东财个股新闻获取失败 {code}: {e}")
        return []


def format_stock_news_context(news, stock_name):
    """把东财个股新闻拼成与 format_intel_report 同形的 news_context;无新闻返回 ""。

    来源明确标注"东财个股新闻"而非搜索商,避免 LLM 把它当成多源搜索结果高估。
    """
    if not news:
        return ""
    lines = [f"{stock_name} 情报搜索结果", "", "最新消息 (来源: 东财个股新闻):"]
    for i, item in enumerate(news[:5], start=1):
        title = item.get("title") or ""
        meta = " | ".join(
            part for part in (item.get("time") or "", item.get("source") or "") if part
        )
        lines.append(f"  {i}. {title}" + (f" [{meta}]" if meta else ""))
        content = (item.get("content") or "").strip()
        if content:
            lines.append(f"     {content}")
    return "\n".join(lines)


def run_stock_news_fallback(code, stock_name, market, is_index):
    """东财个股新闻兜底(pipeline Step 4 调用)。

    Returns:
        (news_context, news_result_count):
        - 非 A 股/指数:(None, None),不调用,news_result_count 保持"未执行";
        - 拿到新闻:(格式化文本, 条数);
        - 执行了但 0 条:(None, 0),如实记"执行了但零命中"。
        任何异常内部消化,绝不抛给 pipeline。
    """
    if market != "cn" or is_index:
        return None, None
    news = fetch_eastmoney_stock_news(code)
    if not news:
        logger.info(f"东财个股新闻兜底 {code}: 未拿到文章(可能触发东财间歇风控)")
        return None, 0
    logger.info(f"东财个股新闻兜底 {code}: 拿到 {len(news)} 条")
    return format_stock_news_context(news, stock_name), len(news)


if __name__ == "__main__":
    # 手动测试代码(不联网的单元测试见 tests/test_asd_port_fallbacks.py)
    logging.basicConfig(level=logging.DEBUG)
    try:
        news = fetch_eastmoney_stock_news("600170")
        print(f"拿到 {len(news)} 条")
        print(format_stock_news_context(news, "上海建工"))
    except Exception as e:
        print(f"获取失败: {e}")
