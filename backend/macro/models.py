"""宏观数据模型与指标目录。"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import date, datetime
from typing import Dict


@dataclass(frozen=True)
class MacroSeries:
    series_id: str
    name_cn: str
    source: str          # NBS / PBOC
    frequency: str       # monthly / quarterly / daily / event
    unit: str
    category: str
    description: str = ""
    country: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MacroObservation:
    series_id: str
    period: date
    value: float
    unit: str = ""
    source: str = ""
    provider: str = ""
    source_url: str = ""
    raw_label: str = ""
    fetched_at: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["period"] = self.period.isoformat()
        return data


@dataclass
class RawTable:
    source: str
    title: str
    url: str
    table_index: int
    columns: list
    rows: list
    fetched_at: str = ""


# ---------------------------------------------------------------- 指标目录
SERIES: Dict[str, MacroSeries] = {
    # 国家统计局
    "nbs_cpi_yoy": MacroSeries("nbs_cpi_yoy", "CPI 同比", "NBS", "monthly", "%", "价格", "居民消费价格指数同比"),
    "nbs_cpi_mom": MacroSeries("nbs_cpi_mom", "CPI 环比", "NBS", "monthly", "%", "价格", "居民消费价格指数环比"),
    "nbs_ppi_yoy": MacroSeries("nbs_ppi_yoy", "PPI 同比", "NBS", "monthly", "%", "价格", "工业生产者出厂价格指数同比"),
    "nbs_pmi_manufacturing": MacroSeries("nbs_pmi_manufacturing", "制造业 PMI", "NBS", "monthly", "指数", "景气", "制造业采购经理指数"),
    "nbs_pmi_non_manufacturing": MacroSeries("nbs_pmi_non_manufacturing", "非制造业 PMI", "NBS", "monthly", "指数", "景气", "非制造业商务活动指数"),
    "nbs_gdp_yoy": MacroSeries("nbs_gdp_yoy", "GDP 同比", "NBS", "quarterly", "%", "增长", "国内生产总值同比"),
    "nbs_gdp_value": MacroSeries("nbs_gdp_value", "GDP 绝对值", "NBS", "quarterly", "亿元", "增长", "国内生产总值绝对值"),
    "nbs_industrial_production_yoy": MacroSeries("nbs_industrial_production_yoy", "工业增加值同比", "NBS", "monthly", "%", "增长", "规模以上工业增加值同比"),
    "nbs_retail_sales_yoy": MacroSeries("nbs_retail_sales_yoy", "社零同比", "NBS", "monthly", "%", "消费", "社会消费品零售总额同比"),
    "nbs_retail_sales_value": MacroSeries("nbs_retail_sales_value", "社零当月值", "NBS", "monthly", "亿元", "消费", "社会消费品零售总额当月值"),
    "nbs_fai_yoy": MacroSeries("nbs_fai_yoy", "固定资产投资累计同比", "NBS", "monthly", "%", "投资", "固定资产投资（不含农户）累计同比"),
    # 中国人民银行
    "pbc_m2_yoy": MacroSeries("pbc_m2_yoy", "M2 同比", "PBOC", "monthly", "%", "货币", "广义货币供应量同比"),
    "pbc_m2_value": MacroSeries("pbc_m2_value", "M2 余额", "PBOC", "monthly", "亿元", "货币", "广义货币供应量余额"),
    "pbc_m1_yoy": MacroSeries("pbc_m1_yoy", "M1 同比", "PBOC", "monthly", "%", "货币", "狭义货币供应量同比"),
    "pbc_m1_value": MacroSeries("pbc_m1_value", "M1 余额", "PBOC", "monthly", "亿元", "货币", "狭义货币供应量余额"),
    "pbc_m0_yoy": MacroSeries("pbc_m0_yoy", "M0 同比", "PBOC", "monthly", "%", "货币", "流通中现金同比"),
    "pbc_m0_value": MacroSeries("pbc_m0_value", "M0 余额", "PBOC", "monthly", "亿元", "货币", "流通中现金余额"),
    "pbc_shrzgm_increment": MacroSeries("pbc_shrzgm_increment", "社融增量", "PBOC", "monthly", "亿元", "社融", "社会融资规模增量"),
    "pbc_shrzgm_rmb_loan": MacroSeries("pbc_shrzgm_rmb_loan", "社融-人民币贷款", "PBOC", "monthly", "亿元", "社融", "社融中人民币贷款"),
    "pbc_shrzgm_government_bond": MacroSeries("pbc_shrzgm_government_bond", "社融-政府债券", "PBOC", "monthly", "亿元", "社融", "社融中政府债券"),
    "pbc_shrzgm_corporate_bond": MacroSeries("pbc_shrzgm_corporate_bond", "社融-企业债券", "PBOC", "monthly", "亿元", "社融", "社融中企业债券"),
    "pbc_new_rmb_loans": MacroSeries("pbc_new_rmb_loans", "新增人民币贷款", "PBOC", "monthly", "亿元", "信贷", "金融机构新增人民币贷款"),
    "pbc_lpr_1y": MacroSeries("pbc_lpr_1y", "LPR 1年", "PBOC", "daily", "%", "利率", "贷款市场报价利率1年期"),
    "pbc_lpr_5y": MacroSeries("pbc_lpr_5y", "LPR 5年", "PBOC", "daily", "%", "利率", "贷款市场报价利率5年期"),
    "pbc_rrr_large": MacroSeries("pbc_rrr_large", "大型机构存准率", "PBOC", "event", "%", "货币", "大型金融机构存款准备金率"),
    "pbc_rrr_small": MacroSeries("pbc_rrr_small", "中小机构存准率", "PBOC", "event", "%", "货币", "中小金融机构存款准备金率"),

    # ---------------- 中国利率曲线 ----------------
    "cn_gov_2y": MacroSeries("cn_gov_2y", "中国国债 2Y", "CN", "daily", "%", "利率", "中国 2 年期国债收益率", "CN"),
    "cn_gov_5y": MacroSeries("cn_gov_5y", "中国国债 5Y", "CN", "daily", "%", "利率", "中国 5 年期国债收益率", "CN"),
    "cn_gov_10y": MacroSeries("cn_gov_10y", "中国国债 10Y", "CN", "daily", "%", "利率", "中国 10 年期国债收益率", "CN"),
    "cn_gov_30y": MacroSeries("cn_gov_30y", "中国国债 30Y", "CN", "daily", "%", "利率", "中国 30 年期国债收益率", "CN"),
    "cn_gov_10y_2y": MacroSeries("cn_gov_10y_2y", "中国 10Y-2Y 利差", "CN", "daily", "%", "利率", "中国期限利差", "CN"),
    # ---------------- 美国 ----------------
    "us_cpi_yoy": MacroSeries("us_cpi_yoy", "美国 CPI 同比", "US", "monthly", "%", "通胀", "美国 CPI 同比", "US"),
    "us_core_cpi_mom": MacroSeries("us_core_cpi_mom", "美国核心 CPI 月率", "US", "monthly", "%", "通胀", "美国核心 CPI 月率", "US"),
    "us_core_pce_yoy": MacroSeries("us_core_pce_yoy", "美国核心 PCE 同比", "US", "monthly", "%", "通胀", "美国核心 PCE 物价指数年率", "US"),
    "us_unemployment": MacroSeries("us_unemployment", "美国失业率", "US", "monthly", "%", "就业", "美国失业率", "US"),
    "us_nonfarm_payrolls": MacroSeries("us_nonfarm_payrolls", "美国非农就业", "US", "monthly", "万人", "就业", "美国非农就业人数", "US"),
    "us_gdp_growth": MacroSeries("us_gdp_growth", "美国 GDP 增速", "US", "monthly", "%", "增长", "美国 GDP 环比折年率（报告值）", "US"),
    "us_ism_pmi": MacroSeries("us_ism_pmi", "美国 ISM 制造业 PMI", "US", "monthly", "指数", "景气", "美国 ISM 制造业 PMI", "US"),
    "us_retail_sales_mom": MacroSeries("us_retail_sales_mom", "美国零售销售月率", "US", "monthly", "%", "消费", "美国零售销售月率", "US"),
    "us_industrial_production_mom": MacroSeries("us_industrial_production_mom", "美国工业产出月率", "US", "monthly", "%", "增长", "美国工业产出月率", "US"),
    "us_consumer_sentiment": MacroSeries("us_consumer_sentiment", "美国消费者信心", "US", "monthly", "指数", "景气", "密歇根大学消费者信心指数", "US"),
    "us_trade_balance": MacroSeries("us_trade_balance", "美国贸易帐", "US", "monthly", "亿美元", "贸易", "美国贸易帐", "US"),
    "us_fed_funds": MacroSeries("us_fed_funds", "美国联邦基金利率", "US", "event", "%", "利率", "美联储利率决议", "US"),
    "us_gov_2y": MacroSeries("us_gov_2y", "美国国债 2Y", "US", "daily", "%", "利率", "美国 2 年期国债收益率", "US"),
    "us_gov_10y": MacroSeries("us_gov_10y", "美国国债 10Y", "US", "daily", "%", "利率", "美国 10 年期国债收益率", "US"),
    "us_gov_30y": MacroSeries("us_gov_30y", "美国国债 30Y", "US", "daily", "%", "利率", "美国 30 年期国债收益率", "US"),
    "us_gov_10y_2y": MacroSeries("us_gov_10y_2y", "美国 10Y-2Y 利差", "US", "daily", "%", "利率", "美国期限利差", "US"),
    # ---------------- 日本 ----------------
    "jp_cpi_yoy": MacroSeries("jp_cpi_yoy", "日本 CPI 同比", "JP", "monthly", "%", "通胀", "日本 CPI 同比", "JP"),
    "jp_core_cpi_yoy": MacroSeries("jp_core_cpi_yoy", "日本核心 CPI 同比", "JP", "monthly", "%", "通胀", "日本核心 CPI 同比", "JP"),
    "jp_unemployment": MacroSeries("jp_unemployment", "日本失业率", "JP", "monthly", "%", "就业", "日本失业率", "JP"),
    "jp_boj_rate": MacroSeries("jp_boj_rate", "日本央行政策利率", "JP", "monthly", "%", "利率", "日本央行政策利率", "JP"),
    "jp_leading_indicator": MacroSeries("jp_leading_indicator", "日本领先指标", "JP", "monthly", "指数", "景气", "日本领先指标", "JP"),
    "jp_gdp_annual": MacroSeries("jp_gdp_annual", "日本 GDP 增速(年)", "JP", "annual", "%", "增长", "世界银行实际 GDP 增速", "JP"),
    "jp_inflation_annual": MacroSeries("jp_inflation_annual", "日本通胀(年)", "JP", "annual", "%", "通胀", "世界银行 CPI 通胀", "JP"),
    "jp_unemployment_annual": MacroSeries("jp_unemployment_annual", "日本失业率(年)", "JP", "annual", "%", "就业", "世界银行 ILO 失业率", "JP"),
    "jp_manufacturing_growth": MacroSeries("jp_manufacturing_growth", "日本制造业增速(年)", "JP", "annual", "%", "增长", "世界银行制造业增加值增速", "JP"),
    "jp_current_account_gdp": MacroSeries("jp_current_account_gdp", "日本经常账户/GDP", "JP", "annual", "%", "贸易", "世界银行经常账户占 GDP", "JP"),
    # ---------------- 美国/中国年度补充 ----------------
    "us_gdp_annual": MacroSeries("us_gdp_annual", "美国 GDP 增速(年)", "US", "annual", "%", "增长", "世界银行实际 GDP 增速", "US"),
    "us_inflation_annual": MacroSeries("us_inflation_annual", "美国通胀(年)", "US", "annual", "%", "通胀", "世界银行 CPI 通胀", "US"),
    "us_unemployment_annual": MacroSeries("us_unemployment_annual", "美国失业率(年)", "US", "annual", "%", "就业", "世界银行 ILO 失业率", "US"),
    "us_manufacturing_growth": MacroSeries("us_manufacturing_growth", "美国制造业增速(年)", "US", "annual", "%", "增长", "世界银行制造业增加值增速", "US"),
    "us_current_account_gdp": MacroSeries("us_current_account_gdp", "美国经常账户/GDP", "US", "annual", "%", "贸易", "世界银行经常账户占 GDP", "US"),
    "cn_gdp_annual": MacroSeries("cn_gdp_annual", "中国 GDP 增速(年)", "CN", "annual", "%", "增长", "世界银行实际 GDP 增速", "CN"),
    "cn_inflation_annual": MacroSeries("cn_inflation_annual", "中国通胀(年)", "CN", "annual", "%", "通胀", "世界银行 CPI 通胀", "CN"),
    "cn_unemployment_annual": MacroSeries("cn_unemployment_annual", "中国失业率(年)", "CN", "annual", "%", "就业", "世界银行 ILO 失业率", "CN"),
    "cn_manufacturing_growth": MacroSeries("cn_manufacturing_growth", "中国制造业增速(年)", "CN", "annual", "%", "增长", "世界银行制造业增加值增速", "CN"),
    "cn_current_account_gdp": MacroSeries("cn_current_account_gdp", "中国经常账户/GDP", "CN", "annual", "%", "贸易", "世界银行经常账户占 GDP", "CN"),

    # ---------------- 中国扩展（akshare 当前口径） ----------------
    "cn_fai_month_yoy": MacroSeries("cn_fai_month_yoy", "固定资产投资当月同比", "CN", "monthly", "%", "投资", "固定资产投资当月同比", "CN"),
    "cn_fai_cum_value": MacroSeries("cn_fai_cum_value", "固定资产投资累计值", "CN", "monthly", "亿元", "投资", "固定资产投资自年初累计", "CN"),
    "cn_electricity_yoy": MacroSeries("cn_electricity_yoy", "全社会用电量同比", "CN", "monthly", "%", "增长", "全社会用电量同比", "CN"),
    "cn_new_house_price_yoy": MacroSeries("cn_new_house_price_yoy", "70城新房价格同比", "CN", "monthly", "%", "地产", "70城新建商品住宅价格指数同比中位数", "CN"),
    "cn_second_hand_price_yoy": MacroSeries("cn_second_hand_price_yoy", "70城二手房价同比", "CN", "monthly", "%", "地产", "70城二手住宅价格指数同比中位数", "CN"),
    "cn_cb_total_assets": MacroSeries("cn_cb_total_assets", "央行总资产", "CN", "monthly", "亿元", "货币", "人民银行资产负债表总资产", "CN"),
    "cn_cb_reserve_money": MacroSeries("cn_cb_reserve_money", "储备货币", "CN", "monthly", "亿元", "货币", "人民银行储备货币", "CN"),
    "cn_shibor_3m": MacroSeries("cn_shibor_3m", "SHIBOR 3M", "CN", "daily", "%", "利率", "3个月 SHIBOR", "CN"),
    "cn_shibor_1y": MacroSeries("cn_shibor_1y", "SHIBOR 1Y", "CN", "daily", "%", "利率", "1年 SHIBOR", "CN"),
    "cn_margin_balance": MacroSeries("cn_margin_balance", "两融余额", "CN", "daily", "亿元", "杠杆", "沪深融资融券余额合计", "CN"),
    "cn_a_pe_median": MacroSeries("cn_a_pe_median", "全A PE 中位数", "CN", "daily", "倍", "估值", "全A市盈率中位数（TTM）", "CN"),
    "cn_a_pe_10y_pct": MacroSeries("cn_a_pe_10y_pct", "全A PE 10年分位", "CN", "daily", "%", "估值", "全A PE 中位数近10年分位", "CN"),
    "cn_a_pb_median": MacroSeries("cn_a_pb_median", "全A PB 中位数", "CN", "daily", "倍", "估值", "全A市净率中位数", "CN"),
    "cn_a_pb_10y_pct": MacroSeries("cn_a_pb_10y_pct", "全A PB 10年分位", "CN", "daily", "%", "估值", "全A PB 中位数近10年分位", "CN"),
    "cn_hsgt_net_buy": MacroSeries("cn_hsgt_net_buy", "北向资金净买入", "CN", "daily", "亿元", "外资", "沪深港通当日成交净买额（披露口径有变化）", "CN"),
    "cn_m1_m2_scissors": MacroSeries("cn_m1_m2_scissors", "M1-M2 剪刀差", "CN", "monthly", "pp", "货币", "M1同比 - M2同比", "CN"),
    "cn_real_rate": MacroSeries("cn_real_rate", "实际利率", "CN", "monthly", "%", "利率", "LPR1Y - CPI同比", "CN"),
}


# 兼容旧条目：未显式标注国家的 NBS/PBOC 序列归为中国
for _sid, _series in list(SERIES.items()):
    if not _series.country:
        SERIES[_sid] = replace(_series, country="CN" if _series.source in {"NBS", "PBOC"} else _series.source)
