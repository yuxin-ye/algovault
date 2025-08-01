import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 股票代码与名称映射
_stock_mapping = {}

def set_stock_mapping(mapping):
    """设置股票代码与名称映射"""
    global _stock_mapping
    _stock_mapping = mapping

def init_tushare(api_key):
    """初始化Tushare API连接"""
    try:
        pro = ts.pro_api(api_key)
        print("Tushare API连接成功")
        return pro
    except Exception as e:
        print(f"Tushare API连接失败: {e}")
        return None

# 数据获取与预处理
def get_stock_data(pro, symbols, start_date, end_date):
    """获取多只股票的历史数据"""
    tickers = list(symbols.keys())
    stock_data = {}
    
    for ticker in tickers:
        try:
            # 获取日线行情数据
            df = pro.daily(
                ts_code=f'{ticker}.SH', 
                start_date=start_date, 
                end_date=end_date,
                fields='ts_code,trade_date,close,vol,amount'
            )
            
            # 获取市值数据
            df_mkt = pro.daily_basic(
                ts_code=f'{ticker}.SH', 
                start_date=start_date, 
                end_date=end_date,
                fields='ts_code,trade_date,turnover_rate,pe_ttm,pb,circ_mv'
            )
            
            # 合并数据
            df = pd.merge(df, df_mkt, on=['ts_code', 'trade_date'])
            
            # 数据处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            # 计算涨跌幅
            df['chgPct'] = df['close'].pct_change() * 100
            
            # 重命名列名
            df.rename(columns={
                'close': 'closePrice',
                'vol': 'turnoverVol',
                'amount': 'turnoverAmount',
                'circ_mv': 'negMarketValue'
            }, inplace=True)
            
            stock_data[ticker] = df
            print(f"成功获取 {ticker} ({symbols[ticker]}) 数据")
            
        except Exception as e:
            print(f"获取 {ticker} 数据失败: {e}")
    
    return stock_data

# 获取HS300指数数据
def get_hs300_data(pro, start_date, end_date):
    """获取沪深300指数数据"""
    try:
        df = pro.index_daily(
            ts_code="000300.SH",
            start_date=start_date,
            end_date=end_date,
            fields="trade_date,close,pct_chg"
        )
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        print("成功获取HS300指数数据")
        return df
    except Exception as e:
        print(f"获取HS300数据失败: {e}")
        return None

# 计算茅指数收益率 (以股票的流通市值为权重捏合"茅指数")
def calculate_mao_index(stock_data):
    """计算茅指数每日收益率"""
    # 对齐所有股票的日期索引
    all_dates = pd.DatetimeIndex([])
    for df in stock_data.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()
    
    # 初始化结果容器
    mao_r = pd.Series(0.0, index=all_dates, name='mao_return')
    
    for date in all_dates:
        total_mkt = 0.0  # 当日总流通市值
        weighted_ret = 0.0  # 加权收益率
        
        for ticker, df in stock_data.items():
            if date in df.index:
                mkt_value = df.loc[date, 'negMarketValue']
                ret = df.loc[date, 'chgPct'] # 这里用涨跌幅表示收益率
                total_mkt += mkt_value
                weighted_ret += mkt_value * ret
        
        # 计算当日茅指数收益率
        if total_mkt > 0:
            mao_r.loc[date] = weighted_ret / total_mkt
    
    return mao_r

# 均值回归策略实现
def mean_reversion_strategy(stock_data, mao_returns, hs300_data=None):
    """实现均值回归策略并计算净值"""
    # 准备数据
    tickers = list(stock_data.keys())
    dates = mao_returns.index
    
    # 收盘价矩阵
    close = pd.DataFrame(index=dates)
    for ticker in tickers:
        close[ticker] = stock_data[ticker]['closePrice'].reindex(dates) # 对齐日期
    
    # 收益率矩阵
    returns = pd.DataFrame(index=dates)
    for ticker in tickers:
        returns[ticker] = stock_data[ticker]['chgPct'].reindex(dates)
    
    # 计算5日均值MA5和偏离度ratio
    ma5 = close.rolling(window=5).mean()
    ratio = (close - ma5) / ma5  # (收盘价-MA5)/MA5
    
    # 确定每日持仓
    positions = pd.DataFrame(0, index=dates, columns=tickers)
    first_date = dates[0]
    
    # 首个交易日等权持有所有股票
    positions.loc[first_date] = 1.0
    
    # 后续交易日：持有ratio为负或NaN的股票 (ratio<0时认为价格低于均值有反弹趋势，产生买入信号)
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        
        for ticker in tickers:
            r = ratio.loc[prev_date, ticker]
            if pd.isna(r) or r < 0:
                positions.loc[date, ticker] = 1.0
    
    # 计算策略每日收益率
    daily_count = positions.sum(axis=1).replace(0, 1)  # 避免除零
    strategy_ret = (positions * returns).sum(axis=1) / daily_count
    
    # 计算净值
    strategy_nav = (1 + strategy_ret/100).cumprod()
    mao_nav = (1 + mao_returns/100).cumprod()
    
    # 处理HS300数据
    if hs300_data is not None:
        hs300_data = hs300_data.reindex(dates)
        hs300_data.ffill(inplace=True)
        hs300_nav = (1 + hs300_data['pct_chg']/100).cumprod()
    else:
        hs300_nav = mao_nav.copy()
    
    return strategy_nav, mao_nav, hs300_nav, positions, returns

# 策略绩效评估
def evaluate_strategy(strategy_nav, mao_nav, hs300_nav):
    """评估策略绩效"""
    # 计算累计收益率
    strategy_total_return = (strategy_nav.iloc[-1] - 1) * 100
    mao_total_return = (mao_nav.iloc[-1] - 1) * 100
    hs300_total_return = (hs300_nav.iloc[-1] - 1) * 100
    
    # 计算策略年化收益率
    days = len(strategy_nav)
    strategy_annual_return = (1 + strategy_total_return/100) ** (252/days) - 1
    strategy_annual_return = strategy_annual_return * 100
    
    mao_annual_return = (1 + mao_total_return/100) ** (252/len(mao_nav)) - 1
    mao_annual_return = mao_annual_return * 100
    
    hs300_annual_return = (1 + hs300_total_return/100) ** (252/len(hs300_nav)) - 1
    hs300_annual_return = hs300_annual_return * 100

    # 计算最大回撤
    def max_drawdown(nav):
        roll_max = nav.cummax()
        drawdown = (nav / roll_max - 1) * 100
        return drawdown.min()
    
    strategy_dd = max_drawdown(strategy_nav)
    mao_dd = max_drawdown(mao_nav)
    hs300_dd = max_drawdown(hs300_nav)
    
    # 计算夏普比率
    risk_free_rate = 0.01  # 假设无风险利率为1%
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1 # 转化为日无风险利率
    
    def sharpe_ratio(returns):
        excess_return = returns/100 - daily_risk_free
        return np.sqrt(252) * excess_return.mean() / excess_return.std() # np.sqrt(252)为日夏普比率的年化处理
    
    strategy_sharpe = sharpe_ratio(strategy_nav.pct_change() * 100)
    mao_sharpe = sharpe_ratio(mao_nav.pct_change() * 100)
    hs300_sharpe = sharpe_ratio(hs300_nav.pct_change() * 100)
    
    # 计算卡玛比率
    def calmar_ratio(annual_return, max_drawdown):
        if abs(max_drawdown) > 0.0001:  # 避免除零错误
            return annual_return / abs(max_drawdown)
        else:
            return np.inf  # 如果最大回撤接近0，则卡玛比率趋于无穷大
    
    strategy_calmar = calmar_ratio(strategy_annual_return, strategy_dd)
    mao_calmar = calmar_ratio(mao_annual_return, mao_dd)
    hs300_calmar = calmar_ratio(hs300_annual_return, hs300_dd)

    # 打印结果
    print("==== 策略绩效评估 ====")
    print(f"累计收益率: 策略 {strategy_total_return:.2f}% | 茅指数 {mao_total_return:.2f}% | HS300 {hs300_total_return:.2f}%")
    print(f"年化收益率: 策略 {strategy_annual_return:.2f}% | 茅指数 {mao_annual_return:.2f}% | HS300 {hs300_annual_return:.2f}%")
    print(f"最大回撤: 策略 {strategy_dd:.2f}% | 茅指数 {mao_dd:.2f}% | HS300 {hs300_dd:.2f}%")
    print(f"夏普比率: 策略 {strategy_sharpe:.2f} | 茅指数 {mao_sharpe:.2f} | HS300 {hs300_sharpe:.2f}")
    print(f"卡玛比率: 策略 {strategy_calmar:.2f} | 茅指数 {mao_calmar:.2f} | HS300 {hs300_calmar:.2f}")

    return {
        'total_return': strategy_total_return,
        'annual_return': strategy_annual_return,
        'max_drawdown': strategy_dd,
        'sharpe_ratio': strategy_sharpe,
        'calmar_ratio': strategy_calmar
    }

# 可视化函数
def visualize_results(strategy_nav, mao_nav, hs300_nav, positions, stock_data, language='Chi'):
    """可视化策略结果"""
    # 1. 绘制净值曲线对比
    plt.figure(figsize=(14, 6))
    if language == 'Chi':
        plt.plot(strategy_nav.index, strategy_nav, label='均值回归策略', linewidth=2)
        plt.plot(mao_nav.index, mao_nav, label='茅指数', linewidth=2)
        plt.plot(hs300_nav.index, hs300_nav, label='沪深300', linewidth=2)
        plt.title('策略净值与基准指数对比', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值', fontsize=12)
    elif language == 'Eng':
        plt.plot(strategy_nav.index, strategy_nav, label='Strategy', linewidth=2)
        plt.plot(mao_nav.index, mao_nav, label='mao', linewidth=2)
        plt.plot(hs300_nav.index, hs300_nav, label='HS300', linewidth=2)
        plt.title('Net Value of the Strategy and Benchmark Indexes', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Nav', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2. 绘制股票持仓变化
    plt.figure(figsize=(14, 8))
    for i, ticker in enumerate(positions.columns):
        plt.subplot(5, 2, i+1)
        plt.plot(positions.index, positions[ticker], label=f'{ticker} ({_stock_mapping[ticker]})')
        plt.title(_stock_mapping[ticker], fontsize=10)
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    if language == 'Chi':
        plt.suptitle('各股票持仓状态变化', fontsize=14)
    elif language == 'Eng':
        plt.suptitle('Changes in Stocks\' Holding Position', fontsize=14)
    plt.tight_layout()  # 调整布局，避免标题重叠
    plt.show()
    
    # 3. 绘制股票收盘价与MA5
    target_tickers = ['600519', '600036', '600900', '600276']  # 选择部分股票展示
    plt.figure(figsize=(14, 8))
    
    for i, ticker in enumerate(target_tickers):
        df = stock_data[ticker]
        plt.subplot(2, 2, i+1)
        
        # 绘制收盘价和MA5
        if language == 'Chi':
            plt.plot(df.index, df['closePrice'], 'b-', label='收盘价')
        elif language == 'Eng':
            plt.plot(df.index, df['closePrice'], 'b-', label='close')
        plt.plot(df.index, df['closePrice'].rolling(window=5).mean(), 'r--', label='MA5')
        
        # 标记持仓状态
        pos = positions[ticker].reindex(df.index)
        pos = pos.fillna(0)
        
        # 持仓期间背景高亮
        start = None
        for j in range(1, len(pos)):
            if pos.iloc[j] == 1 and pos.iloc[j-1] == 0:
                start = pos.index[j]  # 记录持仓开始
            if pos.iloc[j] == 0 and pos.iloc[j-1] == 1 and start is not None:
                end = pos.index[j]    # 记录持仓结束
                plt.axvspan(start, end, alpha=0.2, color='green')  # 绘制高亮区域
                start = None  # 重置开始标记

        # 处理最后一天仍持仓的情况
        if start is not None:
            end = pos.index[-1]  # 如果最后一天仍持仓，使用最后一个交易日作为结束
            plt.axvspan(start, end, alpha=0.2, color='green')
        
        plt.title(f'{_stock_mapping[ticker]}', fontsize=12)
        plt.legend(loc=2)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        if language == 'Chi':
            plt.xlabel('日期')
            plt.ylabel('价格')
        elif language == 'Eng':
            plt.xlabel('Date')
            plt.ylabel('Price')
    
    if language == 'Chi':
        plt.suptitle('部分股票的收盘价、MA5和持仓情况', fontsize=14)
    elif language == 'Eng':
        plt.suptitle('Closing Price, MA5 and Holding Position(partial)', fontsize=14)
    plt.tight_layout()
    plt.show()


# 达标率测算
def calculate_probability(data, TARGET, month):
    qualified_count = 0  # 达标次数
    valid_buy_dates = 0  # 有效买入日期数

    for buy_date in data.index:
        # 计算观察窗口的起始和结束日期
        obs_start = buy_date + pd.DateOffset(months=0) # 不设不止盈观察期
        obs_end = buy_date + pd.DateOffset(months=month)
    
        # 检查结束日期是否在数据范围内
        if obs_end > data.index[-1]:
            continue  # 跳过数据不足的日期
    
        # 获取观察窗口内的净值数据
        window_data = data.loc[obs_start:obs_end]
        if window_data.empty:
            continue  # 窗口无数据则跳过
    
        valid_buy_dates += 1
    
        # 获取买入净值
        nav_start = data.loc[buy_date, 'nav']
    
        # 遍历观察窗口的每个日期计算动态费率
        for sell_date in window_data.index:
            # 计算收益率
            return_rate = ((window_data.loc[sell_date, 'nav'])/ nav_start) - 1
            # 判断是否达标
            if return_rate >= TARGET:
                qualified_count += 1
                break  # 达标后不再检查后续日期
    return qualified_count / valid_buy_dates if valid_buy_dates > 0 else 0


#滚动持有测算
def rolling_win_rate(data, holding_period=252):
    """
    计算滚动持有胜率
    :param data: 包含日期和净值的DataFrame
    :param holding_period: 持有期交易日数（默认252天≈12个月）
    :return: 正收益概率
    """
    # 计算未来持有期结束时的净值
    data['future_nav'] = data['nav'].shift(-holding_period)
    
    # 计算持有期收益率
    data['return'] = (data['future_nav'] - data['nav']) / data['nav']
    
    # 删除无效数据
    valid_data = data.dropna(subset=['return'])
    
    # 计算正收益概率
    positive_count = (valid_data['return'] > 0).sum()
    total_count = len(valid_data)
    
    return positive_count / total_count
