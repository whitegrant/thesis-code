import numpy as np
import pandas as pd
from scipy.stats import lognorm
from itertools import chain, combinations



# tip distribution
tips = lognorm(s=.9, scale=2)

# read in data
BTC_ETH = np.loadtxt('data/BTC_ETH.csv')
USDC_BTC = np.loadtxt('data/USDC_BTC.csv')
USDC_ETH = np.loadtxt('data/USDC_ETH.csv')

# quantity distribution
b_e = lognorm(*lognorm.fit(BTC_ETH))
u_b = lognorm(*lognorm.fit(USDC_BTC))
u_e = lognorm(*lognorm.fit(USDC_ETH))

# constants
TOKENS = ['BTC', 'ETH', 'USDC']
QUANTITIES = [b_e, u_b, u_e]
PRICES = [38000, 2638, 1]
# START = [1, 14.4, 38000]
START = [5000/PRICES[0], 5000/PRICES[1], 5000]
pool_func = lambda a, b: 3*a + b - 1
POOLS = [2714.78, 1874.21, 50826.90, None, 52702.75, 22847370.15, 64330753.58]



# # get list of pending trades
# def get_trades(N):
#     # initialize
#     T = pd.DataFrame(columns=('quantity', 'sell_token', 'buy_token', 'tip', 'index'))
    
#     # add a trade one at a time
#     for i in range(N):
#         sell_idx, buy_idx = np.random.choice([0, 1, 2], 2, replace=False)
#         c_sell, c_buy = TOKENS[sell_idx], TOKENS[buy_idx]
#         qnty = QUANTITIES[sell_idx + buy_idx - 1].rvs() / PRICES[sell_idx]
#         tip = tips.rvs()
        
#         T.loc[i] = [qnty, c_sell, c_buy, tip, i]
        
#     return T

# get list of pending trades
def get_trades(N):
    # get quantities
    be_Q = b_e.rvs(N)/PRICES[0]
    eb_Q = b_e.rvs(N)/PRICES[1]
    ub_Q = u_b.rvs(N)
    bu_Q = u_b.rvs(N)/PRICES[0]
    ue_Q = u_e.rvs(N)
    eu_Q = u_e.rvs(N)/PRICES[1]
    Q = np.concatenate((be_Q, eb_Q, ub_Q, bu_Q, ue_Q, eu_Q))
    
    # get tips
    t = tips.rvs(6*N)
    
    # get token names
    b_temp = np.full(N, 'BTC')
    e_temp = np.full(N, 'ETH')
    u_temp = np.full(N, 'USDC')
    c_sell = np.concatenate((b_temp, e_temp, u_temp, b_temp, u_temp, e_temp))
    c_buy = np.concatenate((e_temp, b_temp, b_temp, u_temp, e_temp, u_temp))
    
    # array of trades
    df =  pd.DataFrame(
        np.transpose([Q, c_sell, c_buy, t]),
        columns=('quantity', 'sell_token', 'buy_token', 'tip')
    )
    df = df.astype({'quantity':float, 'tip':float})
    df['idx'] = df.index
    return df

# arbitrage profit
P = lambda a, N, A: (a*N*(a + 2*A + N)) / (a**2 + 2*a*A + a*N + A**2)

# get list of sandwich groups
def get_T_hat(T, weight):
    G_eb = T[(T['sell_token'] == 'ETH') & (T['buy_token'] == 'BTC')].values
    g_eb = list(chain.from_iterable(combinations(G_eb, r) for r in range(1, min(len(G_eb)+1, weight-2))))
    G_be = T[(T['sell_token'] == 'BTC') & (T['buy_token'] == 'ETH')].values
    g_be = list(chain.from_iterable(combinations(G_be, r) for r in range(1, min(len(G_be)+1, weight-2))))
    G_bu = T[(T['sell_token'] == 'BTC') & (T['buy_token'] == 'USDC')].values
    g_bu = list(chain.from_iterable(combinations(G_bu, r) for r in range(1, min(len(G_bu)+1, weight-2))))
    G_ub = T[(T['sell_token'] == 'USDC') & (T['buy_token'] == 'BTC')].values
    g_ub = list(chain.from_iterable(combinations(G_ub, r) for r in range(1, min(len(G_ub)+1, weight-2))))
    G_eu = T[(T['sell_token'] == 'ETH') & (T['buy_token'] == 'USDC')].values
    g_eu = list(chain.from_iterable(combinations(G_eu, r) for r in range(1, min(len(G_eu)+1, weight-2))))
    G_ue = T[(T['sell_token'] == 'USDC') & (T['buy_token'] == 'ETH')].values
    g_ue = list(chain.from_iterable(combinations(G_ue, r) for r in range(1, min(len(G_ue)+1, weight-2))))
    
    return g_eb + g_be + g_bu + g_ub + g_eu + g_ue

# calculate appt for each sandwich
def get_appt(T_hat):
    # initialize
    APPT = pd.DataFrame(columns=['sell_token', 'buy_token', 'profit', 'weight', 'appt'])
    trade_dict = {}

    # calculate powersets
    for i, g in enumerate(T_hat):
        # get starting investment and pool size
        c_sell, c_buy = g[0][1], g[0][2]
        sell_idx = TOKENS.index(c_sell)
        buy_idx = TOKENS.index(c_buy)
        a0 = START[sell_idx]
        A0 = POOLS[pool_func(sell_idx, buy_idx)]
        
        # add to dataframe
        N = sum([g_i[0] for g_i in g])
        profit = P(a0, N, A0)
        tip_sum = sum([g_i[3] for g_i in g])
        profit += tip_sum
        weight = len(g) + 2
        appt = profit / weight
        APPT.loc[len(APPT)] = [c_sell, c_buy, profit, weight, appt]
        
        # add to dictionary
        trade_dict[i] = {g_i[4] for g_i in g}

    return APPT, trade_dict

# adjust after adding a group
def adjust_group(k, T, APPT, trade_dict):
    # initialize
    new_dict = {}
    group = trade_dict[k]
    k_sell = APPT.loc[k, 'sell_token']
    k_buy = APPT.loc[k, 'buy_token']
    k_profit = APPT.loc[k, 'profit']
    k_weight = APPT.loc[k, 'weight']
    
    # add to block
    APPT.drop(k, inplace=True)
    trade_dict.pop(k)
    
    # adjust/remove groups
    for key, value in trade_dict.items():
        # adjust supersets
        if group.issubset(value):
            APPT.loc[key, 'profit'] = APPT.loc[key, 'profit'] - k_profit
            APPT.loc[key, 'weight'] = APPT.loc[key, 'weight'] - k_weight
            APPT.loc[key, 'appt'] = APPT.loc[key, 'profit'] / APPT.loc[key, 'weight']
            new_dict[key] = value
        else:
            # remove nonsupersets
            if (k_sell == APPT.loc[key, 'sell_token']) & (k_buy == APPT.loc[key, 'buy_token']):
                APPT.drop(key, inplace=True)
            else:
                new_dict[key] = value
            
    # remove single trades
    T = T[(T['sell_token'] != k_sell) | (T['buy_token'] != k_buy)]

    return new_dict

# adjust after making a single trade
def adjust_trade(k, T, APPT, trade_dict):
    # initialize
    new_dict = {}
    k_sell = T.loc[k, 'sell_token']
    k_buy = T.loc[k, 'buy_token']
    k_profit = T.loc[k, 'tip']
    k_weight = 1
    
    # add to block
    T.drop(k, inplace=True)
    
    # adjust/remove groups
    for key, value in trade_dict.items():
        # adjust supersets
        if k in value:
            APPT.loc[key, 'profit'] = APPT.loc[key, 'profit'] - k_profit
            APPT.loc[key, 'weight'] = APPT.loc[key, 'weight'] - k_weight
            APPT.loc[key, 'appt'] = APPT.loc[key, 'profit'] / APPT.loc[key, 'weight']
            new_dict[key] = value
        else:
            # remove nonsupersets
            if (APPT.loc[key, ['sell_token', 'buy_token']].values == [k_sell, k_buy]).all():
                APPT.drop(key, inplace=True)
            else:
                new_dict[key] = value

    return new_dict



# no arbitrage
def naive_method(T, weight=100):
    return T.nlargest(weight, 'tip')['tip'].sum()

# only the best sandwich
def single_method(T, APPT, trade_dict, weight=100):
    APPT2 = APPT[APPT['weight'] <= weight]
    sand_idx = APPT2[APPT2['profit'] == APPT2['profit'].max()].index[0]
    c_sell = APPT2.loc[sand_idx, 'sell_token']
    sell_idx = TOKENS.index(c_sell)
    profit = APPT2.loc[sand_idx, 'profit'] * PRICES[sell_idx]
    rem_weight = weight - APPT2.loc[sand_idx, 'weight']
    T2 = T.drop(trade_dict[sand_idx], inplace=False)
    return profit + T2.nlargest(rem_weight, 'tip')['tip'].sum()


# return mev
def mev_method(T, APPT, trade_dict, weight=100):
    # initialize
    rem_weight = weight
    profit = 0

    # repeat until the block is full
    while rem_weight > 0:
        APPT = APPT[APPT['weight'] <= rem_weight]
        if len(APPT.index) != len(trade_dict):
            trade_dict = {i:trade_dict[i] for i in APPT.index}

        # if no more sandwiches
        if len(APPT) == 0:
            profit += naive_method(T, rem_weight)
            break

        # best single trade
        single_idx = T[T['tip'] == T['tip'].max()].index[0]
        max_tip = T.loc[single_idx, 'tip']

        # best group
        sand_idx = APPT[APPT['profit'] == APPT['profit'].max()].index[0]
        appt = APPT.loc[sand_idx, 'appt']

        # add the better of the two
        if max_tip > appt:
            profit += max_tip
            rem_weight -= 1
            trade_dict = adjust_trade(single_idx, T, APPT, trade_dict)
        else:
            c_sell = APPT.loc[sand_idx, 'sell_token']
            sell_idx = TOKENS.index(c_sell)
            profit += (APPT.loc[sand_idx, 'profit'] * PRICES[sell_idx])
            rem_weight -= APPT.loc[sand_idx, 'weight']
            trade_dict = adjust_group(sand_idx, T, APPT, trade_dict)

    return profit



def run_test(N, weight=100):
    # get trades
    T = get_trades(N)

    # get groups
    T_hat = get_T_hat(T, weight)
    APPT, trade_dict = get_appt(T_hat)

    # get profit
    naive_profit = naive_method(T, weight)
    single_profit = single_method(T, APPT, trade_dict, weight)
    mev_profit = mev_method(T, APPT, trade_dict, weight)

    return naive_profit, single_profit, mev_profit

# def run_multi_test(N, weights):
#     # initialize
#     naive_profit = []
#     single_profit = []
#     mev_profit = []

#     # get trades
#     T = get_trades(N)

#     # get groups
#     T_hat = get_T_hat(T, N)
#     APPT, trade_dict = get_appt(T_hat)

#     for weight in weights:
#         # get profit
#         naive_profit.append(naive_method(T, weight))
#         single_profit.append(single_method(T, APPT, trade_dict, weight))
#         mev_profit.append(mev_method(T.copy(), APPT.copy(), trade_dict.copy(), weight))

#     return naive_profit, single_profit, mev_profit

# def run_both_tests(N, weight, weights):
#     # initialize
#     naive_profit = []
#     single_profit = []
#     mev_profit = []

#     # get trades
#     T = get_trades(N)

#     # get groups
#     T_hat = get_T_hat(T, N)
#     APPT, trade_dict = get_appt(T_hat)

#     # get profit
#     n_pro = naive_method(T, weight)
#     s_pro = single_method(T, APPT, trade_dict, weight)
#     m_pro = mev_method(T, APPT, trade_dict, weight)

#     for weight in weights:
#         # get profit
#         naive_profit.append(naive_method(T, weight))
#         single_profit.append(single_method(T, APPT, trade_dict, weight))
#         mev_profit.append(mev_method(T.copy(), APPT.copy(), trade_dict.copy(), weight))

#     return n_pro, s_pro, m_pro, naive_profit, single_profit, mev_profit



# num_tests = 50
# domain = np.arange(38, 51)
# num_N = len(domain)
# nL, sL, mL = np.empty((num_N, num_tests)), np.empty((num_N, num_tests)), np.empty((num_N, num_tests))

# for i, N in enumerate(domain):
#     print(N)
#     for j in range(num_tests):
#         print(j)
#         a, b, c = run_test(N, 20)
#         nL[i,j] = a
#         sL[i,j] = b
#         mL[i,j] = c
#     print()

# np.save('results/size_naive.npy', nL)
# np.save('results/size_single.npy', sL)
# np.save('results/size_mev.npy', mL)



num_tests = 50
domain = [13]
num_N = len(domain)
nL, sL, mL = np.empty((num_N, num_tests)), np.empty((num_N, num_tests)), np.empty((num_N, num_tests))

for i, N in enumerate(domain):
    print(N)
    for j in range(num_tests):
        print(j)
        a, b, c = run_test(N, 20)
        nL[i,j] = a
        sL[i,j] = b
        mL[i,j] = c
    print()

np.save('results/size_naive2.npy', nL)
np.save('results/size_single2.npy', sL)
np.save('results/size_mev2.npy', mL)



# num_tests = 50
# domain = np.arange(20, 51)
# num_N = len(domain)
# nL, sL, mL = np.empty((num_N, num_tests)), np.empty((num_N, num_tests)), np.empty((num_N, num_tests))

# for j in range(num_tests):
#     print(j)
#     a, b, c = run_multi_test(50, domain)
#     nL[:,j] = a
#     sL[:,j] = b
#     mL[:,j] = c

# np.save('results/pend_naive.npy', nL)
# np.save('results/pend_single.npy', sL)
# np.save('results/pend_mev.npy', mL)



# # num_tests = 50
# # domain = np.arange(20, 51)
# num_tests = 5
# domain = np.arange(20, 23)

# for N in domain[:-1]:
#     print(N)
#     nL_s, sL_s, mL_s = np.empty(num_tests), np.empty(num_tests), np.empty(num_tests)

#     for j in range(num_tests):
#         print(j)
#         a, b, c = run_test(N, 20)
#         nL_s[j] = a
#         sL_s[j] = b
#         mL_s[j] = c
#     print()

#     np.save(f'results/size_naive_{N}.npy', nL)
#     np.save(f'results/size_single_{N}.npy', sL)
#     np.save(f'results/size_mev_{N}.npy', mL)

# nL_s, sL_s, mL_s = np.empty(num_tests), np.empty(num_tests), np.empty(num_tests)
# nL_p, sL_p, mL_p = np.empty((num_N, num_tests)), np.empty((num_N, num_tests)), np.empty((num_N, num_tests))
# N = domain[-1]

# for j in range(num_tests):
#     print(j)
#     a, b, c, d, e, f = run_multi_test(50, N, domain)
#     nL_s[j] = a
#     sL_s[j] = b
#     mL_s[j] = c
#     nL_p[:,j] = d
#     sL_p[:,j] = e
#     mL_p[:,j] = f

# np.save(f'results/size_naive_{N}.npy', nL)
# np.save(f'results/size_single_{N}.npy', sL)
# np.save(f'results/size_mev_{N}.npy', mL)
# np.save(f'results/pend_naive_{N}.npy', nL)
# np.save(f'results/pend_single_{N}.npy', sL)
# np.save(f'results/pend_mev_{N}.npy', mL)
