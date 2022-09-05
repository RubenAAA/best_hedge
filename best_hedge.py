import pandas as pd
import riskfolio as rp
from matplotlib import pyplot as plt
import warnings

"""
Please refer to the attached historical data and propose a portfolio of 1 to 5
of the Eurostoxx 50 constituents which would best hedge a position of
long 1,000 shares of Pernod Ricard.

Please include the following:

1) how you are defining what is the best hedge

2) an explanation of your approach compared with alternative methods

3) your selected hedges along with the number of shares of each
"""

# The riskfolio library returns a lot of useless warnings, so I hide them
warnings.filterwarnings("ignore")

# Importing the excel file as pandas dataframe
sdata = pd.read_excel("source_data.xlsx", decimal='.')
# Cleaning the dataframe and transforming price data into returns
sdata = sdata.drop(sdata.columns[[0, 2]], axis=1).sort_index(axis=1)\
    .apply(pd.to_numeric).pct_change().dropna()


# Analyze the equities
def analyze_equities(sdata):
    # plot dendrogram
    ax = rp.plot_dendrogram(returns=sdata,
                            codependence='spearman',  # 'pearson', 'spearman'
                            linkage='ward',
                            k=None,
                            max_k=5,
                            leaf_order=True,
                            ax=None)
    plt.show()  # Unnecessary in Jupyter Notebook

    # plot clusters and dendogram
    ax = rp.plot_clusters(returns=sdata, codependence='spearman',
                          linkage='ward', k=None, max_k=5, dendrogram=False,
                          leaf_order=True, ax=None)
    plt.show()  # Unnecessary in Jupyter Notebook


# Helper function to calculate the correlation between the stocks
def return_correlations(sdata):
    corrtable1 = sdata.corr(method='pearson')
    corrpearson = corrtable1["PERNOD RICARD SA"].sort_values(ascending=False)
    corrtable2 = sdata.corr(method='spearman')
    corrpspearman = corrtable2["PERNOD RICARD SA"].sort_values(ascending=False)
    corrtable3 = sdata.corr(method='kendall')
    corrkendall = corrtable3["PERNOD RICARD SA"].sort_values(ascending=False)
    corrall = pd.concat((corrpearson, corrpspearman, corrkendall), axis=1,
                        keys=['pearson', 'spearman', 'kendall'])
    return corrall


# First we find the stocks that are negatively correlated with Pernod Ricard
# to construct long only portfolio
def data_to_consider_if_long_only():
    df = return_correlations(sdata)
    a = list(df.index[df['pearson'] < 0])
    b = list(df.index[df['spearman'] < 0])
    c = list(df.index[df['kendall'] < 0])
    ndata = pd.DataFrame()
    ndata["PERNOD RICARD SA"] = sdata["PERNOD RICARD SA"]
    for x in set(a + b + c):
        ndata[x] = sdata[x]
    return ndata


# Creating the optimal Portfolio for long only
# and plotting efficient frontier and weights
def Create_portfolio(Y):
    port = rp.Portfolio(returns=Y)

    # To display dataframes values in percentage format
    pd.options.display.float_format = '{:.4%}'.format

    # Calculate the inputs that will be used by the optimization method when we
    # select the input model=’Classic’.
    port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)

    # Estimate the portfolio that maximizes the risk adjusted return ratio
    w1 = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0.00,
                           l=0, hist=True)

    ax = rp.plot_pie(w=w1, title='Portfolio', height=6, width=10,
                     cmap="tab20")
    plt.show()  # Unnecessary in Jupyter Notebook

    # Estimate points in the efficient frontier mean - standard deviation
    ws = port.efficient_frontier(model='Classic', rm='MV', points=50, rf=0.00,
                                 hist=True)

    ax = rp.plot_frontier(label='Max Risk Adjusted Return Portfolio',
                          w_frontier=ws, mu=port.mu, cov=port.cov,
                          returns=port.returns, rm='MV', rf=0.00, alpha=0.05,
                          cmap='viridis', w=w1, marker='*', s=16,
                          c='r', height=6, width=10, t_factor=252)
    plt.show()  # Unnecessary in Jupyter Notebook


# analyze_equities(sdata)
Create_portfolio(data_to_consider_if_long_only())
