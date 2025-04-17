import pandas as pd


def get_top_percent_asv(asv_path, save_path, percent=0.3):
    df = pd.read_csv(asv_path, index_col=0).T
    bac = df.loc[df.index.str.contains('B_'), :].copy()
    fun = df.loc[df.index.str.contains('F_'), :].copy()

    bac['sum'] = bac.apply(lambda x: sum(x), axis=1)
    bac = bac.sort_values(by='sum', ascending=False)
    bac = bac.iloc[range(0, int(bac.shape[0] * percent)), :]
    bac = bac.drop(columns='sum')

    fun['sum'] = fun.apply(lambda x: sum(x), axis=1)
    fun = fun.sort_values(by='sum', ascending=False)
    fun = fun.iloc[range(0, int(fun.shape[0] * percent)), :]
    fun = fun.drop(columns='sum')
    pd.concat([bac, fun]).T.to_csv(save_path)


if __name__ == '__main__':
    get_top_percent_asv('../data/ASVs.csv', "../data/top_percent_asv.csv")
