import pandas as pd


def bac_fun_merge(bac_path, fun_path):
    bac = pd.read_csv(bac_path, index_col=0)
    fun = pd.read_csv(fun_path, index_col=0)
    bac.index = ['B_' + i for i in bac.index]
    fun.index = ['F_' + i for i in fun.index]
    bac_fun = pd.concat([bac, fun])
    return bac_fun


def get_bac_fun_tax(bac_path, fun_path, taxonomy_column_name, save_path):
    bac_fun = bac_fun_merge(bac_path, fun_path)
    bac_fun.loc[:, taxonomy_column_name].to_csv(save_path)


if __name__ == '__main__':
    get_bac_fun_tax('../data/Daqu bacteria asv.csv', '../data/Daqu fungi asv.csv', 'taxonomy', '../data/Tax.csv')
