import pandas as pd
import scipy.stats
from typing import Tuple


class CorrNetworkGraph:
    def __init__(self, sheet1: pd.DataFrame, sheet2: pd.DataFrame, method: str = "pearson"):
        self.sheet1 = sheet1
        self.sheet2 = sheet2
        self.r, self.p = self.__two_mat(method)

    def __two_mat(self, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        res_r = pd.DataFrame(index=self.sheet1.columns, columns=self.sheet2.columns)
        res_p = pd.DataFrame(index=self.sheet1.columns, columns=self.sheet2.columns)

        for name1 in self.sheet1.columns:
            for name2 in self.sheet2.columns:
                row1 = self.sheet1[name1]
                row2 = self.sheet2[name2]

                if method == "pearson":
                    r, p = scipy.stats.pearsonr(row1, row2)
                elif method == "spearman":
                    r, p = scipy.stats.spearmanr(row1, row2)
                else:
                    raise ValueError(f"Unknown method: {method}")

                if abs(r) > 0.6 and p < 0.05:
                    res_r.loc[name1, name2] = r
                    res_p.loc[name1, name2] = p

        return res_r, res_p

    def edge(self) -> pd.DataFrame:
        """
        构建相关性网络图的边信息
        """
        data = []

        for source in self.r.index:
            for target in self.r.columns:
                r = self.r.loc[source, target]
                p = self.p.loc[source, target]
                if pd.notna(r):
                    data.append({
                        "Source": source,
                        "Target": target,
                        "Weight": abs(r),
                        "Relevance": r,
                        "Significance": p,
                        "Color": "red" if r >= 0 else "blue"
                    })

        edge_df = pd.DataFrame(data)
        return edge_df

    def node(self) -> pd.DataFrame:
        """
        构建相关性网络图的节点信息
        """
        edge_df = self.edge()
        nodes = set(edge_df["Source"]).union(edge_df["Target"])

        data = []
        for node in nodes:
            color = "node1" if node in self.sheet1.columns else "node2"
            data.append({
                "Id": node,
                "Label": node,
                "Color": color
            })

        return pd.DataFrame(data)
