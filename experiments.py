from resources.exp_resources import *


def investigate(self, data):

    frame = pd.DataFrame(data,
                         columns=handler.neighborhood_se.index,
                         index=handler.neighborhood_se.index)
    ij_mat = []
    for i, n_i in enumerate(self.comps):
        i_times = frame.loc[list(n_i.nodes())]
        ij_list = []
        for j, n_j in enumerate(self.comps):
            if j == i:
                ij_list.append(np.nan)
            else:
                ij_list.append(i_times.loc[:, list(n_j.nodes())].mean().mean())
        ij_mat.append(ij_list)
        a = 1


handler = DataHandling(new=True)
handler.matrices()