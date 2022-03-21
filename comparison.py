from resources.exp_resources import *

from scipy.cluster import hierarchy


class Hierarchical:

    def __init__(self, vars, init_labels):
        print("Initializing " + self.__class__.__name__)
        self.vars = vars
        self.labeling = init_labels
        self.z = self.form_hierarchy()

    def form_hierarchy(self):
        new = max(self.labeling)
        z = []
        size = {}
        or_labeling = np.unique(self.labeling)
        cur_labeling = np.unique(self.labeling)
        while len(np.unique(self.labeling)) > 1:
            new += 1
            print('getting resemblance matrice for ' + str(len(np.unique(self.labeling))) + ' clusters')
            r = np.triu(self.upgma())
            r[np.tril_indices(r.shape[0], 0)] = np.nan
            merge_ind = np.unravel_index(np.nanargmin(r), r.shape)

            x = np.unique(self.labeling)[merge_ind[0]]
            y = np.unique(self.labeling)[merge_ind[1]]
            cur_labeling[np.where(cur_labeling == x)] = new
            cur_labeling[np.where(cur_labeling == y)] = new
            self.labeling[self.labeling == x] = new
            self.labeling[self.labeling == y] = new

            unique, counts = np.unique(cur_labeling, return_counts=True)
            counts = dict(zip(unique, counts))
            size[new] = counts[new]
            z.append([x, y, np.nanmin(r), size[new]])
        return z



    # unweighted pair-group method using arithmetic averages
    def upgma(self):
        # initialize resemblance matrix
        r_mat = np.empty(shape=(len(np.unique(self.labeling)), len(np.unique(self.labeling))))
        for i_ind, i in enumerate(np.unique(self.labeling)):
            for j_ind, j in enumerate(np.unique(self.labeling)):
                if i == j:
                    continue
                x = handler.model_[self.labeling == i][self.vars]
                y = handler.model_[self.labeling == j][self.vars]
                xy = pd.concat([x, y])

                # standardized Euclidean distance takes the standard deviation between the compartments into account
                e_mat = spatial.distance.squareform(spatial.distance.pdist(X=xy,
                                                                           metric='seuclidean',
                                                                           V=xy.std(axis=0).values))
                e_mat = pd.DataFrame(e_mat, columns=xy.index, index=xy.index)
                e_xy = e_mat.loc[x.index, y.index]
                r_mat[i_ind][j_ind] = np.nanmean(e_xy.values.flatten())
        return r_mat




handler = DataHandling(new=True)
handler.matrices()

#acc = ['otp_clust', 'bike_clust', 'otp_q', 'bike_q']

handler.edu_score()
model = list(census_variables)
model.append('edu_score')
model = model[3:]
# model.extend(acc)

handler.stat_prep(vars=model)


skat_labels = np.load(file=os.path.join(path_experiments, 'reg_result.npy'))
handler.model_.reset_index(inplace=True, drop=True)
handler.neighborhood_se.reset_index(inplace=True, drop=True)

nan_rows = pd.isnull(handler.model_).any(axis=1)

skat_labels = skat_labels[~nan_rows]
handler.model_ = handler.model_[~nan_rows]
handler.neighborhood_se = handler.neighborhood_se[~nan_rows]

real_labels = np.unique(skat_labels)

def relabeling():
    for mask_l, real_l in enumerate(real_labels):
        skat_labels[skat_labels == real_l] = mask_l



relabeling()
real_labels = dict(zip(np.unique(skat_labels), real_labels))


h = Hierarchical(vars=model, init_labels=skat_labels)

fig, ax = plt.subplots()

dn = hierarchy.dendrogram(h.z)

real_labels = [real_labels[each] for each in dn['leaves']]

ax.set_xticklabels(list(map(str, np.array(real_labels) + 1)))

a = 1




# trse_box(data=handler.neighborhood_se,
#          labels=skat_labels,
#          feature=census_variables[-1],
#          acc='otp_clust')

