from Classes import *
from config import *
from experiments import *


plt.style.use('seaborn')  # pretty matplotlib plots
plt.rc('font', size=24)
sns.set_theme(style="ticks")

handler = DataHandling()

path_plot = os.path.join(path_repo, 'plots')
path_hists = os.path.join(path_plot, 'hists')

travel_times = ['Bike', 'Public Transport']


def hist_modes():
    edges_flat = pd.DataFrame(columns=travel_times)
    edges_flat[travel_times[0]] = handler.bike.values.flatten()
    edges_flat[travel_times[1]] = handler.pt.values.flatten()

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    colors = {'Bike': 'red', 'Public Transport': 'blue'}

    for mode in travel_times:
        sns.histplot(data=edges_flat, x=mode, binwidth=60, color=colors[mode], label=mode, alpha=0.5)
    ax.set_title('Travel time histogram')
    ax.margins(x=0)
    plt.tight_layout()
    plt.legend()

    plt.savefig(fname=os.path.join(path_hists, 'travel_times'))
    plt.close(f)


def hist_flows():
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    sns.histplot(data=pd.DataFrame(data=handler.flows.values.flatten(), columns=['values']), log_scale=True)
    ax.set_title('Passenger flow histogram')
    ax.margins(x=0)
    plt.tight_layout()
    plt.xlabel('Travellers on between two areas')
    plt.savefig(fname=os.path.join(path_hists, 'flow'))
    plt.close(f)


def hist_se():
    for variable in census_variables:
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)

        sns.histplot(data=handler.neighborhood_se, x=variable)
        ax.set_title('Histogram of ' + variable)
        ax.margins(x=0)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hists, variable + '_hist'))
        plt.close(f)


def hist_cluster():
    cluster_list = {'pt_all': 'blue',

                    'pt_rel': 'red'
                    }

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    for cluster in cluster_list:
        sns.histplot(data=clusters, x=cluster, color=cluster_list[cluster], binwidth=0.025, label=cluster, alpha=0.6)
    ax.set_title('Clustering coefficient for Public Transport')
    plt.xlabel('Clustering coefficient')
    ax.margins(x=0)
    plt.tight_layout()
    plt.legend()
    plt.savefig(fname=os.path.join(path_hists, 'cluster_hist'))
    plt.close(f)


# hist_modes()
# hist_flows()
# hist_se()

clusters = get_cluster()

hist_cluster()
