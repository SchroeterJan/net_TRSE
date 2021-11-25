from experiments import *

plt.style.use('seaborn')  # pretty matplotlib plots
plt.rc('font', size=24)
sns.set_theme(style="ticks")

plotting = Plotting()


# def se_year_miss(data, height, labels):
#     fig, ax = plt.subplots()
#     bars = ax.bar(x=data, height=height, align='center', tick_label=labels)
#     ax.set_title('Missing Census Data Points per Year')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Total Missing Data Points among Variables')
#     ax.bar_label(bars, label_type='center')
#     plt.savefig(fname=os.path.join(path_explore, 'missing_data'))
#     plt.close()


def hist_modes(travel_times):
    time_frame = pd.DataFrame(columns=travel_times)
    time_frame[travel_times[0]] = handler.bike.values.flatten()
    time_frame[travel_times[1]] = handler.pt.values.flatten()

    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)

    colors = ['red', 'blue']

    plotting.comp_hist(frame=time_frame, colors=colors)

    for i, mode in enumerate(travel_times):
        sns.histplot(data=time_frame, x=mode, binwidth=60, color=colors[mode], label=mode, alpha=0.5)
        plotting.meanline(data=time_frame, variable=mode, x=i+1)
    ax.set_title('Travel time histogram')
    ax.set_xlabel('Time')
    ax.margins(x=0)
    plt.tight_layout()
    plt.legend()

    plt.savefig(fname=os.path.join(path_hists, 'travel_times'))
    plt.close(f)


def hist_flows():
    f, ax = plt.subplots(figsize=(7, 5))
    sns.despine(f)
    data = pd.DataFrame(data=handler.flows.values.flatten(),
                        columns=['values'])
    sns.histplot(data, log_scale=True, legend=False)
    plotting.meanline(data=data, variable='values')
    ax.set_title('Passenger flow histogram')
    ax.margins(x=0)
    plt.xlabel('Passengers between two areas')
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path_hists, 'flow'))
    plt.close(f)


def hist_se():
    for i, variable in enumerate(census_variables):
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)

        handler.neighborhood_se[variable] = handler.neighborhood_se[variable].replace(to_replace=0.0, value=np.nan)
        sns.histplot(data=handler.neighborhood_se, x=variable)
        plotting.meanline(data=handler.neighborhood_se, variable=variable, x=i+1)
        ax.set_title('Histogram of ' + variable)
        ax.margins(x=0)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hists, variable + '_hist'))
        plt.close(f)


def hist_scaled_se():
    for variable in scaling_variables:
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)
        sns.histplot(data=handler.neighborhood_se, x=variable + '_scaled')
        ax.set_title('Histogram of ' + variable + '_scaled')
        ax.margins(x=0)
        plt.tight_layout()
        plt.savefig(fname=os.path.join(path_hists, variable + '_scaled_hist'))
        plt.close(f)


