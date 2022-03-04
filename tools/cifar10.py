import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_dict(timestep=''):
    if timestep=='20220126':
        data_dict = {
            'HBS-RL (All)_0': [27.09, 28.13, 29.28, 30.17, 30.95, 31.15],
            'HBS-RL (All)_1': [26.80, 27.91, 29.21, 30.47, 30.94, 31.47],
            'HBS-RL (All)_2': [26.18, 28.30, 29.06, 30.65, 30.61, 31.37],
            'HBS-RL (All)_3': [26.84, 28.25, 29.09, 30.21, 30.65, 30.99],
            'HBS-RL (All)_4': [26.84, 28.12, 28.80, 30.28, 30.58, 31.18],

            'HBS-RL (SpH)_0': [25.10,25.73,25.98,26.32,26.31,26.22],
            'HBS-RL (SpH)_1': [25.17,25.81,26.21,26.39,26.56,26.55],
            'HBS-RL (SpH)_2': [25.10,25.81,25.89,26.60,26.15,26.62],
            'HBS-RL (SpH)_3': [25.17,25.81,26.01,26.19,26.52,26.46],
            'HBS-RL (SpH)_4': [25.10,25.62,26.05,25.99,26.22,26.30],

            'SpH': [18.06, 18.41, 19.19, 20.11, 20.51, 22.98],

            'NDomSet (All)_0': [16.87, 18.53, 20.94, 22.06, 23.14, 23.66],
            'NDomSet (All)_1': [17.28, 18.41, 20.59, 22.17, 22.64, 23.51],
            'NDomSet (All)_2': [17.00, 18.98, 21.24, 21.92, 22.88, 23.78],
            'NDomSet (All)_3': [17.46, 18.96, 20.96, 21.98, 22.73, 23.78],
            'NDomSet (All)_4': [16.93, 18.38, 21.32, 22.05, 22.43, 23.66],

            'HBS-RL (ITQ)_1': [25.51,28.15,28.89,29.16,30.06,30.42],
            'HBS-RL (ITQ)_0': [26.18,28.00,28.96,29.57,30.03,30.41],
            'HBS-RL (ITQ)_2': [25.92,27.59,28.71,29.22,30.27,30.51],
            'HBS-RL (ITQ)_3': [25.91,27.53,28.70,29.73,30.56,30.59],
            'HBS-RL (ITQ)_4': [26.09,27.85,28.97,29.58,30.32,30.55],

            'ITQ': [21.52, 21.57, 22.21, 23.12, 24.43, 25.23],

            'Random (All)_0': [15.50,17.45,18.14,20.30,21.31,22.98],
            'Random (All)_1': [17.26,17.38,18.88,20.84,21.28,22.41],
            'Random (All)_2': [13.99,15.06,16.08,17.62,19.20,22.34],
            'Random (All)_3': [15.86,16.65,19.19,20.71,21.52,23.10],
            'Random (All)_4': [15.19,16.02,16.30,19.05,20.37,23.47],

            'HBS-RL (SH)_0': [19.59, 19.73, 20.34, 19.73, 20.03, 19.68],
            'HBS-RL (SH)_1': [19.38, 19.74, 20.34, 20.32, 19.57, 19.62],
            'HBS-RL (SH)_2': [19.59, 19.79, 20.43, 19.74, 19.92, 19.58],
            'HBS-RL (SH)_3': [19.59, 19.79, 20.49, 20.17, 19.81, 19.55],
            'HBS-RL (SH)_4': [19.59, 19.79, 20.29, 20.22, 19.87, 19.67],

            'SH': [14.93, 14.77, 15.30, 15.00, 15.16, 14.89 ]
        }
    else:
        raise RuntimeError()
    return data_dict


def main():
    num_bits = [16, 24, 32, 48, 64, 128]

    x_values = range(len(num_bits))
    xaxis = 'Number of bits'
    yaxis = 'mAP (Test)'
    font_dict = {'family': 'Times New Roman', 'size': 12.5}
    legend_font_dict = {'family': 'Times New Roman', 'size': 9.1}

    data_dict=get_dict('20220126')

    cmap = sns.color_palette("deep")
    settings = {'HBS-RL (All)': [cmap[0], (), 'o'],
                'NDomSet (All)': [cmap[1], (), 'o'],
                'Random (All)': [cmap[2], (), 'o'],
                'HBS-RL (SpH)': [cmap[3], (3, 1, 1, 1), '^'],
                'HBS-RL (ITQ)': [cmap[4], (3, 1, 1, 1), '<'],
                'HBS-RL (SH)': [cmap[5], (3, 1, 1, 1), '>'],
                'SpH': [cmap[3], (1, 1), '^'],
                'ITQ': [cmap[4], (1, 1), '<'],
                'SH': [cmap[5], (1, 1), '>'],
                # 'NDomSet-sourcecode': [cmap[7], (), '*'],
                }
    data = []
    for key, value in data_dict.items():
        data_i = pd.DataFrame.from_dict({xaxis: x_values, yaxis: value})
        data_i['label'] = key.split('_')[0]
        data.append(data_i)
    data = pd.concat(data, ignore_index=True)

    palette={k: v[0] for k, v in settings.items()}
    data_keys = data['label'].unique()
    dashes=[settings[k][1] for k in data_keys]
    markers=[settings[k][2] for k in data_keys]

    plt.figure()
    sns.set(style='darkgrid', font_scale=1)
    ax = sns.lineplot(x=xaxis, y=yaxis, data=data, ci='sd', hue='label', linewidth=1.5, style='label',err_style='band',palette=palette,dashes=dashes, markers=markers)
    # ax = sns.lineplot(x=xaxis, y=yaxis, data=data, ci='sd', hue='label', linewidth=1.5, palette={k: v[0] for k, v in settings.items()})
    #
    # data_keys = data['label'].unique()
    # linestyles = [settings[k][1] for k in data_keys]
    # linemarkers = [settings[k][2] for k in data_keys]
    # print(linestyles)
    # print(linemarkers)
    # for i in range(len(ax.lines)):
    #     key = ax.lines[i].get_label()
    #     if '_line' in key:
    #         linestyle = linestyles[int(key[-1])]
    #         linemarker = linemarkers[int(key[-1])]
    #     elif key in settings.keys():
    #         linestyle = settings[key][1]
    #         linemarker = settings[key][2]
    #     else:
    #         continue
    #     print(i, key, linestyle, linemarker)
    #     ax.lines[i].set_linestyle(linestyle)
    #     ax.lines[i].set_marker(linemarker)
    #     # ax.lines[i].set_markersize(6)
    #
    # # dash_list = sns._core.unique_dashes(data['label'].unique().size + 1)

    handles, labels = ax.get_legend_handles_labels()
    if labels[0] == 'label':
        handles = handles[1:]
        labels = labels[1:]
    legend = ax.legend(handles=handles, labels=labels, title='', loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.21), prop=legend_font_dict)
    legend.set_draggable(True)

    plt.xticks(x_values, num_bits)
    plt.ylim([12.5, 32.5])
    plt.xticks(fontproperties='Times New Roman', size=10)
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xlabel(xaxis, fontdict=font_dict)
    plt.ylabel(yaxis, fontdict=font_dict)
    ax = plt.gca()
    ax.set(aspect=1.0 / ax.get_data_ratio() * 0.812)
    plt.tight_layout()
    plt.savefig('./cifar_result.pdf', dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
