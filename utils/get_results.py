import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as t


def compute_mAP(testbook, codebook, groundtruth=None, testlabel=None, codelabel=None, topK=None, num_interval=50, float_type=t.float32):
    if groundtruth is None:
        groundtruth = testlabel @ codelabel.t()
        groundtruth.clamp_max_(1)
    else:
        groundtruth = groundtruth
    groundtruth = groundtruth.type(float_type)
    testbook = testbook.type(float_type)
    codebook = codebook.type(float_type)
    if topK is None:
        topK = codebook.size(0)
    device = codebook.device
    num_test = testbook.size(0)
    train_range = t.arange(1, 1 + topK).to(device).type(float_type).unsqueeze(0)
    testbook2_sum = testbook.pow(2).sum(dim=1, keepdim=True)
    codebook2_sum = codebook.pow(2).sum(dim=1, keepdim=True)

    recall = t.zeros(topK, dtype=float_type, device=device)
    precision = t.zeros(topK, dtype=float_type, device=device)
    ap = t.zeros(num_test, dtype=float_type, device=device)
    for i in range(0, testbook.size(0), num_interval):
        distance = testbook2_sum[i:i + num_interval] + codebook2_sum.t() - 2 * testbook[i:i + num_interval] @ codebook.t()
        distance_idx = distance.argsort(dim=1)[:, :topK]
        groundtruth_sort = groundtruth[i:i + num_interval].gather(dim=1, index=distance_idx)  # TEST,topK
        groundtruth_sort_sum = groundtruth_sort.sum(dim=1, keepdim=True)  # TEST,1
        groundtruth_sort_cumsum = groundtruth_sort.cumsum(dim=1)  # TEST,topK
        ind_valid = (groundtruth_sort_sum > 0).type(float_type)  # TEST,1
        groundtruth_sort_sum.clamp_min_(1e-6)

        recall += (ind_valid * (groundtruth_sort_cumsum / groundtruth_sort_sum)).sum(0)
        precision_i = groundtruth_sort_cumsum / train_range
        precision += (ind_valid * precision_i).sum(0)
        ap[i:i + num_interval] = (ind_valid * ((groundtruth_sort * precision_i).sum(dim=1, keepdim=True) / groundtruth_sort_sum)).squeeze(1)
    recall /= num_test
    precision /= num_test
    m_ap = ap.mean()
    return recall.cpu().numpy(), precision.cpu().numpy(), m_ap.item()


def get_results(books, device, topK=0, num_interval=50, float_type=1):
    if topK == 0:
        topK = None
    if float_type == 1:
        float_type = t.float32
    elif float_type == 2:
        float_type = t.float64
    else:
        raise NotImplementedError()
    results = {}
    for book_name, book in books.items():
        codebook = book['rB'].to(device)
        testbook = book['qB'].to(device)
        groundtruth = book['qL'].to(device) @ book['rL'].to(device).t()
        groundtruth.clamp_max_(1)
        recall, precision, mAP = compute_mAP(testbook, codebook, groundtruth, topK=topK, num_interval=num_interval, float_type=float_type)
        results[book_name] = {'recall': recall, 'precision': precision, 'map': mAP}
    return results


def show_results(dataset, results, position=None):
    if position is None:
        num_points = list(results.values())[0]['recall'].size
        position = [1, 2, 3, 5, 10, 20, 50, 100] + np.arange(300, num_points, 200).tolist()
        if position[-1] != num_points - 1:
            position.append(num_points - 1)
        position = np.asarray(position, dtype=np.int64)

    # show_type False: seperate, True Together
    # show_type 0:none, 1:show, 2:show and save,
    show_type = [0, 0, 2, 0]
    font_dict = {'family': 'Times New Roman', 'size': 16}
    three_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][::-1]
    two_lines = ['-', '--', '-.']
    cmap = sns.color_palette("deep")
    # labels = ['HBS-RL (LSH)', 'NDomSet (LSH)', 'Random (LSH)']
    settings = {
        'HBS-RL (LSH)': [cmap[0], (), 'o'],
        'NDomSet (LSH)': [cmap[1], (3, 1, 1, 1), 'o'],
        'Random (LSH)': [cmap[2], (1, 1), 'o'],
    }

    # settings = {
    #     'HBS-RL (SpH)': [cmap[0], (), 'o'],
    #     'NDomSet (SpH)': [cmap[1], (3, 1, 1, 1), 'o'],
    #     'Random (SpH)': [cmap[2], (1, 1), 'o'],
    # }
    # labels = ['HBS-RL (SpH)', 'NDomSet (SpH)', 'Random (SpH)']
    labels = list(settings.keys())

    tick_font_dict = {'family': 'Times New Roman', 'size': 14}

    if show_type[0] > 0:
        plt.figure()
        sns.set(style='darkgrid', font_scale=1.5)
        plt.title(dataset.upper(), fontdict=font_dict)
        for i, method in enumerate(results.keys()):
            # print(i, len(results[method]['recall']))
            plt.plot(position, results[method]['recall'][position], color=three_colors[i % len(three_colors)], linestyle=two_lines[i % len(two_lines)], linewidth=2, label=labels[i])
        plt.legend(loc='lower right', prop=font_dict)
        plt.xlabel('number of retrieved samples', fontdict=font_dict)
        plt.ylabel('Recall @ 32 bits', fontdict=font_dict)
        # plt.xlim(left=0, right=50000)
        # plt.ylim(bottom=0, top=1.1)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        # plt.grid('on', linestyle='-.')
        ax = plt.gca()
        # xscale
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(tick_font_dict['size'])
        ax.xaxis.get_offset_text().set_fontfamily(tick_font_dict['family'])

        ax.set(aspect=1.0 / ax.get_data_ratio() * 0.8)
        if show_type[0] == 2:
            plt.savefig('recall32.pdf', dpi=500, bbox_inches='tight')
        plt.show()

    if show_type[1] > 0:
        plt.figure()
        sns.set(style='darkgrid', font_scale=1.5)
        plt.title(dataset.upper(), fontdict=font_dict)
        for i, method in enumerate(results.keys()):
            plt.plot(position, results[method]['precision'][position], color=three_colors[i % len(three_colors)], linestyle=two_lines[i % len(two_lines)], linewidth=2, label=labels[i])
        plt.legend(loc='upper right', prop=font_dict)
        plt.xlabel('number of retrieved samples', fontdict=font_dict)
        # plt.xlim(left=0, right=50000)
        # plt.ylim(bottom=0.1, top=0.8)
        plt.ylabel('Precision @ 32 bits', fontdict=font_dict)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        # plt.grid('on', linestyle='-.')
        ax = plt.gca()
        # xscale
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(tick_font_dict['size'])
        ax.xaxis.get_offset_text().set_fontfamily(tick_font_dict['family'])

        ax.set(aspect=1.0 / ax.get_data_ratio() * 0.8)
        if show_type[1] == 2:
            plt.savefig('precision32.pdf', dpi=500, bbox_inches='tight')
        plt.show()

    if show_type[2] > 0:
        plt.figure()
        sns.set(style='darkgrid', font_scale=1)
        plt.title(dataset.upper() + ' @ 32 bits', fontdict=font_dict)
        data = []
        xaxis = 'Recall'
        yaxis = 'Precision'

        for i, method in enumerate(results.keys()):
            data_i = {xaxis: results[method]['recall'][position], yaxis: results[method]['precision'][position]}
            data_i = pd.DataFrame.from_dict(data_i)
            data_i['label'] = labels[i]
            data.append(data_i)
        data = pd.concat(data, ignore_index=True)

        palette = {k: v[0] for k, v in settings.items()}
        data_keys = data['label'].unique()
        dashes = [settings[k][1] for k in data_keys]

        ax = sns.lineplot(x=xaxis, y=yaxis, data=data, hue='label', linewidth=1.5, style='label', dashes=dashes, palette=palette)

        font_dict = {'family': 'Times New Roman', 'size': 16}
        legend_font_dict = {'family': 'Times New Roman', 'size': 14}

        handles, labels = ax.get_legend_handles_labels()
        if labels[0] == 'label':
            handles = handles[1:]
            labels = labels[1:]
        legend = ax.legend(handles=handles, labels=labels, title='', loc='upper right', prop=legend_font_dict)
        legend.set_draggable(True)

        plt.xlabel('Recall', fontdict=font_dict)
        plt.ylabel('Precision', fontdict=font_dict)
        # plt.xlim(left=0, right=1)
        # plt.ylim(bottom=0.1, top=0.8)
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        # plt.grid('on', linestyle='-.')
        ax = plt.gca()
        # xscale
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(tick_font_dict['size'])
        ax.xaxis.get_offset_text().set_fontfamily(tick_font_dict['family'])

        ax.set(aspect=1.0 / ax.get_data_ratio() * 1)
        if show_type[2] == 2:
            plt.savefig('pr32.pdf', dpi=500, bbox_inches='tight')
        plt.show()

    if show_type[3] > 0:
        plt.figure()
        sns.set(style='darkgrid', font_scale=1.5)
        plt.title(dataset.upper(), fontdict=font_dict)
        # plt.subplot(1, 4, 4)
        mAPs = {}
        for method, result in results.items():
            mAPs[method] = result['map']

        width = 0.7
        x_ticks = np.arange(len(mAPs))

        for i, method in enumerate(mAPs.keys()):
            plt.bar(x_ticks[i], mAPs[method], width=width * 0.5, label=method)
            plt.text(x_ticks[i], mAPs[method], '{:.4f}'.format(mAPs[method]), size=18, ha='center', va='bottom')

        plt.legend(loc='upper right', prop=font_dict)
        # plt.ylim([np.floor((min_tick-0.15)*10)*0.1,np.ceil((max_tick+0.1)*10)*0.1])
        plt.xticks(x_ticks, mAPs.keys(), fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        plt.ylabel('mAP', fontdict=font_dict)
        ax = plt.gca()
        ax.set(aspect=1.0 / ax.get_data_ratio() * 0.8)
        if show_type[3] == 2:
            plt.savefig('map32.pdf', dpi=500, bbox_inches='tight')
        plt.show()
