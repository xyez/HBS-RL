#!/usr/local/bin/python3
import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

DIV_LINE_WIDTH = 50


def get_dirs(logdirs_pre, config_file='config.yaml', legends=None, select_all=None, select_any=None, exclude=None, filter='', count=False):
    # Expand dir
    logdirs_expand = set()
    for logdir in logdirs_pre:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs_expand.add(logdir)
        else:
            base_dir = osp.dirname(logdir)
            prefix = osp.basename(logdir)
            for name in os.listdir(base_dir):
                if prefix in name:
                    match_dir = osp.join(base_dir, name)
                    if osp.isdir(match_dir):
                        logdirs_expand.add(match_dir)
    logdirs_expand = list(logdirs_expand)
    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs_expand:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)
    # Make sure the legend is compatible with the logdirs
    if legends is None:
        legends = [None] * len(logdirs_expand)
    assert (len(legends) == len(logdirs_expand)), "Must give a legend title for each set of experiments."
    logdirs_post = []
    logdir_idx = 0
    for logdir, legend in zip(logdirs_expand, legends):
        for basedir, subdir, file_names in os.walk(logdir):
            if select_all is not None and not all(x in basedir for x in select_all):
                continue
            if select_any is not None and not any(x in basedir for x in select_any):
                continue
            if exclude is not None and any(x in basedir for x in exclude):
                continue
            if config_file not in file_names:
                continue
            with open(osp.join(basedir, config_file), 'r') as f:
                if config_file.endswith('yaml'):
                    exp_name = yaml.load(f, Loader=yaml.FullLoader)
                elif config_file.endswith('json'):
                    exp_name = json.load(f)
                else:
                    raise NotImplementedError()
                exp_name = exp_name.get('exp_name', 'exp')
                if filter not in exp_name:
                    continue
                condition = legend or exp_name
                if count:
                    condition = '{}-{}'.format(condition, logdir_idx)
                    logdir_idx += 1
            logdirs_post.append((basedir, condition))
    return logdirs_post


def get_data(logdirs, files,sep, xaxiss, yaxiss, merge=False, smooth=1):
    data = defaultdict(lambda: [])
    keys_printed=set()
    for logdir, condition in logdirs:
        for file in files:
            file_path = osp.join(logdir, file)
            if osp.exists(file_path):
                if sep=='':
                    if file.endswith('txt'):
                        exp_data = pd.read_csv(file_path, sep='\t')
                    else:
                        exp_data = pd.read_csv(file_path, sep='\s*,', engine='python')
                else:
                    exp_data = pd.read_csv(file_path, sep=sep, engine='python')
                keys=tuple((exp_data.columns))
                if keys not in keys_printed:
                    print('Keys: ', keys)
                    keys_printed.add(keys)

                for yaxis in yaxiss:
                    if merge:
                        xaxis=None
                        for x in xaxiss:
                            if x in exp_data:
                                exp_data=exp_data.rename(columns={x:xaxiss[0]})
                                xaxis=xaxiss[0]
                                break
                        if not xaxis:
                            xaxis='Index'
                            exp_data[xaxis]=range(len(exp_data))
                    else:
                        xaxis=xaxiss[0]
                    
                    # like Score/x*10
                    yaxis=yaxis.split('/')
                    if len(yaxis)==1:
                        yaxis=yaxis[0]
                        transform_y=None
                    elif len(yaxis)==2:
                        yaxis,transform_y=yaxis
                        transform_y=eval('lambda x: {}'.format(transform_y))
                    if yaxis in exp_data:
                        exp_data_i = exp_data[[xaxis, yaxis]].dropna()
                        if transform_y:
                            exp_data_i[yaxis]=exp_data_i[yaxis].map(transform_y)
                        if merge:
                            exp_data_i=exp_data_i.rename(columns={yaxis:yaxiss[0]})
                            yaxis=yaxiss[0]
                        if smooth > 1:
                            """
                            smooth data with moving window average.
                            that is,
                                smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
                            where the "smooth" param is width of that window (2k+1)
                            """
                            y = np.ones(smooth)
                            x = np.asarray(exp_data_i[yaxis])
                            z = np.ones(len(x))
                            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                            exp_data_i[yaxis] = smoothed_x
                        exp_data_i['Condition'] = condition
                        data[yaxis].append(exp_data_i)

    data = {k: pd.concat(v, ignore_index=True) for k, v in data.items()}
    return data


def plot_data(data, xaxis, xlabels, ylabels, clabels, clocations, title, file, ratio, legend_show,legend_locs, sort, xMinimum, xMaximum):
    title_font_dict = {'family': 'Times New Roman', 'size': 18}
    label_font_dict = {'family': 'Times New Roman', 'size': 14}
    tick_font_dict = {'family': 'Times New Roman', 'size': 12}
    ylabels += [''] * (len(data) - len(ylabels))
    clabels += [''] * (len(data) - len(clabels))
    clocations += [''] * (len(data) - len(clocations))
    legend_locs += [''] * (len(data) - len(legend_locs))
    for ylabel, clabel, clocation,legend_loc, (yaxis, data_i) in zip(ylabels, clabels, clocations,legend_locs, data.items()):
        plt.figure()

        sns.set(style='darkgrid', font_scale=1.0)
        if xaxis not in data_i:
            xaxis='Index'
        ax = sns.lineplot(x=xaxis, y=yaxis, data=data_i, ci='sd', hue='Condition')
        handles, labels = ax.get_legend_handles_labels()
        if labels[0] == 'Condition':
            handles = handles[1:]
            labels = labels[1:]
        if sort > 0:
            if sort == 1:
                reverse = False
            elif sort == 2:
                reverse = True
            else:
                raise NotImplementedError()
            labels_sort = sorted(labels, reverse=reverse)
            handles = [x for _, x in sorted(zip(labels, handles), reverse=reverse)]
            labels = labels_sort
        if len(legend_show) == len(labels):
            for i, j in zip(labels, legend_show):
                print('{} ---> {}'.format(i, j))
            labels = legend_show

        legend = ax.legend(handles=handles, labels=labels, title=clabel, loc=legend_loc or 'best', prop=label_font_dict)
        plt.setp(legend.get_title(), fontsize=label_font_dict['size'], fontfamily=label_font_dict['family'])
        legend.set_draggable(True)
        if clocation != '':
            legend._legend_box.align = clocation

        xscale = np.max(np.asarray(data_i[xaxis])) > 5e3
        if xscale:
            # Just some formatting niceness: x-axis scale in scientific notation if max x is large
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(tick_font_dict['size'])
            ax.xaxis.get_offset_text().set_fontfamily(tick_font_dict['family'])

        
        plt.tight_layout(pad=0.2)
        xlabel = xlabels or xaxis
        ylabel = ylabel or yaxis
        plt.xlabel(xlabel, fontdict=label_font_dict)
        plt.ylabel(ylabel, fontdict=label_font_dict)
        plt.xticks(fontproperties=tick_font_dict['family'], size=tick_font_dict['size'])
        plt.yticks(fontproperties=tick_font_dict['family'], size=tick_font_dict['size'])
        if title != '':
            plt.title(title, fontdict=title_font_dict)
        
        plt.xlim(left=xMinimum,right=xMaximum)
        ax.set(aspect=1.0 / ax.get_data_ratio() * ratio)
        plt.tight_layout()
        if file != '':
            plt.savefig(file, dpi=500, bbox_inches='tight')
        plt.show()


def main(args):
    logdirs = get_dirs(args.logdir, args.config_file, args.legend, args.select_all, args.select_any, args.exclude, args.filter, count=args.count)
    yaxiss = args.yaxis if isinstance(args.yaxis, list) else [args.yaxis]
    data = get_data(logdirs, args.log_file,args.sep, args.xaxis, yaxiss,args.merge, args.smooth)
    plot_data(data, args.xaxis[0], args.xlabel, args.ylabel, args.clabel, args.clocation, args.title, args.path, args.ratio, args.legend_show,args.legend_loc, args.sort, xMinimum=args.xMinimum, xMaximum=args.xMaximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--legend_show', '-L', type=str, default=[], nargs='*')
    parser.add_argument('--legend_loc', '-LL', type=str, default=[], nargs='*')
    parser.add_argument('--title', '-T', default='')
    parser.add_argument('--xlabel', '-xl', default='')
    parser.add_argument('--ylabel', '-yl', default=[''], nargs='*')
    parser.add_argument('--clabel', '-cl', default=[''], nargs='*')
    parser.add_argument('--clocation', '-cL', default=[''], nargs='*')
    parser.add_argument('--xaxis', '-x', default=['Step'],nargs='*')
    parser.add_argument('--yaxis', '-y', default='MeanEpRetReal', nargs='*')
    parser.add_argument('--merge', '-m', type=eval, default=False)
    parser.add_argument('--xMinimum', '-xm', type=eval, default=None)
    parser.add_argument('--xMaximum', '-xM', type=eval, default=None)
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=5)
    parser.add_argument('--filter', '-f', type=str, default='')
    parser.add_argument('--select_all', nargs='*')
    parser.add_argument('--select_any', '-i', nargs='*')
    parser.add_argument('--exclude', '-o', nargs='*')
    parser.add_argument('--sep',type=str,default='')
    parser.add_argument('--estimator', default='mean')
    parser.add_argument('--log_file', '-lf', type=str, default=['eval.csv'], nargs='*', choices=['train.csv','eval.csv','progress.txt','progress.csv'])
    parser.add_argument('--config_file', '-cf', type=int, default=0, choices=[0, 1])
    parser.add_argument('--path', '-p', type=str, default='')
    parser.add_argument('--ratio', '-r', type=float, default=1)
    parser.add_argument('--sort', '-S', type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()
    if args.config_file == 0:
        args.config_file = 'config.yaml'
    elif args.config_file == 1:
        args.config_file = 'config.json'
    else:
        raise NotImplementedError()
    main(args)
