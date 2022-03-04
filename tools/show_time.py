import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# res_idx=[8,16,32,64]
# res_idx = [8, 16, 32, 64, 128, 256]
res_idx = [16, 24, 32, 48, 64, 128]
res_i = np.arange(len(res_idx))
label_font_dict = {'family': 'Times New Roman', 'size': 14}
title_font_dict = {'family': 'Times New Roman', 'size': 18}

time0=[36.20, 36.30, 37.86, 37.96, 42.52, 44.65]

plt.figure()
sns.set(style='darkgrid',font_scale=1)
# plt.plot(res_i, time0, marker='o', markerfacecolor='none', linewidth=2)
# plt.bar(res_i,time0,width=0.6,color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'])
ax=sns.barplot(res_i,time0,saturation=0.9)


legend=ax.legend(loc='best',title='HBS-RL (LSH)',)
# handles, labels = ax.get_legend_handles_labels()
plt.setp(legend.get_title(), fontsize=label_font_dict['size'], fontfamily=label_font_dict['family'])
plt.xticks(res_i, res_idx)
plt.xlabel('number of bits', fontdict=label_font_dict)
plt.ylabel('Time (Minutes)', fontdict=label_font_dict)
plt.xticks(fontproperties='Times New Roman', size=12)
# plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=12)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim([33.,85.])
# plt.savefig('./nus_result_3.pdf', dpi=200, bbox_inches='tight')
# plt.ylim([1,35.])
# plt.ylim([8., 30.])
plt.title('Training Time on MNIST',fontdict=title_font_dict)
ax = plt.gca()
ax.set(aspect=1.0 / ax.get_data_ratio() * 1)
plt.tight_layout()
plt.savefig('./mnist_time.pdf', dpi=500, bbox_inches='tight')
plt.show()
