import numpy as np
from matplotlib import pyplot as plt

# Infections
inf_1 = np.array([218224, 217549, 216289, 217225, 217108, 216173, 215862, 217061, 217278, 219286])
inf_2 = np.array([218364, 216313, 217014, 217300, 216922, 217711, 217599, 218447, 217410, 217036])
inf_3 = np.array([217885, 217123, 217342, 219763, 218896, 216874, 216567, 219270, 217346, 217074])
inf_4 = np.array([217669, 218124, 218957, 218129, 218449, 217583, 219055, 217224, 218271, 218006])
inf_5 = np.array([218380, 220443, 220073, 218085, 218738, 220047, 219722, 218287, 219005, 217957])
inf_6 = np.array([218108, 220280, 217465, 218343, 219313, 216861, 217741, 218287, 218197, 218272])
inf_7 = np.array([218391, 219034, 217146, 218916, 217959, 217509, 218533, 218528, 217338, 216900])
inf_8 = np.array([218309, 217116, 218146, 218969, 218151, 218106, 218415, 218153, 217646, 217261])
inf_9 = np.array([218057, 219713, 217716, 218400, 218370, 218359, 218500, 218221, 218132, 218554])
inf_10 = np.array([218390, 217539, 218677, 217734, 217732, 220272, 218036, 217983, 217461, 218602])

inf_15 = np.array([218292, 217855, 218233, 217821, 218374, 217613, 218367, 218030, 217658, 218673])
inf_20 = np.array([218713, 218483, 218226, 218344, 217803, 219210, 218754, 218523, 219121, 218016])
inf_25 = np.array([218168, 217664, 218944, 218487, 217904, 218318, 218142, 218316, 217806, 217493])
inf_50 = np.array([217832, 218397, 218150, 217998, 218209, 218126, 218580, 218913, 218238, 218295])

fig1, ax1 = plt.subplots()
ax1.set_title('Distribution of mean infections for 10 random seeds')
ax1.boxplot([inf_1, inf_2, inf_3, inf_4, inf_5, inf_6, inf_7, inf_8, inf_9, inf_10,
             inf_15, inf_20, inf_25, inf_50
             ])
ax1.set_xticks(range(1, 11))
plt.xlabel('Number of runs')
plt.ylabel('Number of cumulative infections')
plt.tight_layout()
plt.show()

# Deaths
deaths_1 = np.array([271, 272, 274, 289, 317, 275, 293, 297, 292, 307])
deaths_2 = np.array([316, 286, 282, 294, 303, 278, 274, 310, 301, 298])
deaths_3 = np.array([297, 274, 316, 274, 285, 283, 292, 277, 283, 288])
deaths_4 = np.array([299, 288, 278, 302, 293, 284, 282, 302, 291, 281])
deaths_5 = np.array([291, 284, 300, 276, 287, 297, 292, 287, 290, 280])
deaths_6 = np.array([292, 289, 282, 272, 282, 288, 297, 288, 294, 288])
deaths_7 = np.array([292, 289, 284, 299, 282, 294, 280, 290, 293, 295])
deaths_8 = np.array([295, 297, 294, 285, 290, 291, 288, 286, 278, 281])
deaths_9 = np.array([286, 283, 291, 292, 290, 297, 294, 288, 281, 287])
deaths_10 = np.array([290, 280, 291, 289, 285, 292, 291, 288, 289, 293])

deaths_15 = np.array([290, 281, 288, 294, 287, 289, 289, 289, 291, 283])
deaths_20 = np.array([288, 290, 284, 290, 281, 290, 292, 285, 290, 288])
deaths_25 = np.array([292, 293, 294, 288, 285, 290, 287, 289, 290, 284])
deaths_50 = np.array([287, 289, 289, 287, 290, 290, 288, 286, 289, 288])

fig2, ax2 = plt.subplots()
ax2.set_title('Distribution of mean deaths for 10 random seeds')
ax2.boxplot([deaths_1, deaths_2, deaths_3, deaths_4, deaths_5, deaths_6, deaths_7, deaths_8, deaths_9, deaths_10,
             deaths_15, deaths_20, deaths_25, deaths_50
             ])
ax2.set_xticks(range(1, 11))
plt.xlabel('Number of runs')
plt.ylabel('Number of cumulative deaths')
plt.tight_layout()
plt.show()
