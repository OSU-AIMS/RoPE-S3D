from robotpose import DatasetRenderer, Dataset
import cv2
import numpy as np


dataset = 'set20'

# alpha = .5

# cont = True

# rend = DatasetRenderer(dataset)
# ds = Dataset(dataset)
# imgs = np.copy(ds.og_img)

# bad = set()

# idx = 0

# while cont:
#     rend.setPosesFromDS(idx)
#     if idx in bad:
#         gamma = -50
#     else:
#         gamma = 0
#     color, depth = rend.render()
#     img = cv2.addWeighted(imgs[idx], alpha, color, 1-alpha, gamma)
#     cv2.imshow("",img)
#     ret = cv2.waitKey(0)
#     if ret == ord('w'):
#         bad.add(idx)
#     elif ret == ord('s'):
#         if idx in bad:
#             bad.remove(idx)
#     elif ret == ord('d'):
#         idx += 1
#         if idx >= ds.length:
#             idx = ds.length - 1
#     elif ret == ord('a'):
#         idx -= 1
#         if idx < 0:
#             idx = 0
#     elif ret == ord('0'):
#         cont = False
    


# bad = list(bad)
# bad.sort()
# print(bad)

ds = Dataset(dataset)
imgs = np.copy(ds.og_img).astype(int)

bad = []

for idx in range(1,ds.length):
    if np.sum(np.abs(imgs[idx-1] - imgs[idx])) < 1500000:
        bad.append(idx)

print(bad)
print(len(bad))




