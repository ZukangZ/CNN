'''
Cnn1-3-21.py: simple and slowest implementation for clearity
Cnn1-4-21.py: instead of looping through each filter, calculations for all filters are done at once with numpy
arr[np.newaxis]: adds [] around arr
arr[:, np.newaxis]: adds [] around each element
axis = 0 is .shape[0] and etc
np.repeat: duplicates right after the original in a 1, 1, 2, 2, 3, 3, ... pattern. np.tile([npArray], ([z], [y], [x])): duplicates at the end of the array in a 1, 2, 3, ..., 1, 2, 3, ..., 1, 2, 3 pattern.
hardcoded for pool stride of 2

first 1000 nums from mnist backprop
Cnn1-4-21.py: 54.967097997665405s
Cnn1-3-21.py: 71.33764815330505s
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

class Cnn:
  #xdim: input diminsions
  def __init__(self, xdim, amtFilters, filtersDim, amty, lr):
    #filters with larger dimensions need smaller filter numbers to prevent overflow 
    self.filters = np.random.randn(amtFilters, filtersDim, filtersDim) / filtersDim ** 2
    self.poolStride = 2
    self.hw = np.random.randn(((xdim - filtersDim + 1) // self.poolStride) ** 2 * amtFilters, amty) / 20
    self.hb = np.random.randn(amty)
    self.lr = lr

    self.convolveDim = xdim - filtersDim + 1
    #if the convolve dim is odd, the last row and col is discarded, since you can't use half a stride, only works for a pool stride of 2
    if self.convolveDim & 1:
      self.convolveDim -= 1
      #Not tested, for any pool stride
      # self.convolveDim -= self.convolveDim % self.poolStride
    self.poolDim = self.convolveDim // 2

  #valid padding
  def convolve(self, X):
    #z axis: input is conv with the filters to get n amount of h layers equal to the z axis of the filters, and this pattern repeats for n amount of samples in data X
    self.h = np.zeros((len(self.filters) * len(X), self.convolveDim, self.convolveDim))

    for j in range(self.convolveDim):
      for i in range(self.convolveDim):
        #duplicates sub input values in the z axis/depth to equal the amount of kernals, x_n[3,3] * [3, 1, 1] * fi[3, 3, 3]
        #summing the x and y axis of the product
        #duplicates at the end of the filters equal to the amount of data in X to match each data.
        self.h[:, j, i] = np.sum(
          X[:, j: j + self.filters.shape[1], i: i + self.filters.shape[2]]
          .repeat(len(self.filters), axis = 0) *
          np.tile(self.filters, (len(X), 1, 1)), axis = (1, 2))

  def maxPool(self):
    #conviently works with 1 layer or all data
    self.h2 = np.zeros((len(self.h), self.poolDim, self.poolDim))
    
    for j in range(0, self.convolveDim, self.poolStride):
      for i in range(0, self.convolveDim, self.poolStride):
        self.h2[:, j // 2, i // 2] = np.amax(
          self.h[:, j: j + self.poolStride, i: i + self.poolStride], axis = (1, 2))

  #softmax
  @staticmethod
  def AF(X):
    tmp = np.exp(X)

    return (tmp.T / tmp.sum(axis = 1)).T

  #softmax with the derivative respect to the true output
  @staticmethod
  def dAF(Y, j, i):
    #broadcasting seems to stop working past 100 samples
    tmp = -Y * Y[j, i][:, np.newaxis].repeat(Y.shape[1], axis = 1)
    tmp[j, i] = Y[j, i] * (1 - Y[j, i])

    return tmp
  
  def forward(self, X):
    self.convolve(X)
    self.maxPool()
    #dense layer
    self.h3 = np.dot(self.h2.reshape(len(X), len(self.hw)), self.hw) + self.hb

    return self.AF(self.h3)
  
  def backProp(self, X, YTrain):
    #used to access each output of Y
    index = np.arange(len(X))
    Y = self.forward(X)

    #Cross-Entropy Loss
    #[1 / Y[0, [...]], [1 / Y[1, [...]], [1 / Y[2, [...]], ...]
    dL_dY = 1 / Y[index, YTrain]
    #softmax
    dL_da = dL_dY[:, np.newaxis].repeat(Y.shape[1], axis = 1) * self.dAF(Y, index, YTrain)
    # dL_da = -(dL_dY * self.h3[index, YTrain] / self.sh3 ** 2)[:, np.newaxis] * self.h3
    # dL_da[index, YTrain] = dL_dY * self.h3[index, YTrain] / self.sh3 ** 2 * (self.sh3 - self.h3[index, YTrain])
    #dense
    dL_dh3 = np.dot(dL_da, self.hw.T).flatten()
    #applying gradient to dense weights
    self.hw += np.dot(dL_da.T, self.h2.reshape(len(dL_da), len(self.hw))).T * self.lr
    # self.hw += np.dot(self.h2.reshape(len(self.hw), len(dL_da)), dL_da) * self.lr
    self.hb += dL_da.sum(axis = 0) * self.lr
    #maxpool
    dL_dh2 = np.zeros(self.h.shape)
    #np bool array of where elements == 0
    zeros = self.h2 == 0
    for j in range(self.poolDim):
      for i in range(self.poolDim):
        for k in range(len(dL_dh2)):
          if zeros[k, j, i]:
            continue
          j2, i2 = np.where(self.h[k] == self.h2[k, j, i])
          #using index trick to turn 1D array to 2D
          dL_dh2[k, j2[0], i2[0]] = dL_dh3[(self.poolDim * (self.poolDim * k + j) + i)]
    #convolve
    # dL_dh = np.zeros(self.filters.shape)
    index = np.arange(0, len(X) * len(self.filters), len(self.filters))
    for j in range(self.convolveDim):
      for i in range(self.convolveDim):
        #duplicates the input in the z axis equal to filter * len(X) depth, multiple each i in gradient with each layer along the z axis.
        tmp = np.einsum("i,ijk->ijk", dL_dh2[:, j, i],
          X[:, j: j + self.filters.shape[1], i: i + self.filters.shape[2]]
          .repeat(len(self.filters), axis = 0)) * self.lr
        #beacuse the multiplcation results in "bands"(1, 2, 3, 1, 2, 3, ... pattern), the summation has to loop through the bands for each filter, lr is already applied
        for k in range(len(self.filters)):
          self.filters[k] += tmp[index].sum(axis = 0)
          if k != len(self.filters) - 1:
            index += 1
        index -= len(self.filters) - 1

  def export(self, filename):
    Str = f"{self.convolveDim}\n{self.lr}\n"
    for hb_n in self.hb:
      Str += str(hb_n) + " "
    Str += "\n"
    for j in self.hw:
      for i in j:
        Str += str(i) + " "
    Str += f"\n{self.hw.shape[0]} {self.hw.shape[1]}\n"
    for k in self.filters:
      for j in k:
        for i in j:
          Str += str(i) + " "
    Str += f"\n{self.filters.shape[0]} {self.filters.shape[1]} {self.filters.shape[2]}"

    with open(filename, "w") as f:
      f.write(Str)
  
  def Import(self, filename):
    with open(filename, "r") as File:
      f = File.read().splitlines()

    self.convolveDim = int(f[0])
    self.poolDim = self.convolveDim // 2
    self.lr = float(f[1])
    self.hb = np.array([float(i) for i in f[2].split()])
    self.hw = np.array([float(i) for i in f[3].split()]).reshape([int(s) for s in f[4].split()])
    self.poolStride = 2
    self.filters = np.array([float(i) for i in f[5].split()]).reshape([int(s) for s in f[6].split()])

start = time.time()
subSize = 500
img = mpimg.imread("Number.png")
mnist = np.load("mnist.npz")
# XTrain = mnist["x_train"] / 255
# YTrain = mnist["y_train"]
XTest = mnist["x_test"] / 255
YTest = mnist["y_test"]
# XTrain = np.concatenate((XTrain, XTest))
# YTrain = np.concatenate((YTrain, YTest))
# cnn = Cnn(img.shape[0], 5, 3, 10, 0.1 / subSize)
cnn = Cnn(0, 0, 0, 0, 0)

cnn.Import("Cnn1-9-21.data")
# for i in range(3):
#   for j in range(len(XTrain) // subSize):
#     cnn.backProp(XTrain[j * subSize: (j + 1) * subSize], YTrain[j * subSize: (j + 1) * subSize])

#     count = sum(np.argmax(cnn.forward(XTest[:100]), axis = 1) == YTest[:100])
#     print(f"acc: {count}%, t: {time.time() - start}s, epoch: {i}, samples: {j * subSize}")
# cnn.export("Cnn.data")

# count = sum(np.argmax(cnn.forward(XTest[:100]), axis = 1) == YTest[:100])
# print(f"acc: {count}%, t: {time.time() - start}s")
print(cnn.forward(img[np.newaxis]), np.argmax(cnn.forward(img[np.newaxis])))

fig = plt.figure()
subplot = [fig.add_subplot(i) for i in range(331, 337)]

plt.title("Cnn")
#subplot title
subplot[0].title.set_text("Image")
#set image
subplot[0].imshow(img, cmap = plt.cm.binary)
for i in range(1, len(cnn.filters) + 1):
  #subplot title
  subplot[i].title.set_text(f"Filter {i}")
  #set image
  subplot[i].imshow(cnn.h[i - 1], cmap = plt.cm.binary)
plt.show()