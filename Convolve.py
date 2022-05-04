class Convolve:
  def __init__(dim, amtFilters, filtersDim):
    self.filters = np.random.randn(amtFilters, filtersDim, filtersDim) / filtersDim ** 2

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