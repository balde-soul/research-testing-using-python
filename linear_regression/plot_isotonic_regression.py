"""
===================
Isotonic Regression
===================

An illustration of the isotonic regression on generated data. The
isotonic regression finds a non-decreasing approximation of a function
while minimizing the mean squared error on the training data. The benefit
of such a model is that it does not assume any form for the target
function such as linearity. For comparison a linear regression is also
presented.

"""
print(__doc__)

# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

lr_ = LinearRegression(fit_intercept=False)
lr_.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression witout intercept

# analysis
# # coefficient of determination
lr_r2 = lr.score(np.expand_dims(x, 1), y)
print(lr_r2)
lr__r2 = lr_.score(np.expand_dims(x, 1), y)
print(lr__r2)
# #############################################################################

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=6)
y_pre = lr.predict(x[:, np.newaxis])
plt.plot(x, y_pre, 'b-')
plt.text(x[-1], y_pre[-1], 'r2: {}'.format(round(lr_r2, 2)), ha='center', va='bottom', fontsize=8)
y_pre = lr_.predict(x[:, np.newaxis])
plt.plot(x, y_pre, 'r-')
plt.text(x[-1], y_pre[-1], 'r2: {}'.format(round(lr__r2, 2)), ha='center', va='bottom', fontsize=8)
# plt.gca().add_collection(lc)
plt.legend(('Data', 'Linear Fit', 'Linear Fit no Intercept'), loc='lower right')
plt.title('Isotonic regression')
plt.savefig("./linear_regression/generate/learn_regression.png")