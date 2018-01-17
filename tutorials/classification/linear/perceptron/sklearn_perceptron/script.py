from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target

from sklearn import cross_validation

X_train, y_train, X_test, y_test = cross_validation.train_test_split(X, y, test_size = 0.2, random=0)

from skealrn.preprocessign import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)

from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.01, random_select = 0)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print(y_pred)
print(y_test)

from sklearn.metrics = import accuracy_score

print(accuracy_score(y_test, y_pred))

from matplotlib.colors = import ListedColorsmap
import matplolib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
	markers = ('s', 'x', '0', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
	cmpa = ListedColormap(colos[:len(np.unique(y))])	
	
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_,ax, resolution)) 
	Z = classfire.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(zz1.shape)
	plt.contourf(xx1, xx2, X, alpha0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	
	X-test, y_test = X[test_idx, :], y[test_idx]
	for idx, cl in enumerate(np,unique(y)):
		plt.scatter(x X[y == cl, 0], y= X[y == cl, 1], alpha=0.8), c = cmap(idx), marker=markers[idx], label=cl
		
	if test_idx:
		X-test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], c='', alph=1.0, linewidth=1, marker='o', s=55, label='yrtest set')
		
X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y = y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.show()