import numpy as np

def project(W, X, mu=None):
	if mu is None:
		return np.dot(X,W)
	return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):
	if mu is None:
		return np.dot(Y,W.T)
	return np.dot(Y, W.T) + mu

def pca(X, y, num_components=0):
	[n,d] = X.shape
	if (num_components <= 0) or (num_components>n):
		num_components = n
	mu = X.mean(axis=0)
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in xrange(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	# or simply perform an economy size decomposition
	# eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
	# sort eigenvectors descending by their eigenvalue
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	# select only num_components
	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:,0:num_components].copy()
	return [eigenvalues, eigenvectors, mu]
		
def lda(X, y, num_components=0):
	y = np.asarray(y)
	[n,d] = X.shape
	c = np.unique(y)
	if (num_components <= 0) or (num_component>(len(c)-1)):
		num_components = (len(c)-1)
	meanTotal = X.mean(axis=0)
	Sw = np.zeros((d, d), dtype=np.float32)
	Sb = np.zeros((d, d), dtype=np.float32)
	for i in c:
		Xi = X[np.where(y==i)[0],:]
		meanClass = Xi.mean(axis=0)
		Sw = Sw + np.dot((Xi-meanClass).T, (Xi-meanClass))
		Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))
	eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
	idx = np.argsort(-eigenvalues.real)
	eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
	eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
	eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)
	return [eigenvalues, eigenvectors]

def fisherfaces(X,y,num_components=0):
	y = np.asarray(y)
	[n,d] = X.shape
	c = len(np.unique(y))
	[eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, (n-c))
	[eigenvalues_lda, eigenvectors_lda] = lda(project(eigenvectors_pca, X, mu_pca), y, num_components)
	eigenvectors = np.dot(eigenvectors_pca,eigenvectors_lda)
	return [eigenvalues_lda, eigenvectors, mu_pca]
