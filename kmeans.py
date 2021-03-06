import numpy as np

# Notkun: centroids, labels, J = kmeans(data,k)
# Inntak: data, n x N fylki, n er víddin á punktunum og það eru N punktar
#         k: fjöldi hópa sem á að finna
# Úttak: centroids er n x k fylki af k fulltrúum fyrir hvern hóp
#        labels er N-vigur með gildum frá 0..k-1, þar sem labels[i] er hópurinn
#        sem punktur i var settur í
#        J er J-gildið fyrir hverja ítrun af k-means
def kmeans(data,k):
  n,N = data.shape

  losses = []

  eps = 1e-6
  centroids,labels = init_centroids(data,k)

  while True:
    labels = partition_data(data,centroids)
    centroids = update_centroids(data, labels, k, centroids)
    losses.append(loss(data,centroids,labels))
    if len(losses) >= 2 and abs(losses[-1] - losses[-2]) <= eps:
      break

  return centroids, labels, losses

def init_centroids(data,k):
  centroids = np.zeros((n,k))
  labels = np.zeros(N)

  return centroids,labels

def dist2(x,y):
    u = x-y
    return np.dot(u,u)

def partition_data(data,centroids):
  N = data.shape[1]
  labels = np.zeros(N,dtype=int)
  for i in range(N):
    distances = np.apply_along_axis(lambda x: dist2(x,data[:,i]),0,centroids)
    labels[i] = np.argmin(distances)
  return labels


def update_centroids(data, labels, k, old_centroids):
  n = data.shape[0]
  centroids = np.zeros((n,k))

  for j in range(k):
    points = data[:,labels==j]
    if points.size == 0:
      centroids[:,j] = old_centroids[:,j]
    else:
      centroids[:,j] = points.mean(axis=1)
  return centroids

def loss(data, centroids, labels):
  N = data.shape[1]
  s = 0.0
  for i in range(N):
    s += dist2(data[:,i],centroids[:,labels[i]])
  return s/N

def init_centroids(data, k):
  n,N = data.shape
  labels = np.random.randint(0,k,size=N)
  centroids = update_centroids(data, labels, k, np.zeros((n,k)))

  return centroids, labels


if __name__ == "__main__":
  data = np.array(np.bmat([np.ones((3,10)), np.zeros((3,20))]))
  c,l,J = kmeans(data,2)
  print(c)
  print(l)
  print(J)
