from PIL import Image
import numpy as np
from numpy import asarray 
import matplotlib.pyplot as plt

def Input(ImageName): #Read numpy array from the image
    img = Image.open(ImageName)
    numpydata = asarray(img)
    return numpydata

def Reshaping2d(img_3d): #Reshape from 3d to 2d matrix
    img_shape = img_3d.shape
    img_2d = img_3d.reshape(img_shape[0]*img_shape[1],img_shape[2])
    print(img_2d)
    print(img_3d.shape)
    print(img_2d.shape)
    return img_2d
    
def Reshaping3d(img_2d,shape_3d): #Reshape from 2d to 3d matrix
    img_3d = img_2d.reshape((shape_3d[0],shape_3d[1],shape_3d[2]))
    return img_3d
    
def EuclideanDistance(a,b):
    d = np.square(a[0] -b[0]) + np.square(a[1] - b[1])
    return (np.sqrt(d))

def Initialize(img_2d,k_clusters):
    m, n = img_2d.shape
    # centroids is the array of assumed means or centroids.
    centroids = np.zeros((k_clusters, n)) 
    # initialize random centroids.
    for i in range(k_clusters):
        #choose 10 randoms from the 1d array
        ranIndices = np.random.choice(m, size=10, replace=False) 
        #calculate avg value of each centroids
        centroids[i] = np.mean(img_2d[ranIndices], axis=0)
    return centroids
      
def Kmeans(img_2d, centroids, k_clusters, max_iter):
    # max_iter the number of iterations
    m, n = img_2d.shape
    # these are the index values that correspond to the cluster to
    # which each pixel belongs to.
    index = np.zeros(m)
    # k-means algorithm.
    while max_iter > 0:
        for j in range(m):
            # initialize minimum value to a large value
            min_dist = float('inf')
            temp = None

            for k in range(k_clusters):
                a = (img_2d[j, 0], img_2d[j, 1])
                b = (centroids[k, 0], centroids[k, 1])

                if EuclideanDistance(a,b) <= min_dist:
                    min_dist = EuclideanDistance(a,b)
                    temp = k
                    index[j] = k
                    
        for k in range(k_clusters):
            cluster_points = img_2d[index == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
                
        max_iter -= 1
    return centroids, index
 
def Compress(centroids, index):
    centroid = np.array(centroids.astype(np.uint8))
    assign = centroid[index.astype(int)]
    return assign
    
def Output(ImageType,numpydata,FileName,k_clusters):
    if ImageType == 'png':
        FinalName = FileName[0] + '_' + str(k_clusters) + 'colours' + '.png'
    if ImageType == 'pdf':
        FinalName = FileName[0] + '_' + str(k_clusters) + 'colours' + '.pdf'
    if ImageType == 'jpg':
        FinalName = FileName[0] + '_' + str(k_clusters) + 'colours' + '.jpg'
    data = Image.fromarray(numpydata)
    data.save(FinalName)
    plt.imshow(numpydata)
    plt.show()
        
        
if __name__ == "__main__":
    ImageName = input("Please enter the input name of a file: ")
    img_3d = Input(ImageName)
    FileName = ImageName.split('.')
    shape_3d = img_3d.shape
    k_clusters = int(input("Please enter the number of colors: "))
    img_2d = Reshaping2d(img_3d)
    centroids = Initialize(img_2d,k_clusters)
    centroids,index = Kmeans(img_2d,centroids,k_clusters,20)
    compress = Compress(centroids,index)
    reshaped = Reshaping3d(compress,shape_3d)
    ImageType = input("Please enter the type of output file (pdf/png/jpg): ")
    Output(ImageType,reshaped,FileName,k_clusters)
    
    
