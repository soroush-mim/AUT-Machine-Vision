import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import glob
from sklearn.cluster import KMeans
import sklearn
import random
from sklearn.metrics import mean_squared_error
#helpful lambda functions
#this function extract name of an image from its path
path_to_name = lambda path: path.split('.')[0].split('/')[1].lower()
#this function reads an image from its path and the 
#change it to grayscale and resize it
preprocess = lambda path: cv2.resize(cv2.cvtColor(cv2.imread(path), \
     cv2.COLOR_BGR2GRAY), (200,150), interpolation=cv2.INTER_AREA)
find_label = lambda test, trains, features: trains[np.argmin([mean_squared_error(features[test] , features[train]) for train in trains])][:-2]
prediction_accuracy = lambda labels, tests: sum([test[:-2] == label for label,test in zip(labels,tests)]) / len(tests)
param_to_str = lambda keys,values,sep: sep.join([f'{key}{value}' for key,value in zip(keys,values)])

def purity_score(true, pred):
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(true, pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

random.seed(1)
categories = ['bricks', 'grass', 'gravel']
#this dict contains preprocessed images as values and their names as keys
images = {path_to_name(path):preprocess(path) for path in glob.glob("dataset/*")}
#this lists contians name of 2 photos from each category that are selected randomly
tests = [f'{c}_{i+1}' for c in categories for i in random.sample(range(7), 2)]
#this list contains name of photos that are not in the test set
trains = [name for name in images if name not in tests]


def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))
    g = [g1, -g1*((x_)/(var)), g1*(((x_*x_) - var)/(var**2))][ord]
    return g
    
def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def log2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def makefilter(scale, phasex, phasey, pts, sup):
    gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
    gy = gaussian1d(scale,   0, pts[1,...], phasey)
    image = np.reshape(gx*gy,(sup,sup))
    return image

def makeLMfilters():
    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    edge, bar, spot = [], [], []
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            bar.append(makefilter(scalex[scale], 0, 1, rotpts, sup))
            edge.append(makefilter(scalex[scale], 0, 2, rotpts, sup))

    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        spot.append(gaussian2d(sup, scales[i]))

    for i in range(len(scales)):
        spot.append(log2d(sup, scales[i]))

    for i in range(len(scales)):
        spot.append(log2d(sup, 3*scales[i]))

    return edge, bar, spot

filterbanks = {}
filterbanks['LM'] = [{'kernel': filter} for filter in itertools.chain(*makeLMfilters())]


def makeRFSfilters(radius=24, sigmas=[1, 2, 4], n_orientations=6):
    
    def make_gaussian_filter(x, sigma, order=0):
        if order > 2:
            raise ValueError("Only orders up to 2 are supported")
        # compute unnormalized Gaussian response
        response = np.exp(-x ** 2 / (2. * sigma ** 2))
        if order == 1:
            response = -response * x
        elif order == 2:
            response = response * (x ** 2 - sigma ** 2)
        # normalize
        response /= np.abs(response).sum()
        return response

    def makefilter(scale, phasey, pts, sup):
        gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
        gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
        f = (gx * gy).reshape(sup, sup)
        # normalize
        f /= np.abs(f).sum()
        return f

    support = 2 * radius + 1
    x, y = np.mgrid[-radius:radius + 1, radius:-radius - 1:-1]
    orgpts = np.vstack([x.ravel(), y.ravel()])

    rot, edge, bar = [], [], []
    for sigma in sigmas:
        for orient in range(n_orientations):
            # Not 2pi as filters have symmetry
            angle = np.pi * orient / n_orientations
            c, s = np.cos(angle), np.sin(angle)
            rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
            edge.append(makefilter(sigma, 1, rotpts, support))
            bar.append(makefilter(sigma, 2, rotpts, support))
    length = np.sqrt(x ** 2 + y ** 2)
    rot.append(make_gaussian_filter(length, sigma=10))
    rot.append(make_gaussian_filter(length, sigma=10, order=2))

    # # reshape rot and edge
    # edge = np.asarray(edge)
    # edge = edge.reshape(len(sigmas), n_orientations, support, support)
    # bar = np.asarray(bar).reshape(edge.shape)
    # rot = np.asarray(rot)[:, np.newaxis, :, :]
    return edge, bar, rot



filterbanks['RFS'] = [{'kernel': filter} for filter in itertools.chain(*makeRFSfilters())]


def makeSfilters():
    params = [(2,1),(4,1),(4,2),(6,1),(6,2),(6,3),(8,1),
              (8,2),(8,3),(10,1),(10,2),(10,3),(10,4)]
    filters = [makefilter(49, *param) for param in params]
    return filters

def makefilter(sup,sigma,tau):
    hsup  = (sup - 1)/2
    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]
    [x,y] = np.meshgrid(x,y)
    r=np.sqrt(x*x+y*y)
    f=np.cos(r*(np.pi*tau/sigma))*np.exp(-(r*r)/(2*sigma*sigma))
    f=f-np.mean(f[:])
    f=f/np.sum(np.abs(f[:]))
    return f

filterbanks['S'] = [{'kernel': filter} for filter in makeSfilters()]


def analyse(filters , k):
    results = []
    for name, kernels in filters.items():
        
        features = {name:np.array([cv2.filter2D(image, cv2.CV_8UC3, kernel) for kernel in kernels]).reshape((-1)) for name,image in images.items()}

        cluster_labels = KMeans(k).fit(list(features.values())).labels_
        cluster_purity = round(purity_score([name[:-2] for name in images], list(cluster_labels)), 2)
        
        results = sorted([*results, {
            'name': name,
            'kernels': kernels,
            'features': features,
            'cluster_labels': cluster_labels,
            'cluster_purity': cluster_purity,
            'title': param_to_str(['', 'cluster_purity='],
                                  [name, cluster_purity],'\n')
            }], key=lambda x: -x['cluster_purity'])
        
    return results


filters = {f"{name}_{i}":[filter['kernel']] for name,filterbank in filterbanks.items() for i, filter in enumerate(filterbank[:])}

for k in range(2,7):
    print('clusters num: ',k)

    _filterbanks = {name:[filter['kernel'] for filter in filterbank] for name,filterbank in filterbanks.items()}
    filterbank_results = analyse(_filterbanks,k)
    print('\n\n'.join([result['title'] for result in filterbank_results]))
    print()
