import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from sklearn.metrics import mean_squared_error
#helpful lambda functions
#this function extract name of an image from its path
path_to_name = lambda path: path.split('.')[0].split('/')[1].lower()
#this function reads an image from its path and the 
#change it to grayscale and resize it
preprocess = lambda path: cv2.resize(cv2.cvtColor(cv2.imread(path), \
     cv2.COLOR_BGR2GRAY), (200,150), interpolation=cv2.INTER_AREA)
#find label based on mse errors on features
find_label = lambda test, trains, features: trains[np.argmin([mean_squared_error(features[test] , features[train]) for train in trains])][:-2]
#create feature planes on image with appliing kernels on them
get_features = lambda image , kernels: [cv2.filter2D(image, cv2.CV_8UC3, kernel) for kernel in kernels]

def plot_kernel_feature(images, filterbank,fig_name, labels,predict, scale=2):
    rows, cols = len(filterbank), len(labels)
    fig, axes = plt.subplots(rows+1, cols+1, figsize=(cols*scale,rows*scale), subplot_kw=dict(xticks=[], yticks=[]))
    axes[0][0].axis('off')

    for i, filter in enumerate(filterbank):
        axes[i+1,0].imshow(np.real(cv2.resize(filter['kernel'], (200, 150))), cmap='gray')
        label = 'T:'+str(filter['param'][2]) +' K:'+str(filter['param'][0][0]) +' P:'+str(filter['param'][-1])+' G:'+str(filter['param'][-2]) +' L:'+str(filter['param'][3])
        axes[i+1,0].set_ylabel( label , fontsize=6)
        
        for j, name in enumerate(labels):
            if i == 0:
                axes[i,j+1].imshow(images[name], cmap='gray')
                axes[i,j+1].set_title(name[:-2]+'->'+ predict[j], fontsize=6)
            features = cv2.filter2D(images[name], cv2.CV_8UC3, filter['kernel'])
            axes[i+1,j+1].imshow(features, cmap='gray')
    #plt.show()
    fig.savefig(fig_name)

def makeGfilters(params):
    filters = []
    for param in itertools.product(*params.values()):
        kernel = cv2.getGaborKernel(*param, ktype=cv2.CV_32F)
        filters.append({
            'param': param,
            'kernel': kernel,
            })
    return filters


random.seed(2)
categories = ['bricks', 'grass', 'gravel']
#this dict contains preprocessed images as values and their names as keys
images = {path_to_name(path):preprocess(path) for path in glob.glob("dataset/*")}
#this lists contians name of 2 photos from each category that are selected randomly
tests = [f'{c}_{i+1}' for c in categories for i in random.sample(range(7), 2)]
#this list contains name of photos that are not in the test set
trains = [name for name in images if name not in tests]


#explaining paprameters of gabor filter
# https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97

filterbank = makeGfilters({
    'ksize': [(3,3)  , (6,6)  ] ,# (6,6) ], 
    'sigma': [2], 
    'theta': [round(i*np.pi, 5) for i in [0,1/2 , 1/4 , 3/4]],
    'lambda': [3, 5 ],
    'gamma': [ 1,3],# , 2.0],  
    'psi': [ .4 , .6 ]# , .8],  
})

#selected gabor filters:
selcted_kernels =  [6,17,41 ,15,16,40]
_filterbank = [filterbank[i] for i in selcted_kernels]
#printing parameters of selected kernels
for i in selcted_kernels:
    print('parameters of gabor filter '+str(i))
    print(filterbank[i]['param'])
    print()

#extracting features
kernels = [filter['kernel'] for filter in _filterbank]
features = {name : get_features(images[name] , kernels)  for name in images}
#concating feature planes for each image
for name in features:
    feature = np.array([])
    for i in features[name]:
        feature = np.concatenate([feature , i.flatten()])
    features[name] = feature
#predicting labels using mse 
predicted_labels = []
for label in tests:
    print(label)
    print(find_label(label , trains , features))
    print()
    predicted_labels.append(find_label(label , trains , features))


plot_kernel_feature(
    images=images,
    labels = tests,
    predict = predicted_labels,
    filterbank = _filterbank[:3],
    fig_name= 'predict1.png'
    )

plot_kernel_feature(
    images=images,
    labels = tests,
    predict = predicted_labels,
    filterbank = _filterbank[3:],
    fig_name= 'predict2.png'
    )