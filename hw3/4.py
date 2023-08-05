import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

#a function for plotting filters
def plot_grid_images(images, rows, cols, name, scale=1 ):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*scale,rows*scale), subplot_kw=dict(xticks=[], yticks=[]))

    for i, image in enumerate(images):
        axes[i//cols, i%cols].imshow(image, cmap='gray')
    #plt.show()
    fig.savefig(name)
#first derivatives of Gaussians
def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))
    g = [g1, -g1*((x_)/(var)), g1*(((x_*x_) - var)/(var**2))][ord]
    return g
#second derivatives of Gaussians
def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g
#Laplacian of Gaussian
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

plot_grid_images(images=[cv2.resize(filter['kernel'], (100, 100)) for filter in filterbanks['LM']],
                 rows=4, cols=12 , name = 'LM.png')



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

plot_grid_images(images=[cv2.resize(filter['kernel'], (100, 100)) for filter in filterbanks['RFS']],
                 rows=4, cols=10 , name='RFS.png')


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

plot_grid_images(images=[cv2.resize(filter['kernel'], (100, 100)) for filter in filterbanks['S']],
                 rows=4, cols=4 , name='S.png')