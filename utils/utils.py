import os
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import math
import yaml
import time
from tqdm import tqdm
from scipy.cluster.vq import kmeans
import torchvision




class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    file = Path(weights).name

    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3.pt']  # available models
    print(os.path.isfile(weights))
    if file in models and not os.path.isfile(weights):
        # Google Drive
        # d = {'yolov5s.pt': '1R5T6rIyy3lLwgFXNms8whc-387H0tMQO',
        #      'yolov5m.pt': '1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr',
        #      'yolov5l.pt': '1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV',
        #      'yolov5x.pt': '1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS'}
        # r = gdrive_download(id=d[file], name=weights) if file in d else 1
        # if r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6:  # check
        #    return

        try:  # GitHub
            url = 'https://github.com/ultralytics/yolov3/releases/download/v3.0/' + file
            
            print('Downloading %s to %s...' % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('Download error: %s' % e)
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                print('ERROR: Download failure: %s' % msg)
            print('')
            return

def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.general import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        from dataset.cocoDatasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)

def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    print('\nAnalyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to generate improved anchors, please wait...' % bpr)
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline




def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    attempt_load('yolov5m.pt',device)