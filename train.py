


from dataset.cocoDatasets import create_dataloader
from models.yolo import Model
from loss.loss import compute_loss
import yaml
import torch 
import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.utils import check_anchors

if __name__ == "__main__":

    train_path = "/home/wu/Desktop/yolov3-copy/data/coco/trainvalno5k.txt"
    epochs = 100

    # Read config
    with open('./config/hyp.scratch.yaml') as f:    
        modelHyp = yaml.load(f, Loader=yaml.FullLoader)  # load model hyps

    with open('./config/coco128.yaml') as f:    
        datasetHyp = yaml.load(f, Loader=yaml.FullLoader)  # load dataset hyps

    # training Data loader
    dataloader, dataset = create_dataloader(train_path, imgsz=416, batch_size=4, stride=8, opt=None, hyp=modelHyp, augment=True)
    
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load model
    model = Model("./config/yolov5m.yaml")
    print(model)
    model = model.to(device)

    check_anchors(dataset, model=model, thr=modelHyp['anchor_t'], imgsz=416)


    nc, names = (int(datasetHyp['nc']), datasetHyp['names'])  # number classes, names
    modelHyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = modelHyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    params = [p for p in model.parameters() if p.requires_grad]

    # optimizer and learnign schedule
    optimizer = optim.Adam(
        params, 
        lr=0.1, 
        betas=(0.9, 0.999)
    )  # adjust beta1 to momentum
    lr_schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=0, last_epoch=-1)

    for epoch in range(epochs):
        model.train()
        #continue 
        for batch_i, (imgs, targets, paths, _) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            optimizer.zero_grad()
            batches_done = len(dataloader) * epoch + batch_i 
            imgs = imgs.to(device).float() / 255.0
            #print(imgs.dtype)

            output = model(imgs)
            #print(output)
            #print(imgs.numpy()[0].reshape(640, 640, 3).shape) 
            loss, loss_items = compute_loss(output, targets.to(device), model)  # loss scaled by batch_size
            #print(loss)
             
            loss.backward()
            # Run optimizer
            optimizer.step()

        lr_schedule.step()