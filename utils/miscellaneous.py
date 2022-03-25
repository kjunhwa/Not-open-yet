import torch.optim as optim
from torch.optim import lr_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy




def optimizer_set(args, model_ft):

    if args.optim == 'adam':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    elif args.optim == 'sgd':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adamw':
        optimizer_ft = optim.AdamW(model_ft.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    return optimizer_ft
    
    
def scheduler_set(args, optimizer_ft):

    scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[20,30,40], gamma=0.1)
    
    return scheduler
    
    
def loss_set(args):

    loss = SoftTargetCrossEntropy()
    
    return loss
