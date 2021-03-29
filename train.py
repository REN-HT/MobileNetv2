import torch
from tqdm import tqdm
from config import opt
from dataset.DataSet import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.mobileNetv2 import mobileNetv2


def train():
    net = mobileNetv2()
#     state_dic=torch.load('C:/AllProgram/Pytorch/MobileNetv2/model_weight.pth')
#     net.load_state_dict(state_dic)
    if opt.use_gpu:
        net=net.cuda()
    train_data = DogCat(opt.train_root, train=True)
    # val_data = DogCat(opt.train_root, train=False)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    lr = opt.lr
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,  weight_decay=opt.weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    max_epoch = opt.epoch

    for epoch in range(max_epoch):
        train_loss=0
        for ii, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = Variable(data[0])
            target = Variable(data[1])

            if opt.use_gpu:
                inputs = inputs.cuda()
                target = target.cuda()
                criterion = criterion.cuda()

            optimizer.zero_grad()  # 梯度清零
            score = net(inputs)

            loss = criterion(score, target)
            if opt.use_gpu:
                loss=loss.cpu()
            train_loss+=float(loss.item())

            # {:.2f}保留两位小数
            print('{} epoch loss:{:.2f}'.format(epoch + 1, train_loss / (ii + 1)))
            loss.backward()
            optimizer.step()  # 参数更新

        lr_scheduler.step()
        torch.save(net.state_dict(), 'model_weight{}.pth'.format(epoch+1))


if __name__ == '__main__':
    train()
