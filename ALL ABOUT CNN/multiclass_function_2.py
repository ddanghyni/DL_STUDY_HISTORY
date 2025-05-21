#%% library
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
#%% DEVICE
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#%% Train function
def Train(model, train_DL, val_DL, criterion, optimizer, EPOCH,
          BATCH_SIZE, TRAIN_RATIO,
          save_model_path, save_history_path, **kwargs):

    if "LR_STEP" in kwargs:
        scheduler = StepLR(optimizer, step_size = kwargs["LR_STEP"], gamma = kwargs["LR_GAMMA"])
    else:
        scheduler = None

    # loss & acc 를 누적하기 위한 빈 리스트 생성
    loss_history = {'train': [], 'val': []}
    acc_history = {'train': [], 'val': []}
    best_loss = 9999

    for ep in range(EPOCH):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep + 1}, current_LR = {current_lr}")

        model.train() # train mode로 전환
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, optimizer)
        loss_history['train'] += [train_loss]
        acc_history['train'] += [train_acc]

        model.eval() # test mode로 전환
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion)
            loss_history['val'] += [val_loss]
            acc_history['val'] += [val_acc]

            if val_loss < best_loss:
                '''
                <early stopping>
                val이 가장 작을 때 parms를 save하고 학습은 계속 진행
                '''
                best_loss = val_loss
                # optimizer도 같이 save하면 여기서부터 재학습 시작 가능
                torch.save({"model" : model,
                                 "ep" : ep,
                                 "opmtimizer" : optimizer,
                                 "scheduler":scheduler},save_model_path)

        if "LR_STEP" in kwargs:
            scheduler.step()

        # print loss
        print(f"train loss: {round(train_loss, 5)}, "
              f"val loss: {round(val_loss, 5)} \n"
              f"train acc: {round(train_acc, 1)} %, "
              f"val acc: {round(val_acc, 1)} %, time: {round(time.time() - epoch_start)} s")
        print("-" * 20)

    torch.save({"loss_history": loss_history,
                "acc_history": acc_history,
                "EPOCH": EPOCH,
                "BATCH_SIZE": BATCH_SIZE,
                "TRAIN_RATIO": TRAIN_RATIO}, save_history_path)
    return loss_history

#%% Test Function
def Test(model, test_DL, criterion):

    # 평가 모드로 변형
    model.eval()

    # Test에선 미분 x -> 메모리 효율성을 위해
    with torch.no_grad():
        test_loss_e, test_accuracy_e, rcorrect = loss_epoch(model, test_DL, criterion)
    print()
    print(f"Test loss: {round(test_loss_e, 5)}")
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(test_accuracy_e, 1)} %)")

    return round(test_accuracy_e, 1)

#%% Loss EPOCH
def loss_epoch(model, DL, criterion,optimizer = None):

    # the number of data
    N = len(DL.dataset)
    rloss = 0; rcorrect = 0

    for x_batch, y_batch in tqdm(DL, leave=False):

        # model이랑 Data랑 같은 DEVICE 위에 있어야함.
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # inference
        y_hat = model(x_batch)

        # loss
        loss = criterion(y_hat, y_batch)

        # update
        if optimizer is not None:
            optimizer.zero_grad()  # gradient 누적을 막기 위한 초기화
            loss.backward()  # backpropagation
            optimizer.step()  # weight update

        # loss accum
        loss_b = loss.item() * x_batch.shape[0]
        rloss += loss_b
        # accuracy accumulation
        pred = y_hat.argmax(dim=1)  # check
        corrects_b = torch.sum(pred == y_batch).item()  # tensor(1) -> 1 로
        rcorrect += corrects_b

    # print loss & acc
    loss_e = rloss / N  # 전체 Loss
    accuracy_e = rcorrect / N * 100

    return loss_e, accuracy_e, rcorrect

#%% Ploting function
def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8, 4))
    for idx in range(6):
        plt.subplot(2, 3, idx + 1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1, 2, 0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color="g" if pred_class == true_class else "r")

#%% IM_PLOT
def im_plot(DL):
    x_batch, y_batch = next(iter(DL))
    plt.figure(figsize=(8, 4))

    for idx in range(6):
        im = x_batch[idx]
        im = im - im.min()  # for imshow clipping
        im = im / im.max()  # for imshow clipping

        plt.subplot(2, 3, idx + 1, xticks=[], yticks=[])
        plt.imshow(im.permute(1, 2, 0).squeeze())
        true_class = DL.dataset.classes[y_batch[idx]]
        plt.title(true_class, color="g")
    print(f"x_batch size = {x_batch.shape}")

#%% Count number of params
def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num

