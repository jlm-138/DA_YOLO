
import os

import torch
from tqdm import tqdm
from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,epoch_step_ir, epoch_step_val_ir, gen, gen_val,gen_ir,gen_val_ir, Epoch, cuda, fp16, scaler, save_period, save_dir,save_dir_ir, local_rank=0):
    loss        = 0
    val_loss    = 0
    da_loss=0
    da_val_loss=0
    # 准备训练
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    # source训练
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs,da         = model_train(images)
            loss_value,da_lo      = yolo_loss(outputs, targets, images,da,flag=0,epoch=epoch)

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        da_loss+=da_lo.item()
        print('da_loss',da_lo.item())
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    for myi in range(int(epoch_step/epoch_step_ir)+1):
        # 增加ir训练
        loss_ir = 0
        da_loss_ir = 0

        if local_rank == 0:
            print('Start Train_ir'+'---'+str(myi))
            pbar.close()
            pbar = tqdm(total=epoch_step_ir, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

        for iteration, batch in enumerate(gen_ir):
            if iteration >= epoch_step:
                break
            if myi== int(epoch_step/epoch_step_ir) and iteration==(epoch_step%epoch_step_ir):
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = images.cuda(local_rank)
                    targets = targets.cuda(local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            if not fp16:
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs,da = model_train(images)
                loss_value,da_lo = yolo_loss(outputs, targets, images,da,flag=1,epoch=epoch)

                # ----------------------#
                #   反向传播
                # ----------------------#
                loss_value.backward()
                optimizer.step()
            if ema:
                ema.update(model_train)

            loss_ir += loss_value.item()
            da_loss_ir+=da_lo.item()
            print('ir_daloss',da_lo.item())
            if local_rank == 0:
                pbar.set_postfix(**{'loss_ir': loss_ir / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
        if myi==int(epoch_step/epoch_step_ir)-1:
            my_loss_ir=loss_ir
            my_da_loss_ir=da_loss_ir

    # 准备验证
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    # source验证
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs,da         = model_train_eval(images)
            loss_value,da_lo      = yolo_loss(outputs, targets, images,da,flag=0,epoch=epoch)

        val_loss += loss_value.item()
        da_val_loss+=da_lo.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    # 增加ir验证
    val_loss_ir = 0
    da_val_loss_ir = 0
    if local_rank == 0:
        pbar.close()
        print('Start Validation_ir')
        pbar = tqdm(total=epoch_step_val_ir, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    for iteration, batch in enumerate(gen_val_ir):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs,da         = model_train_eval(images)
            loss_value,da_lo      = yolo_loss(outputs, targets, images,da,flag=1,epoch=epoch)

        val_loss_ir += loss_value.item()
        da_val_loss_ir+=da_lo.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss_ir': val_loss_ir / (iteration + 1)})
            pbar.update(1)


    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, my_loss_ir / epoch_step_ir, val_loss_ir / epoch_step_val_ir)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f  Total Loss_ir: %.3f || Val Loss_ir: %.3f' % (loss / epoch_step, val_loss / epoch_step_val,my_loss_ir / epoch_step_ir, val_loss_ir / epoch_step_val_ir))
        print('DA Loss: %.5f || DA_Val Loss: %.5f  DA Loss_ir: %.5f || DA_Val Loss_ir: %.5f' % (
        da_loss / epoch_step, da_val_loss / epoch_step_val, my_da_loss_ir / epoch_step_ir, da_val_loss_ir / epoch_step_val_ir))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir_ir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, my_loss_ir / epoch_step_ir, val_loss_ir / epoch_step_val_ir)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss_ir / epoch_step_val_ir) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir_ir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir_ir, "last_epoch_weights.pth"))