import os

import torch
from tqdm import tqdm
from functools import partial
from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,epoch_step_ir, epoch_step_val_ir, gen, gen_val,gen_ir,gen_val_ir, Epoch, cuda, fp16, scaler, save_period, save_dir,save_dir_ir,train_dataset_ir,batch_size, local_rank=0):
    loss = 0
    val_loss = 0
    da_loss=0
    da_val_loss=0

    loss_ir = 0
    val_loss_ir = 0
    da_loss_ir = 0
    da_val_loss_ir = 0
    # 准备训练
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    # source训练
    gen_ir_iter = iter(gen_ir)
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]

        # 增加ir训练
        try:
            images_ir, targets_ir = next(gen_ir_iter)
        except StopIteration:
            gen_ir_iter = iter(gen_ir)
            images_ir, targets_ir = next(gen_ir_iter)

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                images_ir = images_ir.cuda(local_rank)
                targets_ir = targets_ir.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs,da,da_one,da_w         = model_train(images)
            loss_value,da_lo     = yolo_loss(outputs, targets, images,da,da_one,da_w,flag=0,epoch=epoch)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, da, da_one, da_w = model_train(images_ir)
            loss_value_ir, da_lo_ir = yolo_loss(outputs, targets_ir, images_ir, da, da_one, da_w, flag=1, epoch=epoch, flag_val=0)  # 训练时，ir的检测损失为0

            #----------------------#
            #   反向传播
            #----------------------#
            (loss_value+loss_value_ir).backward()
            optimizer.step()
        if ema:
            ema.update(model_train)

        loss += (loss_value+loss_value_ir).item()
        da_loss+=(da_lo+da_lo_ir).item()

        # da_loss=0
        # print('da',da_lo.item())
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)


    # 增加ir验证
    if local_rank == 0:
        pbar.close()
        print('Start Validation_ir')
        pbar = tqdm(total=epoch_step_val_ir, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

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
            outputs,da,da_one,da_w         = model_train_eval(images)
            loss_value,da_lo      = yolo_loss(outputs, targets, images,da,da_one,da_w,flag=1,epoch=epoch)

        val_loss_ir += loss_value.item()
        da_val_loss_ir+=da_lo.item()
        # da_val_loss_ir=0
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss_ir': val_loss_ir / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, (val_loss_ir-da_val_loss) / epoch_step_val_ir,da_loss/epoch_step)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f' % (loss / epoch_step, (val_loss_ir-da_val_loss) / epoch_step_val_ir))
        print('DA Loss: %.5f || DA_Val Loss: %.5f ' % (da_loss / epoch_step, da_val_loss_ir / epoch_step_val_ir))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir_ir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss_ir / epoch_step_val_ir)))
            
        if len(loss_history.val_loss) <= 1 or ((val_loss_ir-da_val_loss) / epoch_step_val_ir) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir_ir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir_ir, "last_epoch_weights.pth"))