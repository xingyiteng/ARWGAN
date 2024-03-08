import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.ARWGAN import ARWGAN
from average_meter import AverageMeter


def train(model: ARWGAN,
          device: torch.device,
          net_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):

    # 加载数据加载器
    train_data, val_data = utils.get_data_loaders(net_config, train_options)
    # 训练数据集中的文件数目 后续决定每个epoch中需要执行的步骤数
    file_count = len(train_data.dataset)

    # 如果文件总数可以被批量大小整除
    if file_count % train_options.batch_size == 0:
        # 步骤数 = 文件总数 / 批量大小
        steps_in_epoch = file_count // train_options.batch_size
    else:
        # 步骤数 = 文件总数 / 批量大小 + 1
        steps_in_epoch = file_count // train_options.batch_size + 1

    # 每print_each步打印一次训练信息。
    print_each = 10
    # 保存8张用于检查和可视化的图像数据。
    images_to_save = 8
    # 保存图像数据用于监视和可视化时，它们的分辨率应该调整为512x512像素。
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        # 存储损失的平均值
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        # 标签被忽略，表示为_
        for image, _ in train_data:
            # 将图片移动到之前定义的设备
            image = image.to(device)
            # 生成随机的message
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], net_config.message_length))).to(device)
            # 获得图像损失
            losses, _ = model.train_on_batch([image, message])
            # 更新 training_losses 以记录损失的平均值。
            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        # 记录完整epoch训练的总时间
        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        # 将损失和时间写入一个CSV文件
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            # 将损失、梯度和张量值保存进日志。
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            # 随机生成一个与图像批次大小相符的二进制消息
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], net_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_tensors) = model.validate_on_batch([image, message])
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                if net_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                    # noised_images=noised_images.float()
                # utils.save_images(image.cpu()[:images_to_save, :, :, :],
                #                   encoded_images[:images_to_save, :, :, :].cpu(),
                #                   image.cpu()[:images_to_save, :, :, :]-encoded_images[:images_to_save, :, :, :].cpu(),
                #                   epoch,
                #                   os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False
        # 记录验证过程中的损失进度。
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        # 保存当前模型的状态，可以在未来的某个时间点从当前epoch重新开始训练。
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        # 记录验证损失到一个CSV文件
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
