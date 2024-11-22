import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.encoder import Encoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
from SSIM import SSIM


class ARWGAN:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):

        super(ARWGAN, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.encoder = Encoder(configuration).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        # 使用Adam优化器来优化编解码器和鉴别器的参数。
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        # 如果配置了使用VGG网络，就创建一个VGG网络的损失函数对象。
        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device
        # SSIM损失
        self.ssim_loss = SSIM()
        # 二元交叉熵损失
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        # 均方误差损失
        self.mse_loss = nn.MSELoss().to(device)
        self.adversarial_weight = 0.001
        # 这些权重定义了不同损失函数在总损失中所占的比重。
        self.mse_weight = 0.7
        self.ssim_weight = 0.1
        self.decode_weight = 1.5

        # 定义了用于训练鉴别器的标签，标签值分别代表原始未加水印的图像和加了水印的图像。
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.register_backward_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.register_backward_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = self.discriminator._modules['linear']
            discrim_final.register_backward_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))

    def train_on_batch(self, batch: list):

        images, messages = batch
        batch_size = images.shape[0]

        # 设置编码器-解码器和鉴别器网络为训练模式
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # 区分训练编码器-解码器（生成器）和训练鉴别器（鉴别器）
            # ---------------- Train the discriminator -----------------------------
            # 鉴别器的优化器梯度归零，以防止之前批次的梯度影响当前批次
            self.optimizer_discrim.zero_grad()
            # train on cover

            # 设定原始图片的目标标签
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            # 设定生成图片的目标标签
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            # 它是为了训练生成器时使用的目标标签，给出的标签与封面图像的标签相同，目的是为了骗过鉴别器，让鉴别器认为加密图像实际上是封面图像。
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, (d_target_label_cover).float())
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(batch)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, (d_target_label_encoded).float())

            d_loss_on_encoded.backward()
            # 更新鉴别器网络的权重，完成这一次训练步骤。
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # 通过鉴别器得到预测值
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            # 希望鉴别器把加密图像预测成封面图像
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc.float(), g_target_label_encoded.float())

            if self.vgg_loss == None:
                # 使用均方误差损失来比较加密图像和原始图像
                g_loss_enc = self.mse_loss(encoded_images, images)

            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
            # 计算结构相似性损失
            g_loss_enc_ssim = self.ssim_loss(encoded_images, images)
            # 使用均方误差损失来比较解码器输出的消息和原始消息之间的差异。
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            # 上述所有损失通过加权求和的方式结合起来，形成总的损失函数
            g_loss = self.adversarial_weight * g_loss_adv + self.ssim_weight * (
                        1 - g_loss_enc_ssim) + self.mse_weight * (g_loss_enc) + self.decode_weight * (g_loss_dec)

            # 根据总生成器损失对网络权重计算梯度。
            g_loss.backward()

            # 更新生成器模型的权重。
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        # 返回解码消息的平均逐位误差。理想状态下，这个值应该接近0
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            'encoded_ssim   ': g_loss_enc_ssim.item(),
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():

            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(batch)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc.float(), g_target_label_encoded.float())
            g_loss_enc_ssim = self.ssim_loss(images, encoded_images)
            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.adversarial_weight * g_loss_adv + self.ssim_weight * (
                        1 - g_loss_enc_ssim) + self.mse_weight * (g_loss_enc) + self.decode_weight * (g_loss_dec)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            'encoded_ssim   ': g_loss_enc_ssim.item(),
            'PSNR           ': 10 * torch.log10(4 / g_loss_enc).item(),
            'ssim           ': 1 - g_loss_enc_ssim
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
