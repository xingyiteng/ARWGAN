import torch.nn
import argparse
import os
import utils
from model.ARWGAN import *
from noise_argparser import NoiseArgParser
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
from SSIM import SSIM
import torch.nn as nn

class MetricsCalculator:
    @staticmethod
    def main():
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        parser = argparse.ArgumentParser(description='Calculate metrics for trained models')
        parser.add_argument('--options-file', '-o', default='./pretrain/options-and-config.pickle', type=str,
                            help='The file where the simulation options are stored.')
        parser.add_argument('--checkpoint-file', '-c', default='./pretrain/checkpoints/ARWGAN.pyt', type=str,
                            help='Model checkpoint file')
        parser.add_argument('--source_images', '-s',
                            default='D:\\workspace\\watermark\\DataSet\\COCO\\data\\test\\test_class',
                            type=str, help='The image to watermark')

        args = parser.parse_args()
        train_options, net_config, noise_config = utils.load_options(args.options_file)

        # 手动设置噪声配置
        # noise_config = [Cropout((0.4, 0.4), (0.4, 0.4))]  # 这里可以根据需要更换不同的噪声
        # noise_config = [Crop((0.4, 0.4), (0.4, 0.4))]
        # noise_config = [Dropout((0.5, 0.5))]
        # noise_config = [Resize((0.8, 0.8))]
        # noise_config = ['JpegPlaceholder']

        noiser = Noiser(noise_config, device)

        checkpoint = torch.load(args.checkpoint_file, map_location=device)
        hidden_net = ARWGAN(net_config, device, noiser, None)
        utils.model_from_checkpoint(hidden_net, checkpoint)

        total_ber = 0
        total_psnr = 0
        total_ssim = 0
        image_count = 0

        ssim_loss = SSIM()
        mse_loss = nn.MSELoss().to(device)

        source_images = os.listdir(args.source_images)
        for source_image in source_images:
            image_pil = Image.open(os.path.join(args.source_images, source_image))
            image_pil = image_pil.resize((net_config.H, net_config.W))
            image_tensor = TF.to_tensor(image_pil).to(device)
            image_tensor = image_tensor * 2 - 1
            image_tensor.unsqueeze_(0)

            message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                            net_config.message_length))).to(device)

            losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch(
                [image_tensor, message])

            # 计算BER
            decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
            message_detached = message.detach().cpu().numpy()
            ber = np.mean(decoded_rounded != message_detached)
            total_ber += ber

            # 计算PSNR
            g_loss_enc = mse_loss(encoded_images, image_tensor)
            psnr = 10 * torch.log10(4 / g_loss_enc)
            total_psnr += psnr.item()

            # 计算SSIM
            g_loss_enc_ssim = ssim_loss(encoded_images, image_tensor)
            total_ssim += g_loss_enc_ssim.item()

            image_count += 1

        # 计算并输出平均指标
        avg_ber = total_ber / image_count
        avg_psnr = total_psnr / image_count
        avg_ssim = total_ssim / image_count

        print(f'Average Correct Bit Rate : {1 - avg_ber:.3f}')
        print(f'Average PSNR : {avg_psnr:.3f}')
        print(f'Average SSIM : {avg_ssim:.3f}')

if __name__ == '__main__':
    MetricsCalculator.main()