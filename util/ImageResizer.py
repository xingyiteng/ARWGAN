import os
from PIL import Image
from typing import Union, List


class ImageResizer:
    """图像尺寸处理类，将指定文件夹中的图像统一处理为128×128大小"""

    def __init__(self, target_size: tuple = (128, 128)):
        """
        初始化图像处理类
        Args:
            target_size: 目标图像尺寸，默认(128,128)
        """
        self.target_size = target_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    def process_single_image(self, image_path: str) -> bool:
        """
        处理单张图像
        Args:
            image_path: 图像路径
        Returns:
            bool: 处理是否成功
        """
        try:
            # 打开图像
            with Image.open(image_path) as img:
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 调整图像大小
                resized_img = img.resize(self.target_size, Image.BILINEAR)

                # 保存图像
                resized_img.save(image_path, quality=95)
            return True

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            return False

    def process_directory(self, directory_path: str) -> tuple:
        """
        处理整个文件夹中的图像
        Args:
            directory_path: 文件夹路径
        Returns:
            tuple: (成功处理数量, 失败处理数量)
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"目录 {directory_path} 不存在")

        success_count = 0
        failed_count = 0

        # 遍历文件夹中的所有文件
        for filename in os.listdir(directory_path):
            # 检查文件扩展名
            if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                image_path = os.path.join(directory_path, filename)

                # 处理图像
                if self.process_single_image(image_path):
                    success_count += 1
                else:
                    failed_count += 1

        return success_count, failed_count


# 使用示例
if __name__ == "__main__":
    # 创建图像处理器实例
    resizer = ImageResizer()

    # 指定要处理的文件夹路径
    folder_path = "D:\\workspace\\watermark\\HiDDeN\\image"

    try:
        # 处理文件夹中的所有图像
        success, failed = resizer.process_directory(folder_path)

        # 打印处理结果
        print(f"处理完成！")
        print(f"成功处理: {success} 张图像")
        print(f"处理失败: {failed} 张图像")

    except Exception as e:
        print(f"发生错误: {str(e)}")