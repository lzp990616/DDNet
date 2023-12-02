from PIL import Image

# 存储行图片的文件夹路径
row_folder = './pic_row/'

# 定义合并后的大图的宽度和高度
width = 5 * 384  # 假设每行图片都是384x384
height = 4 * 384-224  # 假设有9行图片

# 创建一个空白图像
merged_image = Image.new('RGB', (width, height), color=(255, 255, 255))

for i in range(6):
    row_image_file = row_folder + 'row_{}.png'.format(i)
    row_image = Image.open(row_image_file)
    # 裁剪图片
    if i == 0:
        crop_top = row_image.height // 3-10  # 上边界
        crop_bottom = crop_top * 3+30  # 下边界
        row_image = row_image.crop((0, crop_top-40, row_image.width, crop_bottom-98))
    else:
        crop_top = row_image.height // 3-10  # 上边界
        crop_bottom = crop_top * 3+30  # 下边界
        row_image = row_image.crop((0, crop_top, row_image.width, crop_bottom))
    # 调整图片大小
    # row_image = row_image.resize((384, 384))
    merged_image.paste(row_image, (0, i * 156))

# 保存合并后的大图
merged_image.save('./merged_image.png')

