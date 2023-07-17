
from PIL import Image, ImageDraw, ImageFont
import platform
import numpy as np
def add_text_watermark_on_img(img, text, rot_angle,
                              font='arial.ttf',
                              fontsize=55,
                              ):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert('RGBA')
    ori_img_size = img.size

    # --- text image ---
    if platform.system() == 'Windows':
        font = ImageFont.truetype('arial.ttf', fontsize)
    elif platform.system() == 'Linux':
        font = ImageFont.truetype('DejaVuSansMono.ttf', fontsize)
    # font = ImageFont.truetype(font, fontsize)
    # font = ImageFont.load_default()

    # calculate text size in pixels (width, height)
    text_size = font.getsize(text) 
    # text_size = (300, 50)
    # create image for text
    text_img = Image.new('RGBA', text_size, (255,255,255,0))
    text_draw = ImageDraw.Draw(text_img)

    # draw text on image
    text_draw.text((0, 0), text, (255, 255, 255, 129), font=font)
    # rotate text image and fill with transparent color
    rotated_text_img = text_img.rotate(rot_angle, expand=True, fillcolor=(0,0,0,0))
    rotated_text_img_size = rotated_text_img.size

    # image with the same size and transparent color (..., ..., ..., 0)
    watermark_img = Image.new('RGBA', ori_img_size, (255,255,255,0))

    # calculate top/left corner for centered text
    x = ori_img_size[0]//2 - rotated_text_img_size[0]//2
    y = ori_img_size[1]//2 - rotated_text_img_size[1]//2

    # put text on watermarks image
    watermark_img.paste(rotated_text_img, (x, y))
    combined_image = Image.alpha_composite(img, watermark_img)
    return combined_image
