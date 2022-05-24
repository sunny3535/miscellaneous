from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import random
import os
##########데이터 로드

paths = [  '1', '2', '3', '4', '5','6', '7', '8', '9', '10', '11', '12', '13', '14', '15','16', '17', '18', '19', '>', '<' ]
for path in paths: #path = "./img"
    file_list = os.listdir(path)
    print(file_list)
    for file_name in file_list:
        if file_name != path : 
            image = Image.open('./'+path+'/'+file_name)
            
            fn, ext = os.path.splitext(file_name)
            tmp = './'+path+'/'+path+'/'+fn
            
            print(fn, ext)
            image = image.convert('L') #'L': greyscale, '1': 이진화, 'RGB' , 'RGBA', 'CMYK'
            image = image.resize((150, 150))

            #밝기
            enhancer = ImageEnhance.Brightness(image)
            brightness_image = enhancer.enhance(1.8)
            brightness_image.save(tmp+'_brightness.png')

            #좌우 대칭
            horizonal_flip_image = image.transpose(Image.FLIP_LEFT_RIGHT) 
            horizonal_flip_image.save(tmp+'_horizonal_flip.png')

            #상하 대칭
            vertical_flip_image = image.transpose(Image.FLIP_TOP_BOTTOM) 
            vertical_flip_image.save(tmp+'_vertical_flip.png')

            #좌우 이동
            width, height = image.size
            shift = random.randint(0, width * 0.3)
            horizonal_shift_image = ImageChops.offset(image, shift, 0)
            horizonal_shift_image.paste((0), (0, 0, shift, height))
            horizonal_shift_image.save(tmp+'_horizonal_shift.png')

            #상하 이동
            width, height = image.size
            shift = random.randint(0, height * 0.3)
            vertical_shift_image = ImageChops.offset(image, 0, shift)
            vertical_shift_image.paste((0), (0, 0, width, shift))
            vertical_shift_image.save(tmp+'_vertical_shift.png')

            #회전 
            rotate_image = image.rotate(random.randint(-50, 50))
            rotate_image.save(tmp+'_rotate.png')

            #기울기
            #cx, cy = 0.1, 0
            #cx, cy = 0, 0.1
            cx, cy = 0, random.uniform(0.0, 0.5)
            shear_image = image.transform(
                image.size,
                method=Image.AFFINE,
                data=[1, cx, 0,
                    cy, 1, 0,])
            shear_image.save(tmp+'_shear.png')

            #확대 축소
            zoom = random.uniform(0.5, 1.7) #0.7 ~ 1.3
            width, height = image.size
            x = width / 2
            y = height / 2
            crop_image = image.crop((x - (width / 2 / zoom), y - (height / 2 / zoom), x + (width / 2 / zoom), y + (height / 2 / zoom)))
            zoom_image = crop_image.resize((width, height), Image.LANCZOS)
            zoom_image.save(tmp+'_zoom.png')