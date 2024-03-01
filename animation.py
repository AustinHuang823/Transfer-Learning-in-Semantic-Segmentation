import imageio
from PIL import Image, ImageDraw, ImageFont
import os

dirs = ["results_0422_v2/iter_{:05d}".format(i) for i in range(26001, 26501)]
image_files = ["{}/2_image.png".format(d) for d in dirs]
target_files = ["{}/2_target.png".format(d) for d in dirs]
overlay_files = ["{}/2_overlay.png".format(d) for d in dirs]
pred_files = ["{}/2_pred.png".format(d) for d in dirs]

gif = []
for i in range(len(image_files)):
    img = Image.open(image_files[i])
    tgt = Image.open(target_files[i])
    ovr = Image.open(overlay_files[i])
    prd = Image.open(pred_files[i])

    min_width = min(img.width, ovr.width, prd.width, tgt.width)
    min_height = min(img.height, ovr.height, prd.height, tgt.height)

    img = img.resize((min_width, min_height))
    tgt = tgt.resize((min_width, min_height))
    ovr = ovr.resize((min_width, min_height))
    prd = prd.resize((min_width, min_height))

    # create a new image with four windows
    new_image = Image.new("RGB", (img.width*2, img.height*2))

    # paste the four images onto the new image
    new_image.paste(img, (0, 0))
    new_image.paste(tgt, (img.width, 0))
    new_image.paste(ovr, (0, img.height))
    new_image.paste(prd, (img.width, img.height))

    # add text to the image
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype("arial.ttf", size=20)
    draw.text((10, 10), f"Iteration {i:0{len(str(len(image_files)))}d}", font=font, fill=(0, 0, 0))

    gif.append(new_image)

imageio.mimsave('animation_0422_v2_2.gif', gif)
