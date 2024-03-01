from PIL import Image
import pprint
import os
import numpy as np
# Open the image form working directory


root = 'datasets/data'
root = os.path.expanduser(root)

targets_dir = os.path.join(root, 'new_targets_label', 'train')

for street in os.listdir(targets_dir):
    street_dir = os.path.join(targets_dir, street)
    if street_dir[-2:] not in ['04','05','06']:
        continue
    fnames = set(os.listdir(os.path.join(street_dir, 'car')))
    fnames = fnames.union(set(os.listdir(os.path.join(street_dir, 'traffic_sign'))))
    for file_name in fnames:
        print(file_name)
        if file_name[:3] != 'JHU':
            continue
        root, exe = os.path.splitext(file_name)
        traffic_sign_path = os.path.join(street_dir, 'traffic_sign', file_name)
        car_path = os.path.join(street_dir, 'car', file_name)

        if os.path.exists(traffic_sign_path):
            traffic_sign_img = Image.open(traffic_sign_path)
            traffic_sign_data = np.asarray(traffic_sign_img)
        else:
            traffic_sign_data = 255*np.ones((2160, 3840, 3))

        if os.path.exists(car_path):
            car_img = Image.open(car_path)
            car_data = np.asarray(car_img)
        else:
            car_data = 255*np.ones(traffic_sign_data.shape)

        # temp = np.zeros((car_data.shape[0], car_data.shape[1]))
        out_data = np.where(traffic_sign_data[:,:,0] < 128, 20, 0)
        out_data = np.where(car_data[:,:,2] < 128, 26, out_data)

        # out_data = np.where(temp == 1, yellow, np.where(temp > 1, blue, 0))
        # out_data[:,:,1] = np.where(traffic_sign_data[:,:,1] < 128, 220, 0)

        # out_data[:,:,2] = np.where(car_data[:,:,2] < 128, 142, 0)

        # out_data[:,:] = np.where(car_data[:,:][0] < 128)
        image = Image.fromarray(out_data.astype(np.uint8))
        image.save(os.path.join(street_dir, root + ".png"))

        # print(car_data.shape)
        # print(out_data.shape)

        
        
        # print(car_data)
        # exit()
        


# dictt = dict()
# for val in image.getdata():
#     if val not in dictt:
#         dictt[val] = 0
#     dictt[val] += 1
# print(len(dictt))
# pprint.pprint(dictt)
# print(image2.format)
# print(image2.size)
# print(image2.mode)