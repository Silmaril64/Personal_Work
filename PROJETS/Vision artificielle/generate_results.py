import cycle_gan as cg
import os, os.path
import imageio
import torch

MAX_IMAGES = 10

def save_images(base_images, images, path):
    images = torch.transpose(images, -1,1)
    images *= 255
    images = images.int()
    base_images = torch.transpose(base_images, -1,1)
    base_images *= 255
    base_images = base_images.int()
    for i in range(len(images)):
        im = images[i]
        b_im = base_images[i]
        print(im.shape)
        im = torch.clamp(im, 0, 255)
        b_im = torch.clamp(b_im, 0, 255)
        print(im.numpy(), path + str(i))
        imageio.imwrite(path + str(i) + ".jpg", im.numpy().astype('uint8'))
        imageio.imwrite(path + str(i) + "_base.jpg", b_im.numpy().astype('uint8'))


PATH = "./trans_A_to_B" 
trans_A_to_B = cg.TransformOtherDimension().float()
if os.path.exists(PATH):
    trans_A_to_B.load_state_dict(torch.load(PATH))

PATH = "./trans_B_to_A" 
trans_B_to_A = cg.TransformOtherDimension().float()
if os.path.exists(PATH):
    trans_B_to_A.load_state_dict(torch.load(PATH))


base_A = cg.construction_images_tensor('./trainA/', MAX_IMAGES).float()
base_B = cg.construction_images_tensor('./trainB/', len(base_A)).float()

new_images_B = trans_A_to_B(base_A)
new_images_A = trans_B_to_A(base_B)

save_images(base_B, new_images_A, "./creation_B/")
save_images(base_A, new_images_B, "./creation_A/")


