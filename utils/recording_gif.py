import imageio
import os

def record_gif():
    image_folder = 'images'  
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  
    image_files = [imageio.imread(os.path.join(image_folder, img)) for img in images]
    imageio.mimsave('pickup_sponge.gif', image_files, duration=1000/60) 