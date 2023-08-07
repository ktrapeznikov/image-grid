import numpy as np
from torchvision.utils import make_grid as make_grid_tv
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
import PIL
from PIL import Image
from math import sqrt,ceil
import glob
import tqdm.auto as tqdm
from PIL import ImageColor
import click


myFont = PIL.ImageFont.truetype('FreeMonoBold.ttf', 12)

def add_text_func(im,t,fill_color):
    d = PIL.ImageDraw.Draw(im)
    d.text((2, 2), t, font=myFont, fill=fill_color)
    return im




def make_grid(files, output_file = None, aspect = 9/16, size = 128, texts = None, text_color = "yellow"):
    N = len(files)
    nrows = ceil(sqrt(N/aspect))

    fill_color = ImageColor.getrgb(text_color)

    transform = Compose([lambda a: Image.open(a).convert("RGB"),  Resize(size,antialias=True), CenterCrop(size)])

    if isinstance(files,str):
        files = sorted(glob.glob(files))
        assert len(files) > 0


    if isinstance(files[0],Image.Image):
        #assuming already PIL images
        transform.transforms = transform.transforms[1:]

    elif isinstance(files[0],np.ndarray):
        transform.transforms[0] = lambda a: Image.fromarray(a).convert("RGB")

    elif isinstance(files[0],str):
        pass

    else:
        raise ValueError("files must be pattern, PIL images, numpy arrays, or filenames")


    

    ims = []

    if texts is None:
        texts = [None]*len(files)
    elif texts == "index":
        texts = [f"{i:03}" for i in range(len(files))]
    else:
        assert len(texts) == len(files)

    for f,t in tqdm.tqdm(zip(files,texts),total=len(files)):
        im = transform(f)
        if t is not None:
            im = add_text_func(im,t,fill_color)
        ims.append(to_tensor(im))
            
    out =  to_pil_image(make_grid_tv(ims,nrow=nrows)) 
    if output_file is not None:
        out.save(output_file)
    return out

def test():
    files = sorted(glob.glob("./test_images/*.png"))
    print(len(files))

    im = make_grid(files,texts="index")
    im.save("grid.jpg")

@click.command()
@click.argument("files")
@click.option("--aspect",default=9/16)
@click.option("--size",default=128)
@click.option("--texts",default=None)
@click.option("--text-color",default="yellow")
@click.option("--output-file",default=None)
def make_grid_cli(*args,**kwargs):
    make_grid(*args,**kwargs)


if __name__=="__main__":
    make_grid_cli()
    