import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator


def inference(img, model_file):
    model = Generator().eval()
    model.cuda()
    model.load_state_dict(torch.load(model_file))

    # Convert CV2 image to PIL image
    img_pil = Image.fromarray(img)

    # Preprocess image
    preprocess = ToTensor()
    image = Variable(preprocess(img_pil), volatile=True).unsqueeze(0).cuda()

    start = time.perf_counter()  # or time.process_time() depending on your requirements
    with torch.no_grad():
        out = model(image)
    elapsed = (time.perf_counter() - start)  # or time.process_time() depending on your requirements
    print('cost' + str(elapsed) + 's')

    # Convert output tensor to CV2 image
    out_img = ToPILImage()(out[0].data.cpu())

    return out_img
