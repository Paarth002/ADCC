import argparse
import os.path as OSPATH
import torchvision.models as models
from metrics import adcc as ADCC
from image_utils import image_utils as IMUT
import torchcam
import torch
import torch.nn.functional as F
import os


def ScoreCAM_extracor(image, model, classidx=None):
    scam = torchcam.methods.ScoreCAM(model)
    with torch.no_grad():
        out = model(image)
        # print(image.shape)

    if classidx is None:
        classidx = out.max(1)[1].item()

    salmap = scam(class_idx=classidx, scores=out)
    return F.interpolate(
        salmap[0].unsqueeze(0),
        (224, 224),
        mode="bilinear",
        align_corners=False,
    )


def GradCAM_extracor(image, model, classidx=None):
    scam = torchcam.methods.GradCAM(model)
    with torch.no_grad():
        out = model(image)
        # print(image.shape)

    if classidx is None:
        classidx = out.max(1)[1].item()

    salmap = scam(class_idx=classidx, scores=out)
    return F.interpolate(
        salmap[0].unsqueeze(0),
        (224, 224),
        mode="bilinear",
        align_corners=False,
    )


def GradCAMpp_extracor(image, model, classidx=None):
    scam = torchcam.methods.GradCAMpp(model)
    with torch.no_grad():
        out = model(image)
        # print(image.shape)

    if classidx is None:
        classidx = out.max(1)[1].item()

    salmap = scam(class_idx=classidx, scores=out)
    return F.interpolate(
        salmap[0].unsqueeze(0),
        (224, 224),
        mode="bilinear",
        align_corners=False,
    )


def solve(opt):

    image = IMUT.image_to_tensors(opt)
    arch_name = opt.model.lower()

    arch_dict = {
        "resnet18": models.resnet18(pretrained=True).eval(),
        "resnet50": models.resnet50(pretrained=True).eval(),
        "vgg16": models.vgg16(pretrained=True).eval(),
    }
    arch = arch_dict[arch_name]

    cam_type = opt.cam
    if cam_type == "scorecam":
        saliency_map = ScoreCAM_extracor(image, arch)
    elif cam_type == "gradcam":
        saliency_map = GradCAM_extracor(image, arch)
    elif cam_type == "gradcampp":
        saliency_map = GradCAMpp_extracor(image, arch)

    explanation_map = image * saliency_map
    return ADCC.ADCC(
        image,
        saliency_map,
        explanation_map,
        arch,
        attr_method=ScoreCAM_extracor,
        debug=True,
    )


def main(opt):
    adcc_avg, avgdrop_avg, coh_avg, com_avg = 0, 0, 0, 0
    dir_path = opt.image_dir
    labels = ["normal", "pneumonia"]
    # img_dir = ["orig1\\", "orig2\\"]
    img_dir = "orig1"
    print("Model:", opt.model)
    print("Image_dir:", img_dir)
    cnt = 0
    for label in labels:
        for img in os.listdir(OSPATH.join(OSPATH.join(dir_path, label), img_dir)):
            img_path = OSPATH.join(
                OSPATH.join(OSPATH.join(dir_path, label), img_dir), img
            )
            opt.image = img_path
            # print(img_path)
            # print(solve(opt))
            adcc, avgdrop, coh, com = solve(opt)
            adcc_avg += adcc
            avgdrop_avg += avgdrop
            coh_avg += coh
            com_avg += com
            cnt += 1
            print(cnt)

    return adcc_avg / cnt, avgdrop_avg / cnt, coh_avg / cnt, com_avg / cnt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--cam", type=str, default="scorecam")

    opt = parser.parse_args()
    # opt.image = "D:\\BTP\\ped_gan_orig\\gan_512\\test\\normal\\gan_normal_1.png"

    # assert OSPATH.exists(opt.image), "Image not found"

    # adcc = main(opt)
    # print(adcc)
    adcc, avgdrop, coh, com = main(opt)

    print("adcc", adcc)
    print("avgdrop", avgdrop)
    print("coh", coh)
    print("com", com)
    # print("A", A)
    # print("B", B)
    print("finish")
