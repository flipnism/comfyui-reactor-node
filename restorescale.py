import os, glob, sys
import shutil
from modules.processing import StableDiffusionProcessingImg2Img
from scripts.reactor_faceswap import (
    get_models,
)
from scripts.reactor_logger import logger
from reactor_utils import (
    batch_tensor_to_pil,
    batched_pil_to_tensor,
    img2tensor,
    tensor2img,
    move_path,
)

import model_management
import torch
import comfy.utils
import numpy as np
import cv2

# import math
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper

# from facelib.detection.retinaface import retinaface
from torchvision.transforms.functional import normalize
from comfy_extras.chainner_models import model_loading
import folder_paths

models_dir = folder_paths.models_dir
REACTOR_MODELS_PATH = os.path.join(models_dir, "reactor")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")
if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)


def get_restorers():
    # basedir = os.path.abspath(os.path.dirname(__file__))
    # global MODELS_DIR
    # models_path = os.path.join(basedir, "models/facerestore_models/*")
    models_path = os.path.join(models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".pth")]
    return models


def get_model_names(get_models):
    models = get_models()
    names = ["none"]
    for x in models:
        names.append(os.path.basename(x))
    return names


def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


models_dir_old = os.path.join(os.path.dirname(__file__), "models")
old_dir_facerestore_models = os.path.join(models_dir_old, "facerestore_models")

dir_facerestore_models = os.path.join(models_dir, "facerestore_models")
os.makedirs(dir_facerestore_models, exist_ok=True)
folder_paths.folder_names_and_paths["facerestore_models"] = (
    [dir_facerestore_models],
    folder_paths.supported_pt_extensions,
)

if os.path.exists(old_dir_facerestore_models):
    move_path(old_dir_facerestore_models, dir_facerestore_models)
if os.path.exists(dir_facerestore_models) and os.path.exists(
    old_dir_facerestore_models
):
    shutil.rmtree(old_dir_facerestore_models)


def restore_face(
    result, face_helper, face_restore_model, facedetection, centerface_only
):
    if face_restore_model != "none":
        logger.status(f"Restoring with {face_restore_model}")
        # print(f"Restoring with {face_restore_model}")

        # model_path = os.path.join(os.path.dirname(__file__), "models", "facerestore_models", face_restore_model)
        model_path = folder_paths.get_full_path(
            "facerestore_models", face_restore_model
        )
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        facerestore_model = model_loading.load_state_dict(sd).eval()

        device = model_management.get_torch_device()
        facerestore_model.to(device)
        if face_helper is None:
            face_helper = FaceRestoreHelper(
                1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model=facedetection,
                save_ext="png",
                use_parse=True,
                device=device,
            )

        image_np = 255.0 * result.cpu().numpy()

        total_images = image_np.shape[0]
        out_images = np.ndarray(shape=image_np.shape)

        for i in range(total_images):
            cur_image_np = image_np[i, :, :, ::-1]

            original_resolution = cur_image_np.shape[0:2]

            if facerestore_model is None or face_helper is None:
                return result

            face_helper.clean_all()
            face_helper.read_image(cur_image_np)
            face_helper.get_face_landmarks_5(
                only_center_face=centerface_only, resize=640, eye_dist_threshold=5
            )
            face_helper.align_warp_face()

            restored_face = None
            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                cropped_face_t = img2tensor(
                    cropped_face / 255.0, bgr2rgb=True, float32=True
                )
                normalize(
                    cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                )
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = facerestore_model(cropped_face_t)[0]
                        restored_face = tensor2img(
                            output, rgb2bgr=True, min_max=(-1, 1)
                        )
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(
                        f"\tFailed inference for CodeFormer: {error}",
                        file=sys.stderr,
                    )
                    restored_face = tensor2img(
                        cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                    )

                restored_face = restored_face.astype("uint8")
                face_helper.add_restored_face(restored_face)

            face_helper.get_inverse_affine(None)

            restored_img = face_helper.paste_faces_to_input_image()
            restored_img = restored_img[:, :, ::-1]

            if original_resolution != restored_img.shape[0:2]:
                restored_img = cv2.resize(
                    restored_img,
                    (0, 0),
                    fx=original_resolution[1] / restored_img.shape[1],
                    fy=original_resolution[0] / restored_img.shape[0],
                    interpolation=cv2.INTER_LINEAR,
                )

            face_helper.clean_all()

            out_images[i] = restored_img

        restored_img_np = np.array(out_images).astype(np.float32) / 255.0
        restored_img_tensor = torch.from_numpy(restored_img_np)

        return restored_img_tensor

    else:
        return result


def upscale_image(upscale_model_name, image):
    model_path = folder_paths.get_full_path("upscale_models", upscale_model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    upscale_model = model_loading.load_state_dict(sd).eval()
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)
    free_memory = model_management.get_free_memory(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[3],
                in_img.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar,
            )
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return s


class RestoreAndScale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "facedetection": (
                    [
                        "retinaface_resnet50",
                        "retinaface_mobile0.25",
                        "YOLOv5l",
                        "YOLOv5n",
                    ],
                ),
                "face_restore_model": (get_model_names(get_restorers),),
                "centerface_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "all faces",
                        "label_on": "center face only",
                    },
                ),
                "upscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "No",
                        "label_on": "Yes",
                    },
                ),
                "upscale_model_name": (
                    folder_paths.get_filename_list("upscale_models"),
                ),
                "scale_restore": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "Restore > Scale",
                        "label_on": "Scale > Restore",
                    },
                ),
                # "coderformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.1}), # list(np.arange(0,1,0.1)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "ReActor"

    def __init__(self):
        self.face_helper = None

    def execute(
        self,
        input_image,
        facedetection,
        face_restore_model,
        centerface_only,
        upscale,
        upscale_model_name,
        scale_restore,
    ):
        scale_then_restore = scale_restore
        pil_images = batch_tensor_to_pil(input_image)
        p = StableDiffusionProcessingImg2Img(pil_images)
        result = batched_pil_to_tensor(p.init_images)
        if scale_then_restore:
            print("scale then restore")
            result = upscale_image(upscale_model_name, result) if upscale else result
            f = restore_face(
                result,
                self.face_helper,
                face_restore_model,
                facedetection,
                centerface_only,
            )
            return (f,)
        else:
            print("restore then scale")
            f = restore_face(
                result,
                self.face_helper,
                face_restore_model,
                facedetection,
                centerface_only,
            )
            result = upscale_image(upscale_model_name, f) if upscale else f
            return (result,)
