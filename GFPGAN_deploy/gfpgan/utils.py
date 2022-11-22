import cv2
import os
import torch
# from basicsr.utils import img2tensor, tensor2img
# from basicsr.utils.download_util import load_file_from_url
# from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
from gfpgan.archs.gfpganv1_arch import GFPGANv1
# from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)



class GFPGANer:
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(
        self,
        model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    ):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize the GFP-GAN
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(
                out_size=128,
                num_style_feat=256,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=0.5,
                sft_half=True,
            )
        else:
            self.gfpgan = GFPGANv1(
                out_size=256,
                num_style_feat=256,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=0.5,
                sft_half=True,
            )
        # initialize face helper
        # self.face_helper = FaceRestoreHelper(
        #     upscale,
        #     face_size=512,
        #     crop_ratio=(1, 1),
        #     det_model="retinaface_resnet50",
        #     save_ext="png",
        #     device=self.device,
        # )
        print(self.gfpgan)
        # if model_path.startswith("https://"):
        #     model_path = load_file_from_url(
        #         url=model_path,
        #         model_dir=os.path.join(ROOT_DIR, "gfpgan/weights"),
        #         progress=True,
        #         file_name=None,
        #     )
        loadnet = torch.load(model_path)
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"

        # state_dict = dict()
        # for k, v in loadnet[keyname].items():
        #     if 'modulated_conv.weight' in k:
        #         state_dict[k] = v

        #         k = k.replace('modulated_conv.weight', 'modulated_conv.conv2d.weight')
        #         print(k, v.shape)
        #         b, c_out, c_in, m, m = v.shape
        #         v = v.view(b * c_out, c_in, m, m)
        #     state_dict[k] = v
        # self.gfpgan.load_state_dict(state_dict, strict=False)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        # self.face_helper.clean_all()

        # if has_aligned:  # the inputs are already aligned
        #     imgs = []
        #     for i in img:
        #         imgs.append(cv2.resize(i, (128, 128)))
        #     self.face_helper.cropped_faces = imgs
        # else:
        #     self.face_helper.read_image(img)
        #     # get face landmarks for each face
        #     self.face_helper.get_face_landmarks_5(
        #         only_center_face=only_center_face, eye_dist_threshold=5
        #     )
        #     # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        #     # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
        #     # align and warp each face
        #     self.face_helper.align_warp_face()

        # # face restoration

        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # #while True:
        # bs = []
        # starter.record()
        # for cropped_face in self.face_helper.cropped_faces:

        #     # prepare data
        #     cropped_face_t = img2tensor(
        #         cropped_face / 255.0, bgr2rgb=True, float32=True
        #     )
        #     normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        #     bs.append(cropped_face_t.to(self.device))

        # prepare data
        imgs = []
        for i in img:
            imgs.append(cv2.resize(i, (256, 256)))
        bs = []
        for cropped_face in imgs:
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            bs.append(cropped_face_t.to(self.device))
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print("intput time")
        # print(curr_time)
        #cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

        # starter.record()
        torch.onnx.export(
            self.gfpgan,
            torch.stack(bs),
            'model.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: '-1'},
                          'output': {0: '-1'}}
        )
        print('Converted to ONNX successfully')
        # outputs = self.gfpgan(torch.stack(bs), return_rgb=False)[0]
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print("inference time")
        # print(curr_time)
        # convert to image

        # starter.record()
        # for output in outputs:
        #     restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        #     restored_face = restored_face.astype("uint8")
        #     self.face_helper.add_restored_face(restored_face)
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print("output time")
        # print(curr_time)

        # if not has_aligned and paste_back:
        #     # upsample the background
        #     if self.bg_upsampler is not None:
        #         # Now only support RealESRGAN for upsampling background
        #         bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
        #     else:
        #         bg_img = None

        #     self.face_helper.get_inverse_affine(None)
        #     # paste each restored face to the input image
        #     restored_img = self.face_helper.paste_faces_to_input_image(
        #         upsample_img=bg_img
        #     )
        #     return (
        #         self.face_helper.cropped_faces,
        #         self.face_helper.restored_faces,
        #         restored_img,
        #     )
        # else:
        #     return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
