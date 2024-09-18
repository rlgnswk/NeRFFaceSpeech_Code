import os
import re
from typing import List
import numpy as np
import PIL.Image
import torch
from os.path import dirname, join, basename, isfile
import torch
import numpy as np
from glob import glob
import os, random, cv2, argparse
import matplotlib.pyplot as plt
import sys

class TemporalSmoothing:
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.buffer = []

    def smooth(self, current_frame_mask):
        """
        Args:
        current_frame_mask (torch.Tensor): 현재 프레임의 마스크 (H, W) 또는 (C, H, W)
        
        Returns:
        torch.Tensor: 스무딩된 마스크 (H, W) 또는 (C, H, W)
        """
        # 현재 프레임의 마스크를 버퍼에 추가
        self.buffer.append(current_frame_mask)
        
        # 버퍼 크기 유지
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # 버퍼에 있는 모든 마스크의 평균을 계산
        smoothed_mask = torch.mean(torch.stack(self.buffer), dim=0)

        return smoothed_mask

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def creat_final_video(outdir_valid_final, test_data, outdir, watermark=True):
    import glob
    import natsort
    import shutil
    from imwatermark import WatermarkEncoder
    encoder = WatermarkEncoder()
    wm = 'fake'
    encoder.set_watermark('bytes', wm.encode('utf-8'))
    print("video processing .. ")
    img_array = []
    for filename in natsort.natsorted(glob.glob(f'{outdir_valid_final}/*.png')):
        #print(filename)
        
        try:
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width,height)
            if watermark:
                img = encoder.encode(img, 'dwtDct')
                cv2.putText(img, "fake video", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
            
            img_array.append(img)
        except:
            pass
            
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{outdir}/out_refine.mp4', fourcc , 25, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()    

    import subprocess
    #ours_path = "/source/gihoon/Dataset_FFHQ_SYH_sorted/out_origin_PTI"
    cmd = f"ffmpeg -y -i {test_data} -i {outdir}/out_refine.mp4 -c:v copy -c:a aac {outdir}/output_NeRFFaceSpeech.mp4"
    subprocess.call(cmd, shell=True)
    
    epoch_folder = os.path.join(outdir, 'epoch_0_final')
    if os.path.exists(epoch_folder):
        shutil.rmtree(epoch_folder)
        print(f"Deleted folder: {epoch_folder}")
    
    out_refine_path = os.path.join(outdir, 'out_refine.mp4')
    if os.path.exists(out_refine_path):
        os.remove(out_refine_path)
        print(f"Deleted file: {out_refine_path}")
    #import ipdb;ipdb.set_trace()

def creat_final_video_motion(outdir_valid_final, test_data, outdir, watermark=True):
    import glob
    import natsort
    import shutil
    from imwatermark import WatermarkEncoder
    encoder = WatermarkEncoder()
    wm = 'fake'
    encoder.set_watermark('bytes', wm.encode('utf-8'))
    print("video processing .. ")
    img_array = []
    for filename in natsort.natsorted(glob.glob(f'{outdir_valid_final}/*.png')):
        #print(filename)
        
        try:
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width,height)
            if watermark:
                img = encoder.encode(img, 'dwtDct')
                cv2.putText(img, "fake video", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
            
            img_array.append(img)
        except:
            pass
            
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{outdir}/output_NeRFFaceSpeech.mp4', fourcc , 25, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()    
    
    epoch_folder = os.path.join(outdir, 'epoch_0_final')
    if os.path.exists(epoch_folder):
        shutil.rmtree(epoch_folder)
        print(f"Deleted folder: {epoch_folder}")
    
    out_refine_path = os.path.join(outdir, 'out_refine.mp4')
    if os.path.exists(out_refine_path):
        os.remove(out_refine_path)
        print(f"Deleted file: {out_refine_path}")
    #import ipdb;ipdb.set_trace()

def load_audio2exp_model(device):
    
    sys.path.insert(0,'SadTalker')
    #from src.test_audio2coeff import Audio2Coeff  
    from src.face3d.models import networks
    from src.audio2exp_models.networks import SimpleWrapperV2
    #from src.audio2exp_models.audio2exp import Audio2Exp 
    audio2exp_model = SimpleWrapperV2()
    audio2exp_model = audio2exp_model.to(device)
    for param in audio2exp_model.parameters():
        audio2exp_model.requires_grad = False
    audio2exp_model.eval()
    
    import safetensors
    import safetensors.torch
    
    def load_x_from_safetensor(checkpoint, key):
        x_generator = {}
        for k,v in checkpoint.items():
            if key in k:
                x_generator[k.replace(key+'.', '')] = v
        return x_generator
    
    sadtalker_path = "pretrained_networks/sad_talker_pretrained/SadTalker_V0.0.2_256.safetensors"
    
    checkpoints = safetensors.torch.load_file(sadtalker_path)
    audio2exp_model.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
    return audio2exp_model

def fit_3dmm(outdir):    
    sys.path.insert(0,'3DMM-Fitting-Pytorch')
    #from src.test_audio2coeff import Audio2Coeff
    from fit_single_img_custom import fit
    from core.options import ImageFittingOptions
    import easydict
    TestOptions = easydict.EasyDict({
            "tar_size" : 224,#256,
            "padding_ratio" : 0.3,
            "recon_model" : "bfm09",
            "first_rf_iters" : 1000,
            "first_nrf_iters" : 500,
            "rest_rf_iters" :50,
            "rest_nrf_iters" : 30,
            "rf_lr": 1e-2,
            "nrf_lr" : 1e-2,
            "lm_loss_w" : 100,
            "rgb_loss_w" : 1.6,
            "id_reg_w" : 1e-3,
            "exp_reg_w" : 0.8e-3,
            "tex_reg_w" : 1.7e-6,
            "rot_reg_w" : 1,
            "trans_reg_w": 1,
            "tex_w" : 1,
            "cache_folder" : 'fitting_cache',
            "nframes_shape" : 16,
            "gpu" : 0
        })
    
    args = TestOptions
    args.img_path = f'{outdir}/img_tesnor_224_resize_prc.png'
    args.device = 'cuda:%d' % args.gpu
        
    if os.path.isfile(f"{outdir}/fitted_coeffs.pt"):
        fitted_coeffs = torch.load(f"{outdir}/fitted_coeffs.pt")
    else:
        
        fitted_coeffs = fit(args)
        torch.save(fitted_coeffs, f"{outdir}/fitted_coeffs.pt")
    return fitted_coeffs



def vis_seg(pred):
    num_labels = 16
    color = np.array([[0, 0, 0],  ## 0
                      [102, 204, 255],  ## 1
                      [255, 204, 255],  ## 2
                      [255, 255, 153],  ## 3
                      [255, 255, 153],  ## 4
                      [255, 255, 102],  ## 5
                      [51, 255, 51],  ## 6
                      [0, 153, 255],  ## 7 ## maybe inner mouse
                      [0, 255, 255],  ## 8
                      [0, 255, 255],  ## 9
                      [204, 102, 255],  ## 10
                      [0, 153, 255],  ## 11
                      [0, 255, 153],  ## 12
                      [0, 51, 0],
                      [102, 153, 255],  ## 14
                      [255, 153, 102],  ## 15
                      ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]
    return rgb


def extract_non_overlapping_points(A, B, range_threshold):
    # A와 B의 점 간의 거리 계산
    distances = torch.cdist(A, B)
    
    # 겹치지 않는 포인트 추출
    non_overlapping_points = []
    for i in range(len(A)):
        min_distance = torch.min(distances[i])
        if min_distance > range_threshold:
            non_overlapping_points.append(A[i])
    
    return torch.stack(non_overlapping_points)


def point_plot(point, outdir, name):
    
    pred_lm = point.cpu().numpy()
    x = pred_lm[ :, 0]
    y = pred_lm[ :, 1]

    fig, ax = plt.subplots()
    #plt.imshow(np.flipud(img_tesnor_224_prc[0]))
    ax.scatter(x, y, s=1)  # scatter plot 생성, s는 점의 크기 설정
    
    # 그래프 저장
    plt.xlim(0,224)
    plt.ylim(0,224)
    plt.savefig(f'{outdir}/{name}.png')
    
def load_Deep3Dmodel():
    import easydict
    TestOptions = easydict.EasyDict({
            "name" : 'face_recon',
            "gpu_ids" : "0",
            "checkpoints_dir" : "pretrained_networks/Deep3DFaceRecon_pytorch/",
            "vis_batch_nums" : 1,
            "eval_batch_nums" : float('inf'),
            "use_ddp" :True,
            "ddp_port" : "12355",
            "display_per_batch": True,
            "add_image" : True,
            "world_size" : 1,
            "model" : "facerecon",
            "epoch" : 20,
            "verbose" : True,
            "suffix" : '',
            "phase" : 'test',
            "dataset_mode": None,
            "img_folder" : 'examples',
            "isTrain" : False,
            "net_recon" : 'resnet50',
            "init_path" : 'checkpoints/init_model/resnet50-0676ba61.pth',
            "use_last_fc" : False,
            "bfm_folder" : 'pretrained_networks/BFM/',
            "bfm_model" : 'BFM_model_front.mat',
            "focal" : 1015.,
            "center" : 112.,
            "camera_d" : 10.,
            "z_near" : 5.,
            "z_far" : 15.,
            "use_opengl" : True
        })
    
    import sys
    sys.path.insert(0,'Deep3DFaceRecon_pytorch')
    #from options.test_options import TestOptions ## 이거는 아예 옮길 수 있을듯..?
    from Deep3Dmodels import create_model
    Deep3Dopt = TestOptions
    Deep3Dmodel = create_model(Deep3Dopt)
    Deep3Dmodel.setup(Deep3Dopt)
    
    return Deep3Dmodel

def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import math as m

def Rx(theta):
    return np.matrix([[ 1, 0           , 0           ],
                    [ 0, m.cos(theta),-m.sin(theta)],
                    [ 0, m.sin(theta), m.cos(theta)]])

def Ry(theta):
    return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                    [ 0           , 1, 0           ],
                    [-m.sin(theta), 0, m.cos(theta)]])
    
def Rz(theta):
    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                    [ m.sin(theta), m.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])
    
def get_front_matrix(device, rot=None):
    
    import math as m
    phi = m.pi/2
    eye = -np.eye(3,3)
    eye[0,0] = -eye[0,0]
    origin = eye@Rz(-phi)@Rx(-phi)
    
    if rot is None:
        rot_mat_2 = origin
    else:
        #rot_mat_2 = rot@eye@Rz(-phi)@Rx(-phi)
        #rot_mat_2 = np.transpose(rot[0])@origin
        rot_mat_2 = origin @ rot[0]
    
    #import ipdb;ipdb.set_trace()    
    temp = np.eye(4,4)
    temp[:3,:3] = rot_mat_2
    temp[:3,3:] = temp[:3,2:3]
    temp = torch.from_numpy(temp).type(torch.float32).to(device).unsqueeze(0)
    cam_mat = temp.repeat(1,1,1)
    
    return cam_mat

def proc_img(img): 
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

def proc_img2(img):
    return (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

def render_3dmm(Deep3Dmodel, img_tesnor_224_prc, full_coeff, outdir=None, save_option=False, prefix=None, get_rot=False, get_lm=False):

    if get_rot is False:
        pred_vertex, pred_tex, pred_color, pred_lm, face_proj = \
            Deep3Dmodel.facemodel.compute_for_render(full_coeff, get_rot=get_rot)
    else:
        pred_vertex, pred_tex, pred_color, pred_lm, face_proj, pre_rot_mat = \
            Deep3Dmodel.facemodel.compute_for_render(full_coeff, get_rot=get_rot)
            
    pred_mask, what, pred_face = Deep3Dmodel.renderer(
        pred_vertex, Deep3Dmodel.facemodel.face_buf, feat=pred_color)
    if get_lm is True:
        return pred_face, pred_mask, face_proj, pred_lm
    
    if save_option is False:
        if get_rot is False:
            return pred_face, pred_mask, face_proj
        else:
            return pred_face, pred_mask, face_proj, pre_rot_mat
    else:
        img_pred_face = proc_img2(pred_face)
        PIL.Image.fromarray(img_pred_face[0], 'RGB').save(f'{outdir}/{prefix}_3dmm.png')
        
        output_vis = pred_face * pred_mask 
        img_output_vis = proc_img2(output_vis)
        
        pred_mask_numpy = pred_mask.detach().repeat(1,3,1,1).cpu().permute(0, 2, 3, 1).numpy()
        img_output_other = np.where(pred_mask_numpy == 0, img_tesnor_224_prc, 0)
        PIL.Image.fromarray(img_output_other[0], 'RGB').save(f'{outdir}/{prefix}_3dmm_other.png')
        img_output_vis = img_output_other + img_output_vis
        PIL.Image.fromarray(img_output_vis[0], 'RGB').save(f'{outdir}/{prefix}_3dmm_merge.png')
        return pred_face, pred_mask, face_proj
 
def scale_and_shift_coordinates(coords, in_min=0, in_max=224, out_min=-0.10, out_max=0.10):

    # 입력 범위에서의 비율 계산
    ratio = (coords - in_min) / (in_max - in_min)
    
    # 출력 범위에서의 좌표 계산
    scaled_coords = out_min + (out_max - out_min) * ratio
    
    return scaled_coords



def audio_mel_load_sadtalker(test_data, device):
    
    import sys
    sys.path.append('SadTalker/src')
    from generate_batch import get_data_mel
    
    return get_data_mel(test_data, device)



def audio_mel_load(test_data, device):
    
    import sys
    sys.path.append('Wav2Lip')
    import audio
    from hparams import hparams, get_image_list

    valid_data_path = test_data
    
    valid_wav = audio.load_wav(valid_data_path, hparams.sample_rate) # sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    fps = 25
    bit_per_frames = 16000 / fps
    val_num_frames = int(len(valid_wav) / bit_per_frames)
    
    valid_orig_mel = audio.melspectrogram(valid_wav).T
    syncnet_mel_step_size = 16
    
    valid_mel_input = torch.FloatTensor(valid_orig_mel.T).unsqueeze(0).unsqueeze(0).to(device) # torch.Size([1, 1, 80, 292])
    
    spec = valid_mel_input.clone()
    for i in range(val_num_frames):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx)) # [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> 
        seq = [ min(max(item, 0), valid_mel_input.shape[-1]-1) for item in seq ] # [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 양끝 프레임을 padding하는 방식으로 원하는 프레임만큼 만듦
        m = spec[..., seq]
        if i == 0:
            frame_mel = m
        else:
            frame_mel = torch.cat((frame_mel, m), dim=1)
            
    return frame_mel

def audio_mel_load_mut(test_data, device):
    
    import sys
    sys.path.append('Wav2Lip')
    import audio
    from hparams import hparams, get_image_list

    valid_data_path = test_data
    
    valid_wav = audio.load_wav(valid_data_path, hparams.sample_rate) # sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    
    # waveform 시각화
    waveform = valid_wav
    time_axis = torch.linspace(0, len(waveform) / 16000, steps=len(waveform))
    
    #save_wavenet_wav(wav, path, sr)
    import torchaudio
    # 변환된 음성 저장

    new_wav = np.where(waveform > 0.01, waveform, 0)
    audio.save_wav(new_wav, "waveform.wav", 16000)

    magnitude = abs(valid_wav)
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, magnitude.T)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform Visualization")
    plt.savefig("magnitude.png")
    import ipdb; ipdb.set_trace()

    magnitude = abs(valid_wav)
    energy_threshold = 0.01
    valid_wav = np.where(magnitude > energy_threshold, valid_wav, 0.0)

    fps = 25
    bit_per_frames = 16000 / fps
    val_num_frames = int(len(valid_wav) / bit_per_frames)
    
    valid_orig_mel = audio.melspectrogram(valid_wav).T
    syncnet_mel_step_size = 16
    
    valid_mel_input = torch.FloatTensor(valid_orig_mel.T).unsqueeze(0).unsqueeze(0).to(device) # torch.Size([1, 1, 80, 292])
    spec = valid_mel_input.clone()
    
    for i in range(val_num_frames):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx)) # [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> 
        seq = [ min(max(item, 0), valid_mel_input.shape[-1]-1) for item in seq ] # [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 양끝 프레임을 padding하는 방식으로 원하는 프레임만큼 만듦
        m = spec[..., seq]
        if i == 0:
            frame_mel = m
        else:
            frame_mel = torch.cat((frame_mel, m), dim=1)
            
    return frame_mel


def process_mask(pred_mask, size=224):
    
    pred_mask2 = pred_mask.clone()
    
    for i in range(size):    
        idx = 0
        while idx < size:
            # 0이면 1로 만들고 1이 나올때까지 나오면 역으로 또 다시
            #print( idx, i )
            if idx != size and pred_mask2[:,:,i,idx] == 0 :
                pred_mask2[:,:,i,idx] = 1
                idx += 1
            else:
                break
            
        idx = size -1
        while idx != 0 and  idx >= 0 :
            if pred_mask2[:,:,i,idx] == 0:
                pred_mask2[:,:,i,idx] = 1
                idx -= 1
            else:
                break  

    for i in range(size):    
        idx = 0
        while idx < size:
            # 0이면 1로 만들고 1이 나올때까지 나오면 역으로 또 다시
            
            if idx != size and pred_mask[:,:,idx,i] == 0 :
                pred_mask[:,:,idx,i] = 1
                idx += 1
            else:
                break
            
        idx = size -1
        while idx != 0 and idx >= 0:
            if pred_mask[:,:,idx,i] == 0:
                pred_mask[:,:,idx,i] = 1
                idx -= 1
            else:
                break  
            
    return torch.where((pred_mask != 0) | (pred_mask2 != 0), torch.ones_like(pred_mask), torch.zeros_like(pred_mask))   

