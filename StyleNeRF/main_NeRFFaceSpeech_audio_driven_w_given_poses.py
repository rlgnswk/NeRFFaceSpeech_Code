import os
from typing import List, Optional
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from os.path import dirname, join, basename, isfile
import torch
from audio2NeRF_utils import *
from tqdm import tqdm

'''



'''

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--test_data', default=".", type=str)
@click.option('--test_img', default=".", type=str)
@click.option('--motion_guide_img_folder', default=".", type=str)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    test_img=".",
    test_data=".",
    motion_guide_img_folder='.'
):
    
    set_seed(0)
    batch_size = 1
    device = torch.device('cuda')
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(f"{outdir}/output_NeRFFaceSpeech.mp4"):
        
        ##-- Network Loading
        if os.path.isdir(network_pkl):
            network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
        print('Loading networks from "%s"...' % network_pkl)

        with dnnlib.util.open_url(network_pkl) as f:
            network = legacy.load_network_pkl(f)
            G = network['G_ema'].to(device) # type: ignore
            D = network['D'].to(device)
        
        # Labels Setting
        label = torch.zeros([1, G.c_dim], device=device)
        class_idx = None
        
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        # Overriding the function
        from training.networks import Generator
        # from training.stylenerf import Discriminator
        from torch_utils import misc
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
        G2.eval()
        
        ##-- Sampling from Z
        image_path = test_img 
        input_img = PIL.Image.open(os.path.join(image_path)).convert('RGB')
        input_img_224= input_img.resize((224, 224))
        input_img.save(f'{outdir}/input_img.png')
        input_img_224_proc = torch.tensor(np.array(input_img_224)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        
        ## -- Load BiSeNet
        from audio2NeRF_segNet import BiSeNet, seg_mean, seg_std
        segNet = BiSeNet(n_classes=16).cuda()
        segNet.load_state_dict(torch.load("pretrained_networks/seg.pth"))
        for param in segNet.parameters():
            param.requires_grad = False
        segNet.eval()
        
        ## -- Load Deep3DModel for 3DMM Estimnation
        Deep3Dmodel = load_Deep3Dmodel()
        Deep3Dmodel.device = device
        Deep3Dmodel.eval()
        Deep3Dmodel.net_recon.to(device)
        Deep3Dmodel.facemodel.to(device)
        
        #Load the Input Image
        #image_path = test_img 
        input_img = PIL.Image.open(image_path).convert('RGB')
        input_img_224= input_img.resize((224, 224))
        #input_img.save(image_path)
        input_img_224_proc = torch.tensor(np.array(input_img_224)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        # Get Input Image Pose
        with torch.no_grad():
            frame_full_coeff = Deep3Dmodel.net_recon(input_img_224_proc)
            _, _, _, pred_rot_mat = render_3dmm(Deep3Dmodel, input_img_224_proc, frame_full_coeff, get_rot=True)
            input_pose = get_front_matrix(device, rot=pred_rot_mat.cpu().numpy())
        
        if os.path.isfile(f"{outdir}/G_PTI.pt"):
            G_PTI = torch.load(f"{outdir}/G_PTI.pt")
            with torch.no_grad():
                G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
                misc.copy_params_and_buffers(G_PTI, G2, require_all=False)
            G2.eval()
            ws = torch.load(f"{outdir}/w_PTI.pt")
            bg_latents = torch.load(f"{outdir}/bg_PTI.pt")
        else:
            sys.path.insert(0,'PTI')
            from training_pti.coaches.single_id_coach import SingleIDCoach_custom
        
            G_PTI, ws, bg_latents = SingleIDCoach_custom(image_path, G2, input_pose, tun_iter= 2000).train()
            torch.save(G_PTI, f"{outdir}/G_PTI.pt")
            torch.save(ws, f"{outdir}/w_PTI.pt")
            torch.save(bg_latents, f"{outdir}/bg_PTI.pt")
            G2 = G_PTI
        
        
        synthesis_kwargs = {}
        synthesis_kwargs['render_option'] = "freeze_bg"
        synthesis_kwargs['latent_codes'] = bg_latents
        synthesis_kwargs['camera_RT'] = get_front_matrix(device)
        
        ### calculating ray points for deformation
        
        with torch.no_grad():
                        
            output = G2(styles=ws, truncation_psi=truncation_psi, noise_mode=noise_mode, **synthesis_kwargs)
            
            #####################################################Frontal View
            
            ref_di = output['di']
            ref_di_fine = output['di_fine']
            ref_p_i = output['p_i']
            ref_p_f = output['p_f']

            img_tesnor = output['img']
            img_tesnor_256 = torch.nn.functional.interpolate(img_tesnor, size=(256,256), mode='bicubic')
            img = proc_img(img_tesnor)
            PIL.Image.fromarray(img[0], 'RGB').save(f'{outdir}/output_from_w_pivot_w_G_PTI.png')
            #exit(0)
            ws_2d = ws[:,10:,:].clone().detach()
            ws_nerf = ws[:,:10,:].clone().detach()

            ##########################################################################################
            #Fix the torso from ray deformation
            
            i_frame_224 = torch.nn.functional.interpolate(img_tesnor, size=(224,224), mode='bicubic')
            PIL.Image.fromarray(proc_img(i_frame_224[:1])[0], 'RGB').save(f'{outdir}/img_tesnor_224_resize_prc.png')
            
            i_frame_224_norm = (i_frame_224 * 127.5 + 128).clamp(0, 255) / 255
            
            full_coeff = Deep3Dmodel.net_recon(i_frame_224_norm)

            ref_full_coeff = full_coeff.clone()
            coeffs = split_coeff(full_coeff)
            ref_exp_coeffs = coeffs['exp']
            
            i_frame_sample = img_tesnor
            
            img_512 = torch.nn.functional.interpolate(i_frame_sample, size=(512,512), mode='bicubic')
            im_seg_norm = (img_512.clamp(0, 1) - seg_mean) / seg_std
        
            down_seg, _, _ = segNet(im_seg_norm)
            seg_target = torch.argmax(down_seg, dim=1).long()
        
            torso_mask = ((seg_target == 14)|(seg_target == 15)|(seg_target == 10)).unsqueeze(1).float() ## neck and shouder (seg_target == 10) ## hair? 
            #PIL.Image.fromarray(proc_img2(torso_mask.repeat(1,3,1,1).cpu())[0], 'RGB').save(f'{outdir}/torso_mask.png')  
            coords = torch.nonzero(torso_mask[0][0] == 1)
            coords = coords[:, [1, 0]] #* -1.
            
            ############################################################################################
            audio2exp_model = load_audio2exp_model(device)

            print("########################### audio2exp_model Loading Done #################################")

            img_tesnor_256 = img_tesnor_256.repeat(batch_size,1,1,1)
            
            ######################
            img_tesnor_224 = torch.nn.functional.interpolate(img_tesnor, size=(224,224), mode='bicubic')
            img_tesnor_224_prc = proc_img(img_tesnor_224)
            
            PIL.Image.fromarray(img_tesnor_224_prc[0], 'RGB').save(f'{outdir}/img_tesnor_224_resize_prc.png')
            
            from PIL import Image
            im = Image.open(f'{outdir}/img_tesnor_224_resize_prc.png').convert('RGB')
            im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            fitted_coeffs_nofit = Deep3Dmodel.net_recon(im)
            
            coeffs_nofit = split_coeff(fitted_coeffs_nofit)
            ref_exp_coeffs_nofit = coeffs_nofit['exp']
            ref_full_coeff_nofit = fitted_coeffs_nofit

        
        ## 3D fitting
        fitted_coeffs = fit_3dmm(outdir)
        
        with torch.no_grad():
            _, _, proj_2d_vertex, _ = render_3dmm(Deep3Dmodel, img_tesnor_224_prc, fitted_coeffs, outdir, save_option=True, prefix="fitted_img", get_lm=True)
            
            ref_scaled_proj_2d_vertex =  scale_and_shift_coordinates(proj_2d_vertex[:, ::4,:])
            
            ref_torso = scale_and_shift_coordinates(coords[::4,:], in_min=0, in_max=512).unsqueeze(0)
            ref_torso[:,:, 1] = ref_torso[:,:, 1] * -1 
                
            ref_scaled_proj_2d_vertex = torch.cat((ref_scaled_proj_2d_vertex, ref_torso), dim=1)
            
            distances = torch.cdist(ref_scaled_proj_2d_vertex, ref_p_i[..., 1:]) # without z
            nearest_indices = torch.argmin(distances, dim=1) # tensor([[28429, 28429, 28429,  ..., 34913, 34913, 34913]], device='cuda:0')
            
            distances_f = torch.cdist(ref_scaled_proj_2d_vertex, ref_p_f[..., 1:]) # without z
            nearest_indices_f = torch.argmin(distances_f, dim=1) # tensor([[28429, 28429, 28429,  ..., 34913, 34913, 34913]], device='cuda:0')
            
            #import ipdb; ipdb.set_trace
            coeffs = split_coeff(fitted_coeffs)
            ref_exp_coeffs = coeffs['exp']
            ref_full_coeff = fitted_coeffs
            
            num_epoch = 0

            outdir_valid_final = os.path.join(outdir, f"epoch_{num_epoch}_final")
            os.makedirs(outdir_valid_final, exist_ok=True)
            
            count = 0
            #####################################
            # LipaintNet ##################
            from audio2NeRF_network import transfer_decoder
            exp_transfer_decoder = transfer_decoder().to(device)
            
            PATH = "pretrained_networks/LipaintNet.pt"
            checkpoint = torch.load(PATH)
            exp_transfer_decoder.load_state_dict(checkpoint['model_state_dict'])
            
            val_ratio = torch.zeros((1,)).to(device)
            frame_mel = audio_mel_load_sadtalker(test_data, device)

            synthesis_kwargs['di'] = ref_di
            synthesis_kwargs['di_fine'] = ref_di_fine
            synthesis_kwargs['nearest_indices'] = nearest_indices
            synthesis_kwargs['nearest_indices_f'] = nearest_indices_f
            
            #################################***********************************###########################################
            #expression scaling / Temporal Smoothing 
            expression_scale = 1.5
            smoother = TemporalSmoothing(buffer_size=7)        

            exp_coeff_pred_save = []
            
            root_value = ref_exp_coeffs[:1].clone().detach()
            avg_weight = 0.5
            for i in range(frame_mel.size(1)):
                #import ipdb; ipdb.set_trace()
                gt_exp_param_eval = audio2exp_model(frame_mel[:,i:i+1,:,:], ref_exp_coeffs[:1], val_ratio)
                
                exp_coeff_pred_save += [gt_exp_param_eval]
                ref_exp_coeffs[:1] = root_value * avg_weight + gt_exp_param_eval.squeeze(-1) * (1-avg_weight)
                
            out =  torch.cat(exp_coeff_pred_save, axis=0) * expression_scale
        
        #for i in range(frame_mel.size(1)):
        import natsort
        imglist = natsort.natsorted(os.listdir(motion_guide_img_folder))
        
        pred_rot_prev = input_pose
        #for i in range(len(imglist)):  
        for i in tqdm(range(frame_mel.size(1)), desc="Processing frames"):
            with torch.no_grad():
                try:
                    frame = Image.open(os.path.join(motion_guide_img_folder, imglist[i])).convert('RGB')
                    frame = frame.resize((224, 224))
                    #frame.save(f'{outdir_valid_video}/i_frame_{i}.png')
                    frame = torch.tensor(np.array(frame)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                    frame_full_coeff = Deep3Dmodel.net_recon(frame)
                    
                    _, _, _, pred_rot_mat = render_3dmm(Deep3Dmodel, img_tesnor_224_prc, frame_full_coeff, get_rot=True)
                    pred_rot = get_front_matrix(device, rot=pred_rot_mat.cpu().numpy())
                    pred_rot_prev = pred_rot
                    
                    #frame_full_coeff[:, :80 ] = ref_full_coeff[:, :80 ]
                    #frame_full_coeff[:, 144:] = ref_full_coeff[:, 144:]
                    #_, _, face_proj_moved, _ = render_3dmm(Deep3Dmodel, img_tesnor_224_prc, frame_full_coeff, get_rot=True)
                    
                except:
                    print("pose undetected")
                    pred_rot = pred_rot_prev
                                    
            with torch.no_grad():
                
                w_frame_2d = ws_2d.repeat(1,1,1)
                #synthesis_kwargs['camera_RT'] = input_pose #get_front_matrix(device)#input_pose#get_front_matrix(device) # input_pose
                synthesis_kwargs['camera_RT'] = pred_rot #get_front_matrix(device)#input_pose#get_front_matrix(device) # input_pose

                frame_ref_exp_coeffs = out[i:i+1].squeeze(-1) #gt_exp_param_eval.squeeze(-1)  #
                
                ref_full_coeff[:, 80: 144] = frame_ref_exp_coeffs # replace the exp param which is corresponding to audio
                
                _, _, face_proj_moved, _ = render_3dmm(Deep3Dmodel, img_tesnor_224_prc, ref_full_coeff, get_lm=True)
                
                scaled_face_proj_moved = scale_and_shift_coordinates(face_proj_moved[:, ::4,:])
                
                scaled_face_proj_moved = torch.cat((scaled_face_proj_moved, ref_torso), dim=1)
                
                #gt_exp_param_eval_nofit = audio2exp_model(frame_mel[:,i:i+1,:,:], ref_exp_coeffs_nofit[:1], val_ratio)
                #frame_ref_exp_coeffs_nofit = gt_exp_param_eval_nofit.squeeze(-1)
                #ref_full_coeff_nofit[:, 80: 144] = frame_ref_exp_coeffs_nofit

                synthesis_kwargs['ref_vertices'] = ref_scaled_proj_2d_vertex
                synthesis_kwargs['pred_vertices'] = scaled_face_proj_moved
                
                i_frame = G2(styles=torch.cat((ws_nerf, w_frame_2d), dim=1), truncation_psi=truncation_psi, noise_mode=noise_mode, **synthesis_kwargs)
                #i_frame_tensor = i_frame["img"].detach().clone()                
                del synthesis_kwargs['ref_vertices'] 
                del synthesis_kwargs['pred_vertices']
                
                # concat 5 frames and pop and push iteratively
                
                #frame_ref_exp_coeffs = split_coeff(i_frame_4_3dmm_coeff)['exp']
                synthesis_kwargs['latent_codes'] = synthesis_kwargs['latent_codes']
            
                w_exp_transfered = exp_transfer_decoder(ws[:,0,:], frame_ref_exp_coeffs) # exp_rand =>  torch.Size([b, 64]) ws[:,0,:] => (b ,512)
                w_exp_transfered = w_exp_transfered.unsqueeze(1).repeat(1,10,1)
                
                with torch.no_grad():
                    
                    inpaint_out = G2(styles=torch.cat((w_exp_transfered, ws[:,10:,:]), dim=1), truncation_psi=truncation_psi, noise_mode=noise_mode, **synthesis_kwargs)
                    inpaint_feat = inpaint_out['inpaint_feat']
                    i_frame = inpaint_out["img"]
                    
                    img_512 = torch.nn.functional.interpolate(i_frame, size=(512,512), mode='bicubic')
                    im_seg_norm = (img_512.clamp(0, 1) - seg_mean) / seg_std
                    
                    ##import ipdb; ipdb.set_trace()
                    down_seg, _, _ = segNet(im_seg_norm)
                    seg_target = torch.argmax(down_seg, dim=1).long()
                    
                    mouse_mask = 1.-((seg_target == 7)|(seg_target == 8)|(seg_target == 9)).unsqueeze(1).float()
                    mouse_mask = smoother.smooth(mouse_mask)

                synthesis_kwargs['inpaint_feat'] = inpaint_feat
                synthesis_kwargs['inpaint_mask'] =  mouse_mask  # process_mask(pred_mask, size=224)  # 
                synthesis_kwargs['interpol_size'] = 512
                
                #224
                synthesis_kwargs['Mask_method'] =  "bicubic"
                synthesis_kwargs['Feat_method'] =  "bicubic"
                synthesis_kwargs['ref_vertices'] = ref_scaled_proj_2d_vertex
                synthesis_kwargs['pred_vertices'] = scaled_face_proj_moved

                i_frame = G2(styles=torch.cat((ws_nerf, w_frame_2d), dim=1), truncation_psi=truncation_psi, noise_mode=noise_mode, **synthesis_kwargs)
                #img = torch.nn.functional.interpolate(i_frame["img"], size=(224,224), mode='bicubic') * process_mask(pred_mask, size=224)
                
                i_frame = i_frame["img"]
                    
                PIL.Image.fromarray(proc_img(i_frame)[0], 'RGB').save(f'{outdir_valid_final}/epoch_{num_epoch}_frame_{i}.png')
                #import ipdb; ipdb.set_trace()
                del synthesis_kwargs['inpaint_feat']
                del synthesis_kwargs['inpaint_mask']
                del synthesis_kwargs['interpol_size']
                del synthesis_kwargs['Mask_method']
                del synthesis_kwargs['Feat_method']
                
                del synthesis_kwargs['ref_vertices']
                del synthesis_kwargs['pred_vertices']
                
            count+=1
        
        creat_final_video(outdir_valid_final, test_data, outdir)
        
        
        
        
    else:
        print("The video already exist. Please check that.")
                
if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter    