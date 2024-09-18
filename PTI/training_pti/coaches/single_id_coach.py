import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training_pti.coaches.base_coach import BaseCoach_custom, BaseCoach_custom_bg
from utils.log_utils import log_images_from_w
from torchvision.transforms import transforms
import PIL.Image as Image
tf_function = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

class SingleIDCoach_custom(BaseCoach_custom):

    def __init__(self, img_path, G2, pose, tun_iter= 1000, init_w=None, isbg=False, isbg_img_path=None, fg_bg=False, mask=None):
        super().__init__(img_path, G2, pose)
        
        self.isbg = isbg
        self.isbg_img_path = isbg_img_path
        
        self.fg_bg = fg_bg
        self.mask = mask
        
        self.img_path = img_path
        self.G = G2
        self.pose=pose
        self.tun_iter = tun_iter
        if init_w is not None:
            self.init_w = init_w

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)
        use_ball_holder = True

        image = tf_function(Image.open(self.img_path).convert("RGB"))
        fname = self.img_path.split("/")[-1]
        image_name = fname[0]

        self.restart_training(self.G)

        '''
        
        if self.image_counter >= hyperparameters.max_images_to_invert:
            break
            
        '''

        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None
        
        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)
            
        elif not hyperparameters.use_last_w_pivots or w_pivot is None:
            
            if self.isbg == True:
                #self.isbg_img_path
                image_white = tf_function(Image.open(self.isbg_img_path).convert("RGB"))
                fname_white = self.isbg_img_path.split("/")[-1]
                image_name_white = fname_white[0]
                w_pivot_white, _ = self.calc_inversions(image_white, image_name_white) # , bg_train=False
                w_pivot, bg_latents = self.calc_inversions(image, image_name, initial_w=w_pivot_white)
            
            
            elif self.fg_bg == True:
                w_pivot, _ = self.calc_inversions(image, image_name, mask=self.mask, fg_only=True)
                _, bg_latents = self.calc_inversions(image, image_name, initial_w=w_pivot, mask= 1.-self.mask, bg_only=True)
            
            else:
                w_pivot, bg_latents = self.calc_inversions(image, image_name)

        # w_pivot = w_pivot.detach().clone().to(global_config.device)
        w_pivot = w_pivot.to(global_config.device)

        torch.save(w_pivot, f'{embedding_dir}/0.pt')
        log_images_counter = 0
        real_images_batch = image.to(global_config.device)

        synthesis_kwargs = {}
        synthesis_kwargs['render_option'] = "freeze_bg"
        synthesis_kwargs['latent_codes'] = bg_latents
        synthesis_kwargs['camera_RT'] = self.pose
        
        #for i in tqdm(range(hyperparameters.max_pti_steps)):
        for i in tqdm(range(self.tun_iter)):
            generated_images = self.forward(w_pivot, synthesis_kwargs)
            loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                            self.G, use_ball_holder, w_pivot)

            self.optimizer.zero_grad()

            if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                break
            
            loss.backward()
            self.optimizer.step()

            use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

            if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                log_images_from_w([w_pivot], self.G, [image_name])

            global_config.training_step += 1
            log_images_counter += 1

        self.image_counter += 1

        torch.save(self.G,
                    f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
        
        import shutil
        embeddings_folder = os.path.join('embeddings')
        if os.path.exists(embeddings_folder):
            shutil.rmtree(embeddings_folder)
            print(f"Deleted folder: {embeddings_folder}")
            
        embeddings_folder = os.path.join('checkpoints')
        if os.path.exists(embeddings_folder):
            shutil.rmtree(embeddings_folder)
            print(f"Deleted folder: {embeddings_folder}")    
            
        
        return self.G, w_pivot, bg_latents


class SingleIDCoach_custom_bg(BaseCoach_custom_bg):

    def __init__(self, img_path, G2, pose, tun_iter= 1000, init_w=None):
        super().__init__(img_path, G2, pose)
        self.img_path = img_path
        self.G = G2
        self.pose=pose
        self.tun_iter = tun_iter
        if init_w is not None:
            self.init_w = init_w

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)
        
        use_ball_holder = True

        image = tf_function(Image.open(self.img_path).convert("RGB"))
        fname = self.img_path.split("/")[-1]
        image_name = fname[0]

        self.restart_training(self.G)

        '''if self.image_counter >= hyperparameters.max_images_to_invert:
            break'''

        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        elif not hyperparameters.use_last_w_pivots or w_pivot is None:
            bg_latents = self.calc_inversions_bg(image, image_name, initial_w=self.init_w)

        return bg_latents


