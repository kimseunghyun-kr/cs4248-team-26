import os
from copy import deepcopy as dc

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AdamW

import clip
from can.attack.simple_pgd import bafa_img_iccv
from can.attack.simple_pgd import perturb_bafa_txt_multi_all_noli, \
    perturb_bafa_txt_multi_test, perturb_bafa_txt_multi_ablation_lb_ls
from can.utils.tools import get_parameter_count
from clip import clip
from network.clip import evaluate_clip
from utils.txt_input import mk_prompt_mapping


class Collaboration:  # Collaborative
    def __init__(self, cfg, uo, train_loader_base, val_loader_base, test_loader_base, test_info=None):
        # base setting
        self.cfg, self.mode, self.uo,  = cfg, cfg.mode, dc(uo),   # cfg
        self.device, self.txt_learn_mode, = self.cfg.device, cfg.txt_learn_mode,
        self.best_txt_epoch , self.best_robust_acc, self.best_txt_model_state  = 0, -1, None
        # model setting
        txt_model = SetTarget(cfg, uo.base_model, 1, 'txt', cfg.txt_learn_mode)  # cfg.target_layer
        img_model = SetTarget(cfg, uo.base_model.visual, cfg.target_layer, 'img', cfg.learn_mode)
        self.txt_model = txt_model.to(self.device)
        self.bias_txt_model_freeze, self.bias_txt_model = dc(self.txt_model), dc(self.uo)
        self.img_model, self.bias_img_model = img_model.to(self.device), dc(img_model).to(self.device)

        # dataset setting
        self.train_loader, self.val_loader, self.test_loader  = train_loader_base, val_loader_base, test_loader_base
        self.len_t = len(val_loader_base.dataset)
        if test_info is not None:
            self.test_info = test_info

        # train setting
        self.init_set()
        self.use_debias_vl()

    def init_set(self):
        self.bias_sample_check, self.cls_num_list = True, [0, 1]
        self.data = self.cfg.embeddings_dir.split('/')[-1]
        self.result = {}  # Initialize result dictionary
        self.epo = 0  # Initialize epoch counter
        print(f'txt target parameter : {get_parameter_count(self.txt_model)}')
        print(f'img target parameter : {get_parameter_count(self.img_model)}')

    def exper_set(self, model_save=False, att_mode=None, learn_mode=None, txt_input=None, ):
        self.att_mode, self.learn_mode, self.model_save = att_mode, learn_mode, model_save
        self.l_scale, self.batch, self.txt_input = 100, 128, txt_input
        self.att_bnd, self.att_stp, self.att_itr, self.biasclip = 0.1, 1e-3, 20, True
        self.sub = torch.tensor(0, device=self.cfg.device)

    def use_debias_vl(self):
        from can.attack.debias_vl import debais_vl_s_c, debias_vl
        spurious_prompt, candidate_prompt, self.S, B, text = debais_vl_s_c(self.cfg)
        with torch.no_grad():
            candidate_embeddings = self.uo.get_embedding(candidate_prompt)
            spurious_embeddings = self.uo.get_embedding(spurious_prompt)
            self.P = debias_vl(spurious_embeddings, candidate_embeddings, self.S)

    def output(self, x, modal, mode):
        with torch.no_grad():
            if modal == 'txt':
                if mode == 'bias':
                    out = self.bias_txt_model.get_embedding(x)  # eval
                else:
                    self.txt_model.eval()
                    z = self.txt_model.get_feature(x)  # float16
                    out = self.txt_model(z)
            else:
                if mode == 'bias':
                    self.bias_img_model.eval()
                    out = self.bias_img_model.img_inference(x)
                else:
                    self.img_model.eval()
                    z = self.img_model.get_feature(x)
                    out = self.img_model(z)
        return out.half()


    def clip_img_feat(self, exp=True, data_t='waterbirds'):
        self.cls_text, self.bias_text, self.mix_text, self.mapping, self.test_text = mk_prompt_mapping(data_t)
        self.mix_len, self.bias_len = len(self.mix_text), len(self.bias_text)

        # For counteranimal, override with dynamic class names
        if data_t == 'counteranimal':
            if hasattr(self.train_loader.dataset, 'class_names'):
                class_names = self.train_loader.dataset.class_names
                self.cls_text = [f"A photo of a {name}." for name in class_names]
                print(f"Generated {len(self.cls_text)} text descriptions for counteranimal")

        # FairFace: special handling for stereotype-based debiasing
        # cls_text has 25 stereotypes, but dataset target is gender (2 classes)
        if data_t == 'fairface':
            n_classes = 2  # gender: male/female
            print(f"FairFace: Using {len(self.cls_text)} stereotype prompts, dataset has {n_classes} gender classes")
        else:
            # Handle targets: binary (2 classes) vs multi-class (45+ classes)
            n_classes = len(self.cls_text)

        if n_classes == 2:
            # Binary classification (waterbirds, celebA)
            self.val_y_s = torch.where(torch.tensor(self.val_loader.dataset.targets_all['spurious']).view(-1, 1) == 0,
                                       torch.tensor([1, -1]), torch.tensor([-1, 1]))
            self.val_y_gt = torch.where(torch.tensor(self.val_loader.dataset.targets_all['target']).view(-1, 1) == 0,
                                        torch.tensor([1, -1]), torch.tensor([-1, 1]))
            self.test_y_s = torch.where(torch.tensor(self.test_loader.dataset.targets_all['spurious']).view(-1, 1) == 0,
                                        torch.tensor([1, -1]), torch.tensor([-1, 1]))
            self.test_y_gt = torch.where(torch.tensor(self.test_loader.dataset.targets_all['target']).view(-1, 1) == 0,
                                         torch.tensor([1, -1]), torch.tensor([-1, 1]))
        else:
            # Multi-class classification (counteranimal)
            # One-hot encoding
            val_targets = torch.tensor(self.val_loader.dataset.targets_all['target'])
            test_targets = torch.tensor(self.test_loader.dataset.targets_all['target'])
            val_spurious = torch.tensor(self.val_loader.dataset.targets_all['spurious'])
            test_spurious = torch.tensor(self.test_loader.dataset.targets_all['spurious'])
            
            self.val_y_gt = torch.nn.functional.one_hot(val_targets, num_classes=n_classes).float()
            self.test_y_gt = torch.nn.functional.one_hot(test_targets, num_classes=n_classes).float()
            self.val_y_s = torch.nn.functional.one_hot(val_spurious, num_classes=2).float()
            self.test_y_s = torch.nn.functional.one_hot(test_spurious, num_classes=2).float()
        self.img_model.eval(), self.txt_model.eval()
        name, folder = self.cfg.arch + '_' + self.cfg.dataset, 'codebook'
        self.val_z_path, self.test_z_path, self.train_z_path = f'{folder}/val_{name}', f'{folder}/test_{name}', f'{folder}/train_{name}'
        if not os.path.exists(self.val_z_path):
            save_to_hdf5(self.val_z_path, self.val_loader, self.img_model, self.device)
        if not os.path.exists(self.test_z_path):
            save_to_hdf5(self.test_z_path, self.test_loader, self.img_model, self.device)
        self.cls_cb = self.bias_txt_model.get_embedding(self.cls_text)
        self.bias_cb = self.bias_txt_model.get_embedding(self.bias_text)
        self.mix_cb = self.bias_txt_model.get_embedding(self.mix_text)
        self.test_cb = self.bias_txt_model.get_embedding(self.test_text)
        self.debias_main_cb = self.cls_cb

    def text_iccv(self, txt_iters, iter, use_att):
        self.img_model.eval(), self.txt_model.train(), self.bias_img_model.eval(), self.bias_txt_model_freeze.eval()
        target_model, device, sub = self.txt_model, self.device, torch.tensor(0, device=self.device).half()
        target_model.zero_grad()

        lr, target_s, target = self.cfg.lr, 0, 3
        if self.data == 'waterbirds' or self.cfg.arch != 'ViTL14' or self.data == 'counteranimal' or self.data == 'fairface':
            self.test()
        with torch.no_grad():
            z, t_ = target_model.get_feature(self.cls_text, only_vec=False)
            z_bias, t_b = target_model.get_feature(self.bias_text, only_vec=False)
            z_test, t_tt = target_model.get_feature(self.test_text, only_vec=False)
            cls_cb, bias_cb, test_cb = target_model(z, t_), target_model(z_bias, t_b), target_model(z_test, t_tt)

        optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=lr)

        if self.data == 'waterbirds':
            if self.cfg.arch == 'ViTL14':
                up_, num_sam, bnd = 100, 10, 1.0
                #att_stp, keep_weight, rand = 0.0037, 0.92, 0.22
                att_stp, keep_weight, rand = 0.0037, 0, 0.22
            else:
                up_, num_sam, bnd = 100, 10, 5.0
                #att_stp, keep_weight, rand = 0.1, 0.772, 0.23
                att_stp, keep_weight, rand = 0.007508209275074919, 0.5905595235290401, 0.2667700549735428
        elif self.data == 'counteranimal':
            if self.cfg.arch == 'ViTL14':
                up_, num_sam, bnd = 100, 10, 1.0
                # up_, num_sam, bnd = 100, 1, 1.0
                att_stp, keep_weight, rand = 0.0037, 0.92, 0.22
                att_stp, keep_weight, rand = 0.0037, 0.7, 0.3
                # att_stp, keep_weight, rand = 0.01, 0.8, 0.2
                # up : 100, att_stp : 0.01, keep_weight : 0.6, num_sam" 10 target :0~3, rand:0.2, bnd:1.0
                # up : 100, att_stp : 0.005, keep_weight : 0.6, num_sam" 10 target :0~3, rand:0.2, bnd:1.0
            else:
                up_, num_sam, bnd = 100, 10, 5.0
                att_stp, keep_weight, rand = 0.0037, 0.8, 0.22
        elif self.data == 'fairface':
            # FairFace: BendVLM stereotype-based debiasing (25 stereotypes x 2 genders)
            if self.cfg.arch == 'ViTL14':
                up_, num_sam, bnd = 100, 10, 1.0
                att_stp, keep_weight, rand = 0.0037, 0.92, 0.22
                att_stp, keep_weight, rand = 0.003, 0.92, 0.22
                att_stp, keep_weight, rand = 0.001, 0.9, 0.1
                # att_stp, keep_weight, rand = 0.093, 0.81, 0.168
            else:
                up_, num_sam, bnd = 100, 10, 5.0
                att_stp, keep_weight, rand = 0.01, 0.7, 0.2
            self.bias_cbs = {}
            for attr in ['gender', 'race', 'age']:
                with torch.no_grad():
                    prompts = self.bias_text[attr] # 예: ['Male', 'Female'...]
                    z_b, t_b = self.txt_model.get_feature(prompts, only_vec=False)
                    self.bias_cbs[attr] = self.txt_model(z_b, t_b)
        else:
            up_, num_sam, bnd = 100, 10, 1.0
            if self.cfg.arch == 'ViTL14':
                #att_stp, keep_weight, rand = 0.093, 0, 0.168
                att_stp, keep_weight, rand = 0.093, 0.81, 0.168
                num_sam = 15
            else:
                att_stp, keep_weight, rand = 0.012167458566406495, 0.8383026903635123, 0.44402560542911457
                #att_stp, keep_weight, rand = 0.012167458566406495, 0, 0.44402560542911457

        for epo in range(txt_iters):
            optimizer.zero_grad()
            eq_loss, distil_loss, equa_loss, equa_adv_loss, pgd_loss, ck_loss, match_loss \
                = dc(sub), dc(sub), dc(sub), dc(sub), dc(sub), dc(sub), dc(sub)
            cls_em, test_em = target_model(z, t_), target_model(z_test, t_tt)
            if self.data == 'waterbirds':
                if use_att:
                    z_target = z_test[:, target_s:target]
                    z_adv_set1, z_adv_set2 = (
                        perturb_bafa_txt_multi_ablation_lb_ls(z_target, target_model, bias_cb, t_tt[target_s:target], test_cb,
                                                    att_bnd=bnd, random=rand, keep_weight=keep_weight, att_stp=att_stp,
                                                    num_samples=num_sam, use_ls=self.cfg.use_ls))
                    with torch.no_grad():
                        adv_cb_set2 = target_model(z_adv_set2, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))
                        adv_cb_set1 = target_model(z_adv_set1, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))

                    S = (adv_cb_set1 - adv_cb_set2)
                    match_loss += (S[1::3] @ cls_em[:1].T).pow(2).mean()
                    match_loss += (S[2::3] @ cls_em[1:].T).pow(2).mean()
                    match_loss = match_loss * up_ if self.cfg.arch != 'ViTL14' else match_loss

                # ck_loss = ((bias_cb[:4] - bias_cb[4:]) @ cls_em.T).pow(2).mean() * up_
                if not epo:
                    print(f'====== up : {up_}, att_stp : {att_stp}, keep_weight : {keep_weight}, num_sam" {num_sam} target :{target_s}~{target}, rand:{rand}, bnd:{bnd}')
                if epo % 50 == 0 and epo:
                    self.test()
            elif self.data == 'counteranimal':
                # CounterAnimal: 15 classes with class-specific backgrounds
                # bias_cb[:15] = common backgrounds, bias_cb[15:] = counter backgrounds
                # test_prompts[3:] = 15 class-specific prompts (skip 3 generic)

                # Map our 15 classes to their actual indices in cls_em (45 classes)
                class_indices = [8, 10, 12, 14, 25, 26, 30, 32, 33, 34, 35, 39, 40, 41, 42]
                class_indices = [8, 12, 26, 30, 32, 35]
                # bulbul, vulture, hyena, Arctic fox, mink, otter, agama,
                # hognose snake, king snake, garter snake, water snake,
                # centipede, black grouse, ptarmigan, prairie chicken

                n_target_classes = len(class_indices)  # 15 classes
                z_target = z_test[:, 3:]  # 15 class-specific test prompts

                z_adv_set1, z_adv_set2 = (
                    perturb_bafa_txt_multi_ablation_lb_ls(z_target, target_model, bias_cb, t_tt[3:], test_cb[:3],
                                                att_bnd=bnd, random=rand, keep_weight=keep_weight, att_stp=att_stp,
                                                num_samples=num_sam, use_ls=self.cfg.use_ls))
                with torch.no_grad():
                    adv_cb_set2 = target_model(z_adv_set2, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))
                    adv_cb_set1 = target_model(z_adv_set1, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))

                S = (adv_cb_set1 - adv_cb_set2)

                # match_loss: Class-specific like waterbirds
                # For each class, match the adversarial perturbation to that class's embedding
                for i, cls_idx in enumerate(class_indices):
                    # S has shape [num_sam * n_target_classes, D], take every n_target_classes-th starting from i
                    match_loss += (S[i::n_target_classes] @ cls_em[cls_idx:cls_idx+1].T).pow(2).mean()
                match_loss = match_loss * up_  / n_target_classes

                # ck_loss: Class-specific debiasing
                # bias_cb[:15] = "a photo of [class] on [common_bg]"
                # bias_cb[15:] = "a photo of [class] on [counter_bg]"
                # Make common and counter backgrounds have equal impact on each class
                
                # ck_loss = torch.tensor(0.0, device=cls_em.device)
                # for i, cls_idx in enumerate(class_indices):
                #     common_bg = bias_cb[i:i+1]      # class i's common background
                #     counter_bg = bias_cb[15+i:15+i+1]  # class i's counter background
                #     cls_i = cls_em[cls_idx:cls_idx+1]
                #     ck_loss += ((common_bg - counter_bg) @ cls_i.T).pow(2).mean()
                # ck_loss = ck_loss * up_ / n_target_classes

                if not epo:
                    print(f'====== [CounterAnimal] up: {up_}, att_stp: {att_stp}, keep_weight: {keep_weight}, num_sam: {num_sam}, rand: {rand}, bnd: {bnd}')
                    print(f'       bias_cb size: {bias_cb.shape[0]} (15 common + 15 counter = 30), cls_em size: {cls_em.shape[0]}')
            elif self.data == 'fairface':
                # FairFace: FairerCLIP Stereotype Debiasing (CelebA style)
                #
                # CelebA:
                #   - z_target: 3 prompts (bird, dark hair, blond hair)
                #   - bias_cb: 8 (female×4 + male×4)
                #   - cls_em: 2 (dark hair, blond hair)
                #   - ck_loss: (bias[:4] - bias[4:]) @ cls_em.T
                #
                # FairFace:
                #   - z_target: 10 stereotype prompts
                #   - bias_cb: 80 (male_variants×10×4 + female_variants×10×4)
                #   - stereotype_em: 10 stereotypes (test_em[:10])
                #   - ck_loss: (bias[:40] - bias[40:]) @ stereotype_em.T

                n_stereotypes = 10
                # n_gender_variants = 4  # male/man/boy/gentleman, female/woman/girl/lady
                # n_male = n_stereotypes * n_gender_variants  # 40
                n_gender_variants = len(self.bias_cbs['gender'])

                # test_text structure: [0:10]=stereotypes, [10:12]=gender, [12:19]=race, [19:28]=age
                # Use stereotype prompts for adversarial perturbation (like CelebA uses hair prompts)
                z_target = z_test[:, -n_stereotypes:]  # 10 stereotype prompts
                target = test_em[-n_stereotypes:]
                z_adv_set1, z_adv_set2 = (
                    perturb_bafa_txt_multi_ablation_lb_ls(z_target, target_model, self.bias_cbs['gender'], t_tt[-n_stereotypes:], test_cb[:-n_stereotypes],
                                                att_bnd=bnd, random=rand, keep_weight=keep_weight, att_stp=att_stp,
                                                num_samples=num_sam, use_ls=self.cfg.use_ls))
                with torch.no_grad():
                    adv_cb_set2 = target_model(z_adv_set2, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))
                    adv_cb_set1 = target_model(z_adv_set1, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))

                S = (adv_cb_set1 - adv_cb_set2)

                # match_loss: CelebA style - for each stereotype, match adversarial perturbation
                # S has shape [num_sam * n_stereotypes, D]
                for i in range(n_stereotypes):
                    match_loss += (S[i::n_stereotypes] @  target[i:i+1].T).pow(2).mean()
                match_loss = match_loss * up_ / n_stereotypes

                # ck_loss: CelebA style - (male_variants - female_variants) @ stereotype_em.T
                # bias_cb[:20] = [good male, good man, evil male, evil man, ...]
                # bias_cb[20:] = [good female, good woman, evil female, evil woman, ...]
                # male_embeds = bias_cb[:n_male]       # 20 male variant embeddings
                # female_embeds = bias_cb[n_male:]     # 20 female variant embeddings
                # stereotype_em = test_em[:n_stereotypes]  # 10 stereotype embeddings

                # ck_loss = ((male_embeds - female_embeds) @ stereotype_em.T).pow(2).mean() * up_
                ck_loss = ((self.bias_cbs['gender'][:n_gender_variants//2] - self.bias_cbs['gender'][n_gender_variants//2:]) @ target.T).pow(2).mean() * up_
                # ck_loss = ((self.bias_cbs['race'][:6] - self.bias_cbs['race'][6:]) @ target.T).pow(2).mean() * up_

                if not epo:
                    print(f'====== [FairFace] up: {up_}, att_stp: {att_stp}, keep_weight: {keep_weight}, num_sam: {num_sam}, rand: {rand}, bnd: {bnd}')
                    # print(f'       bias_cb: {n_male}(male/man/boy/gentleman) + {n_male}(female/woman/girl/lady) = {n_male*2}')
                    print(f'       stereotype_em: {n_stereotypes} (good, evil, smart, dumb, ...)')
                if epo % 50 == 0 and epo:
                    self.test()
            else:
                z_target = z_test[:, target_s:target]
                z_adv_set1, z_adv_set2 = (
                    perturb_bafa_txt_multi_ablation_lb_ls(z_target, target_model, bias_cb, t_tt[:3], test_cb, att_bnd=bnd,
                                            random=rand, keep_weight= keep_weight, att_stp=att_stp, num_samples=num_sam, use_ls=self.cfg.use_ls))
                if not epo:
                    print(
                        f'====== up : {up_}, att_stp : {att_stp}, keep_weight : {keep_weight}, num_sam" {num_sam} target :{target_s}~{target}, rand:{rand}, bnd:{bnd}')
                with torch.no_grad():
                    adv_cb_set2 = target_model(z_adv_set2, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))
                    adv_cb_set1 = target_model(z_adv_set1, torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam))
                S =  adv_cb_set1 - adv_cb_set2
                match_loss = (S[1::3] @ cls_em[:1].T).pow(2).mean() * up_
                match_loss += (S[2::3] @ cls_em[1:].T).pow(2).mean() * up_

                if epo % 10 == 0 and epo:
                    self.test()
            loss = eq_loss + distil_loss + ck_loss + match_loss # / 100
            loss.backward()
            optimizer.step()
            # Verbose every 10 epochs for counteranimal to see worst group
            verbose = (self.data == 'counteranimal' and epo % 10 == 0)
            avg_acc, robust_acc, groups_acc = self.val_(epo, verbose=verbose)
            # Show retrieval-based MaxSkew for FairFace, worst acc for others
            if self.cfg.dataset == 'fairface' and isinstance(groups_acc, dict) and 'avg_gender_skew' in groups_acc:
                print(f'txt {epo} epoch | eq_loss: {eq_loss.item():.4f} | ck_loss: {ck_loss.item():.4f} '
                      f'| match: {match_loss.item():.4f} | Gender MaxSkew: {groups_acc["avg_gender_skew"]:.4f} '
                      f'| Race MaxSkew: {groups_acc["avg_race_skew"]:.4f}')
            else:
                print(f'txt {epo} epoch |' 'eq_loss : ', round(eq_loss.item(), 4), '| ck_loss:', round(ck_loss.item(), 4),
                      '| match', round(match_loss.item(), 4)
                      , '| acc : ', round(avg_acc, 4), '| worst : ', round(robust_acc, 4))
        if self.best_model_state:
            self.txt_model.load_state_dict(self.best_model_state)
            print(f'best epoch : {self.best_epoch}')
            path = f"./ckpt/txt_{self.cfg.arch}_{self.data}_up{up_}_stp{att_stp}_kw{keep_weight}_ns{num_sam}_{rand}_b{bnd}_{int(use_att)}.pt"
            torch.save(self.best_model_state, path)
            with torch.no_grad():
                z, t_ = self.txt_model.get_feature(self.cls_text, only_vec=False)
                z_c, t_c = self.txt_model.get_feature(self.test_text, only_vec=False)
                self.debias_main_cb, self.center = self.txt_model(z, t_), self.txt_model(z_c, t_c)
                print('---eval test data---')
                avg_acc, robust_acc = self.test()
                self.result['best'] = {'epoch': self.best_epoch, 'avg_acc': avg_acc, 'robust_acc': robust_acc}
            print(
                f'====== up : {up_}, att_stp : {att_stp}, keep_weight : {keep_weight}, num_sam" {num_sam} target :{target_s}~{target}, rand:{rand}, bnd:{bnd}')
    def text_iccv_test(self, txt_iters, mode=None):
        self.img_model.eval(), self.txt_model.train(), self.bias_img_model.eval(), self.bias_txt_model_freeze.eval()
        target_model, device, sub = self.txt_model, self.device, torch.tensor(0, device=self.device).half()
        target_model.zero_grad()
        lr = self.cfg.lr
        if self.data == 'waterbirds' or self.cfg.arch != 'ViTL14':
            self.test()
        with torch.no_grad():
            z, t_ = target_model.get_feature(self.cls_text, only_vec=False)
            z_bias, t_b = target_model.get_feature(self.bias_text, only_vec=False)
            z_mix, t_m = target_model.get_feature(self.mix_text, only_vec=False)
            z_test, t_tt = target_model.get_feature(self.test_text, only_vec=False)
            cls_cb, bias_cb, test_cb = target_model(z, t_), target_model(z_bias, t_b), target_model(z_test, t_tt)
            self.test_cb, center = test_cb, test_cb[0]

        optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=lr)
        for epo in range(txt_iters):
            optimizer.zero_grad()
            eq_loss, distil_loss, equa_loss, equa_adv_loss, pgd_loss, ck_loss, match_loss = dc(sub), dc(sub), dc(
                sub), dc(sub), dc(sub), dc(sub), dc(sub)
            cls_em, test_em, num_sam, target_s, target = target_model(z, t_), target_model(z_test, t_tt), 30, 0, 3
            up_ = 30 if self.cfg.arch == 'ViTL14' else 100

            z_target = z_test[:, target_s:target]
            if mode == 'prompt':
                with torch.no_grad():
                    mix_em = target_model(z_mix, t_m)
                female_list, male_list = [], []
                for offset in [0, 8]:
                    for i in range(4):
                        female_list.append(mix_em[offset + i])  # female: offset ~ offset+3
                        male_list.append(mix_em[offset + i + 4])  # male: offset+4 ~ offset+7
                cls_emm = cls_em * up_
                match_loss = F.mse_loss(torch.stack(female_list, dim=0)[:4] @ cls_emm.T,
                                        torch.stack(male_list, dim=0)[:4] @ cls_emm.T)
            else:
                att_stp = 4e-3 if self.cfg.arch == 'ViTL14' else 1e-2
                z_adv_set1, z_adv_set2 = perturb_bafa_txt_multi_all_noli(z_target, target_model, bias_cb, t_tt[target_s:target],
                                    att_bnd=0.3, random=0.1, att_stp=att_stp, num_samples=num_sam)
                with torch.no_grad():
                    adv_cb_set2 = target_model(z_adv_set2,
                                               torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam)).float()
                    adv_cb_set1 = target_model(z_adv_set1,
                                               torch.tensor(t_tt[:z_target.size(1)].tolist() * num_sam)).float()
                cls_em = cls_em.float() # * up_
                match_loss += F.mse_loss(adv_cb_set1[0::3] @ cls_em.T, adv_cb_set2[0::3] @ cls_em.T)
                match_loss += F.mse_loss(adv_cb_set1[1::3] @ cls_em[:1].T, adv_cb_set2[1::3] @ cls_em[:1].T)
                match_loss += F.mse_loss(adv_cb_set1[2::3] @ cls_em[1:].T, adv_cb_set2[2::3] @ cls_em[1:].T)

            if epo % 10 == 0 and epo:
                self.test()
            loss = eq_loss + distil_loss + ck_loss + match_loss
            loss.backward()
            optimizer.step()
            avg_acc, robust_acc, groups_acc = self.val_(epo)
            # Show retrieval-based MaxSkew for FairFace, worst acc for others
            if self.cfg.dataset == 'fairface' and isinstance(groups_acc, dict) and 'avg_gender_skew' in groups_acc:
                print(f'txt {epo} epoch | eq_loss: {eq_loss.item():.4f} | ck_loss: {ck_loss.item():.4f} '
                      f'| match: {match_loss.item():.4f} | Gender MaxSkew: {groups_acc["avg_gender_skew"]:.4f} '
                      f'| Race MaxSkew: {groups_acc["avg_race_skew"]:.4f}')
            else:
                print(f'txt {epo} epoch |' 'eq_loss : ', round(eq_loss.item(), 4), '| ck_loss:', round(ck_loss.item(), 4),
                      '| match', round(match_loss.item(), 4)
                      , '| acc : ', round(avg_acc, 4), '| worst : ', round(robust_acc, 4))
        if self.best_model_state:
            self.txt_model.load_state_dict(self.best_model_state)
            print(f'best epoch : {self.best_txt_epoch}')

    def img_iccv(self, img_iters, nw_iter=None, stop_it_=None):
        self.txt_model.eval(), self.img_model.train()
        device = self.device
        target_model, bias_txt_cb = self.img_model, self.cls_cb.to(device).half()
        with torch.no_grad():
            z, t_ = self.txt_model.get_feature(self.cls_text, only_vec=False)
            debias_txt_cb = self.txt_model(z, t_)

        teacher_layer, sub, lambda_ = dc(target_model), torch.tensor(0, device=self.device).half(), 1
        teacher_layer.eval()  # [1-0.001-txt20_img1 | 2-0.01-same | 3 - txt att - same | 4 -]
        l_scale_learn, rand_eps  = 100, 0.0
        if self.data == 'waterbirds':
            l_scale_learn = 100
            if self.cfg.arch == 'ViTL14':
                l_scale_learn = 1000
                bnd, step, iters, l_scale, lr, lambda_, rand_eps = 5.0, 1e-6, 10, 100.0, self.cfg.img_lr, 0.8, 1e-2
            else:
                bnd, step, iters, l_scale, lr, lambda_, rand_eps = 5.0, 3e-6, 10, 100.0, self.cfg.img_lr, 0.8, 1e-3
        else:

            if self.cfg.arch == 'ViTL14':
                bnd, step, iters, l_scale, lr, lambda_ = 5.0, 3e-4, 10, 100.0, self.cfg.img_lr, 1 # ViT celeba
            else:
                bnd, step, iters, l_scale, lr, lambda_ = 5.0, 1e-5, 10, 100.0, self.cfg.img_lr, 1 #
                # python train_bafa.py --att_mode txt_guidance --learn_mode proj --iter 2 --img_epochs 1 --epochs 150
                # --config-file configs/debias_celeba_RN50.yaml --lr 1e-7 --img_lr 1e-7
                # --use_txt_ck_path txt_RN50_celebA_up100_stp0.1_kw0.7_ns10_0.1_b1.0.pt --opt nolabel True mode iccv
        optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=lr)

        for epo in range(img_iters):
            teacher_layer = dc(target_model)
            teacher_layer.eval()
            optimizer.zero_grad()
            eq_l, dist_l, total_right, keep, bias_r, bias_w, keep_ = 0, 0, 0, 0, 0, 0, 0
            for it_, (batch_x, y, _) in enumerate(self.train_loader):
                batch_x = batch_x.to(device)
                with torch.no_grad():
                    z = target_model.get_feature(batch_x)
                    with torch.no_grad():
                        if self.data == 'waterbirds':
                            img_cb = teacher_layer(z)
                        else:
                         img_cb = teacher_layer(z)
                out_embed = target_model(z)
                z_adv_bias = bafa_img_iccv(z, target_model, bias_txt_cb, debias_txt_cb, bnd, step, iters, l_scale, lambda_, rand_eps)

                adv_embed = target_model(z_adv_bias)
                S = l_scale_learn * (img_cb - adv_embed)  # [N, D]
                loss = (S @ debias_txt_cb.T).pow(2).mean()
                if not epo and not it_:
                    print(f"====== bnd:{bnd}, step:{step}, iters:{iters}, l_scale:{l_scale}, lr:{lr}, lambda_ :{lambda_ }")

                if torch.isnan(out_embed).any():
                    print('*' * 50, "Tensor contains NaN values.")
                if not loss:
                    continue

                loss.backward(retain_graph=True)
                optimizer.step()
                target_model.zero_grad()

                if not it_ % 20 and self.data != 'waterbirds':
                    avg_acc, robust_acc, groups_acc = self.val_(it_,img=True)
                    # Show joint classification for FairFace, worst acc for others
                    if self.cfg.dataset == 'fairface' and isinstance(groups_acc, dict) and 'joint_acc' in groups_acc:
                        print(f"img {epo} epoch | loss: {loss.item():.4f} | joint: {groups_acc['joint_acc']*100:.2f}% "
                              f"| G: {groups_acc['gender_acc']*100:.2f}% | R: {groups_acc['race_acc']*100:.2f}% | A: {groups_acc['age_acc']*100:.2f}%")
                    else:
                        print(f"img {epo} epoch | loss: {loss.item():.4f} | acc: {avg_acc:.4f} | worst: {robust_acc:.4f}")
                    if (self.data != 'waterbirds' and not it_ % 500):
                        self.test()
            avg_acc, robust_acc, groups_acc = self.val_(epo, img=True)
            # Show joint classification for FairFace, worst acc for others
            if self.cfg.dataset == 'fairface' and isinstance(groups_acc, dict) and 'joint_acc' in groups_acc:
                print(f"img {epo} epoch | loss: {loss.item():.4f} | joint: {groups_acc['joint_acc']*100:.2f}% "
                      f"| G: {groups_acc['gender_acc']*100:.2f}% | R: {groups_acc['race_acc']*100:.2f}% | A: {groups_acc['age_acc']*100:.2f}%")
            else:
                print(f"img {epo} epoch | loss: {loss.item():.4f} | acc: {avg_acc:.4f} | worst: {robust_acc:.4f}")
            if epo:
                if epo and not epo % 10:
                    self.test()
            if eq_l > 1:
                print('somthing wrong')
        self.img_model.eval()
        print(f"====== bnd:{bnd}, step:{step}, iters:{iters}, l_scale:{l_scale}, lr:{lr}, lambda_ :{lambda_ },scalearn{l_scale_learn}")

        if self.best_model_state and self.data == 'waterbirds':
            self.img_model.load_state_dict(self.best_model_state)
            print(f'best epoch : {self.best_epoch}')
            path = f"./ckpt/img_{self.cfg.arch}_{self.data}_up{l_scale}_stp{step}_b{bnd}_lr{lr}_lambda_{lambda_}.pt"
            torch.save(self.best_model_state, path)



    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.txt_model.load_state_dict(checkpoint["txt_model"])
            self.img_model.load_state_dict(checkpoint["img_model"])
            self.bias_img_model.load_state_dict(checkpoint["bias_img_model"])
            self.bias_txt_model_freeze.load_state_dict(checkpoint["bias_txt_model_freeze"])
            print(f"✅ Best model loaded (robust_acc: {self.best_robust_acc:.4f})")

    def save_best_model(self):
        torch.save({"txt_model": self.txt_model.state_dict(),
            "img_model": self.img_model.state_dict(), }, self.best_model_path)

    def test(self, P=None):
        self.img_model.eval(), self.txt_model.eval()
        img_embeds, gt = [], []
        with torch.no_grad():
            self.img_model.eval(), self.txt_model.eval()
            z, t_ = self.txt_model.get_feature(self.cls_text, only_vec=False)
            debias_txt_em = self.txt_model(z, t_) if P is None else F.normalize(self.txt_model(z, t_) @ P.T, dim=-1)
            for test_batch in load_hdf5_in_batches(self.test_z_path, self.device, batch_size=128):
                img_embeds.append(self.img_model(test_batch.half()))
            img_embeds_cat = torch.cat(img_embeds).float()

            # FairFace: Use retrieval-based MaxSkew evaluation
            if self.cfg.dataset == 'fairface':
                avg_acc, robust_acc, groups_acc = self._eval_fairface_maxskew_test(
                    img_embeds_cat, debias_txt_em.float(), top_k=1000, verbose=True)
            else:
                predict_our = self.uo.get_zeroshot_predictions(img_embeds_cat, debias_txt_em.float(),
                                                               self.cfg, temperature=100.)
                # Get class names for counteranimal
                class_names = None
                if self.cfg.dataset == 'counteranimal' and hasattr(self.test_loader.dataset, 'class_names'):
                    class_names = self.test_loader.dataset.class_names
                avg_acc, robust_acc, groups_acc = evaluate_clip(predict_our, self.test_y_gt, self.test_y_s, verbose=True,
                                                                dataset_type=self.cfg.dataset, class_names=class_names)
            self.result[str(self.epo)] = {'avg_acc': avg_acc, 'robust_acc': robust_acc}
            return avg_acc, robust_acc

    def _eval_fairface_joint_test(self, img_embeds, txt_embeds, verbose=True, top_k=1000):
        """
        Evaluate FairFace using FairerCLIP Retrieval-based MaxSkew on test set.

        FairerCLIP method:
        1. Use 10 stereotype queries (good, evil, smart, ...)
        2. For each query, retrieve top-K images
        3. Measure gender/race distribution in retrieved set
        4. MaxSkew@K = max_g |log(p_g / q_g)|
        """
        import numpy as np
        import pandas as pd
        import torch

        # Normalize embeddings
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        # Get ground truth labels from CSV (test set)
        data_dir = '/home2/connor/data/fairface/'
        val_csv = pd.read_csv(f'{data_dir}/fairface_label_val.csv')

        gender_map = {'Male': 0, 'Female': 1}
        race_map = {'White': 0, 'Black': 1, 'East Asian': 2, 'Southeast Asian': 3,
                    'Indian': 4, 'Middle Eastern': 5, 'Latino_Hispanic': 6}

        gender_labels = val_csv['gender'].map(gender_map).values
        race_labels = val_csv['race'].map(race_map).values

        n_total = len(gender_labels)

        # Gender/Race prior distribution - Use ACTUAL dataset distribution
        gender_prior = {
            0: (gender_labels == 0).sum() / n_total,  # Male ratio
            1: (gender_labels == 1).sum() / n_total   # Female ratio
        }
        race_prior = {r: (race_labels == r).sum() / n_total for r in range(7)}

        # ========================================
        # FairerCLIP Retrieval-based MaxSkew
        # ========================================
        stereotype_queries = [
            'good', 'evil', 'smart', 'dumb', 'attractive',
            'unattractive', 'lawful', 'criminal', 'friendly', 'unfriendly'
        ]

        # Get stereotype embeddings
        with torch.no_grad():
            z_stereo, t_stereo = self.txt_model.get_feature(self.test_text[:10], only_vec=False)
            stereotype_embeds = self.txt_model(z_stereo, t_stereo).float()
            stereotype_embeds = stereotype_embeds / stereotype_embeds.norm(dim=-1, keepdim=True)

        eps = 1e-10
        all_gender_skews = []
        all_race_skews = []
        skew_details = {}

        for i, query_name in enumerate(stereotype_queries):
            query_embed = stereotype_embeds[i:i+1]
            similarities = (img_embeds @ query_embed.T).squeeze()

            top_k_actual = min(top_k, len(similarities))
            top_k_indices = torch.topk(similarities, top_k_actual).indices.cpu().numpy()

            retrieved_genders = gender_labels[top_k_indices]
            retrieved_races = race_labels[top_k_indices]

            # MaxSkew for gender
            max_skew_gender = 0.0
            for g in [0, 1]:
                p_retrieved = max((retrieved_genders == g).mean(), eps)
                p_prior = max(gender_prior[g], eps)
                skew = abs(np.log(p_retrieved / p_prior))
                if skew > max_skew_gender:
                    max_skew_gender = skew

            # MaxSkew for race
            max_skew_race = 0.0
            for r in range(7):
                p_retrieved = max((retrieved_races == r).mean(), eps)
                p_prior = max(race_prior[r], eps)
                skew = abs(np.log(p_retrieved / p_prior))
                if skew > max_skew_race:
                    max_skew_race = skew

            all_gender_skews.append(max_skew_gender)
            all_race_skews.append(max_skew_race)
            skew_details[query_name] = {
                'gender_skew': max_skew_gender,
                'race_skew': max_skew_race
            }

        avg_gender_skew = np.mean(all_gender_skews)
        max_gender_skew = np.max(all_gender_skews)
        avg_race_skew = np.mean(all_race_skews)
        max_race_skew = np.max(all_race_skews)

        if verbose:
            print(f"\n{'='*60}")
            print(f"FairerCLIP Retrieval MaxSkew@{top_k} (Test Set)")
            print(f"{'='*60}")
            print(f"Gender MaxSkew - Avg: {avg_gender_skew:.4f}, Max: {max_gender_skew:.4f}")
            print(f"Race MaxSkew   - Avg: {avg_race_skew:.4f}, Max: {max_race_skew:.4f}")
            print(f"{'-'*60}")
            print(f"Per-query breakdown:")
            for q, details in skew_details.items():
                print(f"  {q:15s}: Gender={details['gender_skew']:.4f}, Race={details['race_skew']:.4f}")
            print(f"{'='*60}\n")

        groups_acc = {
            'avg_gender_skew': avg_gender_skew,
            'max_gender_skew': max_gender_skew,
            'avg_race_skew': avg_race_skew,
            'max_race_skew': max_race_skew,
            'per_query': skew_details
        }

        return -avg_gender_skew, -max_gender_skew, groups_acc

    def _eval_fairface_maxskew_test(self, img_embeds, txt_embeds, top_k=1000, verbose=True):
        """
        Evaluate FairerCLIP Retrieval-based MaxSkew for FairFace test set.
        """
        return self._eval_fairface_joint_test(img_embeds, txt_embeds, verbose, top_k)

    def val_(self, epo=0, img=False, verbose=False):
        self.img_model.eval(), self.txt_model.eval()
        img_embeds = []
        with torch.no_grad():
            z, t_ = self.txt_model.get_feature(self.cls_text, only_vec=False)
            debias_txt_em = self.txt_model(z, t_)
            for val_batch in load_hdf5_in_batches(self.val_z_path, self.device, batch_size=128):
                img_embeds.append(self.img_model(val_batch.half()))
            img_embeds_cat = torch.cat(img_embeds).float()

            # FairFace: Use retrieval-based MaxSkew evaluation
            if self.cfg.dataset == 'fairface':
                avg_acc, robust_acc, groups_acc = self._eval_fairface_maxskew(
                    img_embeds_cat, debias_txt_em.float(), top_k=1000)
            else:
                predict_our = self.uo.get_zeroshot_predictions(img_embeds_cat, debias_txt_em.float(),
                                                               self.cfg, temperature=100.)
                # Get class names for counteranimal
                class_names = None
                if self.cfg.dataset == 'counteranimal' and hasattr(self.val_loader.dataset, 'class_names'):
                    class_names = self.val_loader.dataset.class_names
                avg_acc, robust_acc, groups_acc = evaluate_clip(predict_our, self.val_y_gt, self.val_y_s, verbose=verbose,
                                                                dataset_type=self.cfg.dataset, class_names=class_names)

        # For FairFace joint classification, higher accuracy is better
        # For other datasets, higher robust_acc is better
        comparison_metric = robust_acc
        best_comparison = self.best_robust_acc

        if comparison_metric > best_comparison:
            self.best_robust_acc, self.best_epoch = robust_acc, epo
            if img:
                self.best_model_state = {k: v.cpu() for k, v in self.img_model.state_dict().items()}
            else:
                self.best_model_state = {k: v.cpu() for k, v in self.txt_model.state_dict().items()}

        return avg_acc, robust_acc, groups_acc

    def _eval_fairface_joint(self, img_embeds, txt_embeds):
        """
        Evaluate FairFace joint classification (Gender × Race × Age) on validation set.
        """
        import numpy as np
        import pandas as pd

        # Normalize embeddings
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

        # Get ground truth labels from CSV
        data_dir = '/home2/connor/data/fairface/'
        val_csv = pd.read_csv(f'{data_dir}/fairface_label_val.csv')

        gender_map = {'Male': 0, 'Female': 1}
        race_map = {'White': 0, 'Black': 1, 'East Asian': 2, 'Southeast Asian': 3,
                    'Indian': 4, 'Middle Eastern': 5, 'Latino_Hispanic': 6}
        age_map = {'0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3, '30-39': 4,
                   '40-49': 5, '50-59': 6, '60-69': 7, 'more than 70': 8}

        gender_labels = val_csv['gender'].map(gender_map).values
        race_labels = val_csv['race'].map(race_map).values
        age_labels = val_csv['age'].map(age_map).values

        # Build label map for 126 classes
        label_map = []
        for g in range(2):
            for r in range(7):
                for a in range(9):
                    label_map.append((g, r, a))

        # Compute similarities and get predictions
        similarities = img_embeds @ txt_embeds.T
        pred_indices = similarities.argmax(dim=-1).cpu().numpy()

        # Decode predictions
        gender_preds = np.array([label_map[i][0] for i in pred_indices])
        race_preds = np.array([label_map[i][1] for i in pred_indices])
        age_preds = np.array([label_map[i][2] for i in pred_indices])

        # Compute accuracies
        gender_acc = (gender_preds == gender_labels).mean()
        race_acc = (race_preds == race_labels).mean()
        age_acc = (age_preds == age_labels).mean()
        joint_acc = ((gender_preds == gender_labels) &
                     (race_preds == race_labels) &
                     (age_preds == age_labels)).mean()

        groups_acc = {
            'joint_acc': joint_acc,
            'gender_acc': gender_acc,
            'race_acc': race_acc,
            'age_acc': age_acc
        }
        # Return: avg_acc (joint), robust_acc (gender_acc for best model selection), groups_acc
        return joint_acc, gender_acc, groups_acc

    def _eval_fairface_maxskew(self, img_embeds, txt_embeds, top_k=1000):
        """
        Evaluate FairFace using FairerCLIP Retrieval-based MaxSkew.

        FairerCLIP method:
        1. Use 10 stereotype queries (good, evil, smart, ...)
        2. For each query, retrieve top-K images
        3. Measure gender/race distribution in retrieved set
        4. MaxSkew@K = max_g |log(p_g / q_g)|

        Returns:
            avg_maxskew: average MaxSkew across queries (for model selection)
            max_maxskew: maximum MaxSkew
            groups_acc: dict with all metrics
        """
        import numpy as np
        import pandas as pd
        import torch

        # Normalize embeddings
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        # Get ground truth labels from CSV
        data_dir = '/home2/connor/data/fairface/'
        val_csv = pd.read_csv(f'{data_dir}/fairface_label_val.csv')

        gender_map = {'Male': 0, 'Female': 1}
        race_map = {'White': 0, 'Black': 1, 'East Asian': 2, 'Southeast Asian': 3,
                    'Indian': 4, 'Middle Eastern': 5, 'Latino_Hispanic': 6}

        gender_labels = val_csv['gender'].map(gender_map).values
        race_labels = val_csv['race'].map(race_map).values

        n_total = len(gender_labels)

        # Gender/Race prior distribution - Use ACTUAL dataset distribution
        gender_prior = {
            0: (gender_labels == 0).sum() / n_total,  # Male ratio
            1: (gender_labels == 1).sum() / n_total   # Female ratio
        }
        race_prior = {r: (race_labels == r).sum() / n_total for r in range(7)}

        # ========================================
        # FairerCLIP Retrieval-based MaxSkew
        # test_text[:10] = stereotype queries
        # ========================================
        stereotype_queries = [
            'good', 'evil', 'smart', 'dumb', 'attractive',
            'unattractive', 'lawful', 'criminal', 'friendly', 'unfriendly'
        ]

        # Get stereotype embeddings from test_em (first 10)
        with torch.no_grad():
            z_stereo, t_stereo = self.txt_model.get_feature(self.test_text[:10], only_vec=False)
            stereotype_embeds = self.txt_model(z_stereo, t_stereo).float()
            stereotype_embeds = stereotype_embeds / stereotype_embeds.norm(dim=-1, keepdim=True)

        eps = 1e-10
        all_gender_skews = []
        all_race_skews = []
        skew_details = {}

        for i, query_name in enumerate(stereotype_queries):
            # Compute similarity with this stereotype query
            query_embed = stereotype_embeds[i:i+1]  # [1, D]
            similarities = (img_embeds @ query_embed.T).squeeze()  # [N]

            # Get top-K indices
            top_k_actual = min(top_k, len(similarities))
            top_k_indices = torch.topk(similarities, top_k_actual).indices.cpu().numpy()

            # Get gender/race distribution in retrieved samples
            retrieved_genders = gender_labels[top_k_indices]
            retrieved_races = race_labels[top_k_indices]

            # MaxSkew for gender
            max_skew_gender = 0.0
            for g in [0, 1]:
                p_retrieved = max((retrieved_genders == g).mean(), eps)
                p_prior = max(gender_prior[g], eps)
                skew = abs(np.log(p_retrieved / p_prior))
                if skew > max_skew_gender:
                    max_skew_gender = skew

            # MaxSkew for race
            max_skew_race = 0.0
            for r in range(7):
                p_retrieved = max((retrieved_races == r).mean(), eps)
                p_prior = max(race_prior[r], eps)
                skew = abs(np.log(p_retrieved / p_prior))
                if skew > max_skew_race:
                    max_skew_race = skew

            all_gender_skews.append(max_skew_gender)
            all_race_skews.append(max_skew_race)
            skew_details[query_name] = {
                'gender_skew': max_skew_gender,
                'race_skew': max_skew_race
            }

        # Average and Max across all queries
        avg_gender_skew = np.mean(all_gender_skews)
        max_gender_skew = np.max(all_gender_skews)
        avg_race_skew = np.mean(all_race_skews)
        max_race_skew = np.max(all_race_skews)

        groups_acc = {
            'avg_gender_skew': avg_gender_skew,
            'max_gender_skew': max_gender_skew,
            'avg_race_skew': avg_race_skew,
            'max_race_skew': max_race_skew,
            'per_query': skew_details
        }

        print(f"[FairerCLIP Retrieval MaxSkew@{top_k}]")
        print(f"  Gender - Avg: {avg_gender_skew:.4f}, Max: {max_gender_skew:.4f}")
        print(f"  Race   - Avg: {avg_race_skew:.4f}, Max: {max_race_skew:.4f}")

        # Return: use negative of avg_gender_skew for model selection (lower is better)
        # So higher "robust_acc" means lower skew
        return -avg_gender_skew, -max_gender_skew, groups_acc

    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # Format with class
                class_embeddings = self.text_model.get_embedding(texts)  # Embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.cfg.device)
        return zeroshot_weights

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).sum(0, keepdim=True).cpu().numpy()) for k in topk]

    def zero_shot_evaluation(self, dataset, class_names, templates):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        zeroshot_weights = self.zeroshot_classifier(class_names, templates)

        top1, top5, n = 0., 0., 0.
        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(self.cfg.device)
                target = target.to(self.cfg.device)

                # Predict
                image_features = self.image_model.encod(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                # Measure accuracy
                acc1, acc5 = self.accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        return top1, top5

    def cifar_test(self):
        import os
        from torchvision.datasets import CIFAR100, CIFAR10
        # Load the datasets
        root = os.path.expanduser("~/.cache")
        test_cifar100 = CIFAR100(root, download=True, train=False, transform=self.text_model.base_transform)
        test_cifar10 = CIFAR10(root, download=True, train=False, transform=self.text_model.base_transform)

        # Prompt templates for CIFAR-10 and CIFAR-100
        templates = [
            'a photo of a {}.', 'a blurry photo of a {}.', 'a black and white photo of a {}.',
            'a low contrast photo of a {}.', 'a high contrast photo of a {}.', 'a bad photo of a {}.',
            'a good photo of a {}.', 'a photo of a small {}.', 'a photo of a big {}.', 'a photo of the {}.',
            'a blurry photo of the {}.', 'a black and white photo of the {}.', 'a low contrast photo of the {}.',
            'a high contrast photo of the {}.', 'a bad photo of the {}.', 'a good photo of the {}.',
            'a photo of the small {}.', 'a photo of the big {}.',
        ]

        # Evaluate on CIFAR-100
        cifar100_top1, cifar100_top5 = self.zero_shot_evaluation(test_cifar100, test_cifar100.classes, templates)
        print(f"CIFAR-100 Zero-shot CLIP model Top-1 accuracy: {cifar100_top1:.2f}%")
        print(f"CIFAR-100 Zero-shot CLIP model Top-5 accuracy: {cifar100_top5:.2f}%")

        # Evaluate on CIFAR-10
        cifar10_top1, cifar10_top5 = self.zero_shot_evaluation(test_cifar10, test_cifar10.classes, templates)
        print(f"CIFAR-10 Zero-shot CLIP model Top-1 accuracy: {cifar10_top1:.2f}%")
        print(f"CIFAR-10 Zero-shot CLIP model Top-5 accuracy: {cifar10_top5:.2f}%")

    def cifar_test2(self):
        from test_utils.zs import Cifar_test
        self.txt_model.eval(), self.img_model.eval()
        cifar_test_ = Cifar_test(dc(self.txt_model), dc(self.img_model), self.cfg, self.uo.base_transform)
        cifar_test_.cifar_test()

    def imagenet_test(self):
        from test_utils.zs import Imagenet_test
        # Define paths to validation images and annotations
        img_dir = "../../ILSVRC/Data/CLS-LOC/val/"
        anno_dir = "../../ILSVRC/Annotations/CLS-LOC/val/"

        from utils.imagenet_info import templates, class_names
        imagenet_test_ = Imagenet_test(dc(self.txt_model), dc(self.img_model), self.cfg, self.uo.base_transform)
        imagenet_test_.imagenet_test(img_dir, anno_dir, class_names, templates)


class Use_original:
    def __init__(self, base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions,
                 cfg):
        self.base_model = base_model.to(cfg.device)
        self.tokenizer = clip.tokenize
        self.base_transform = base_transform  # img
        self.get_embeddings = get_embeddings
        self.get_dataset_embeddings = get_dataset_embeddings
        self.get_zeroshot_predictions = get_zeroshot_predictions
        self.cfg = cfg

    def get_embedding(self, x):
        x = self.get_embeddings(x, self.base_model, self.cfg, normalize=True, verbose=False)
        return x


class SetTarget(nn.Module):
    def __init__(self, cfg, encoder, target_layer_num=1, modal='img', learn_mode='linear'):
        super(SetTarget, self).__init__()
        self.encoder = encoder
        self.dtype = self.encoder.dtype if modal == 'txt' else self.encoder.conv1.weight.dtype
        self.learn_mode = learn_mode
        for param in self.encoder.parameters():  # freeze model
            param.requires_grad = False
        if learn_mode == 'proj':
            if modal == 'img':
                try:
                    self.encoder.proj.requires_grad = True
                except AttributeError:
                    self.encoder.attnpool.c_proj.requires_grad_(True)
            elif modal == 'txt':
                self.encoder.text_projection.requires_grad = True
        elif learn_mode == 'linear':
            for param in self.encoder.transformer.resblocks[-target_layer_num:].parameters():
                param.requires_grad = True
        elif learn_mode == 'lora':
            from trainer.peft import TargetLoRA
            for i in range(-target_layer_num, 0):
                block = self.encoder.transformer.resblocks[i]
                block.attn = TargetLoRA(block.attn, r=4)
                for name, param in block.named_parameters():
                    if 'lora_A' not in name and 'lora_B' not in name:
                        param.requires_grad = False
                    if 'ln' not in name and param.dtype == torch.float32:
                        param.data = param.data.to(torch.float16)

        elif learn_mode == 'vpt':
            from trainer.peft import TargetVPT
            for i in range(-target_layer_num, 0):
                self.encoder.transformer.resblocks[i] = TargetVPT(self.encoder.transformer.resblocks[i], prompt_size=10)
        self.target_layer_num = target_layer_num
        self.device = cfg.device
        self.cfg = cfg
        self.modal = modal

    def tokenizer(self, text):
        text_tokens = clip.tokenize(text)
        return text_tokens

    def txt_process(self, txt):
        with torch.no_grad():
            text_tokens = self.tokenizer(txt).to(self.device)
            self.token_max = text_tokens.argmax(dim=-1)
            x = self.encoder.token_embedding(text_tokens.to(self.device)).type(self.dtype)
            x = x + self.encoder.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
        return x, self.token_max

    def img_process(self, img):
        with torch.no_grad():
            if self.cfg['load_base_model'] == 'clip_RN50':
                def stem(x):
                    for conv, bn in [(self.encoder.conv1, self.encoder.bn1), (self.encoder.conv2, self.encoder.bn2),
                                     (self.encoder.conv3, self.encoder.bn3)]:
                        x = self.encoder.relu(bn(conv(x)))
                    x = self.encoder.avgpool(x)
                    return x

                x = img.type(self.encoder.conv1.weight.dtype)
                x = stem(x)
                x = self.encoder.layer1(x)
                x = self.encoder.layer2(x)
                x = self.encoder.layer3(x)
                x = self.encoder.layer4(x)
                # x = self.attnpool(x)
            else:
                x = self.encoder.conv1(img)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                            dtype=x.dtype, device=x.device), x], dim=1)
                # shape = [*, grid ** 2 + 1, width]
                x = x + self.encoder.positional_embedding.to(x.dtype)
                x = self.encoder.ln_pre(x)
                x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def get_feature(self, input, only_vec=True):
        token_max = None
        with torch.no_grad():
            if self.modal == 'txt':
                x, token_max = self.txt_process(input)
            else:
                x = self.img_process(input.type(self.dtype))
                if self.cfg['load_base_model'] == 'clip_RN50':
                    return x
            if self.learn_mode == 'proj':  # only proj [ISIS]
                for layer in self.encoder.transformer.resblocks:
                    x = layer(x)
                x = x.permute(1, 0, 2)
                x = self.encoder.ln_post(x[:, 0, :]) if self.modal == 'img' else self.encoder.ln_final(x).type(
                    self.dtype)
            else:  # grad
                for layer in self.encoder.transformer.resblocks[:-self.target_layer_num]:
                    x = layer(x)
        return x.detach() if only_vec else (x.detach(), token_max)

    def forward(self, z, token_max=None, got_feature=False):
        if not self.learn_mode == 'proj':
            for layer in self.encoder.transformer.resblocks[-self.target_layer_num:]:
                z = layer(z)
            z = z.permute(1, 0, 2)  # LND -> NLD
            z = self.encoder.ln_post(z[:, 0, :]) if self.modal == 'img' else self.encoder.ln_final(z).type(self.dtype)
            if got_feature:
                return z
        if self.modal == 'img':
            if self.cfg['load_base_model'] == 'clip_RN50':
                z = z.reshape(z.shape[0], z.shape[1], z.shape[2] * z.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
                z = torch.cat([z.mean(dim=0, keepdim=True), z], dim=0)  # (HW+1)NC
                z = z + self.encoder.attnpool.positional_embedding[:, None, :].to(z.dtype)  # (HW+1)NC
                z, _ = self.custom_multi_head_attention(z)
                out = F.linear(z, self.encoder.attnpool.c_proj.weight, self.encoder.attnpool.c_proj.bias)
                out = out[0]
            else:
                out = z @ self.encoder.proj
        else:
            if token_max is not None:
                out = z[torch.arange(z.shape[0]), token_max] @ self.encoder.text_projection
            else:
                out = z[torch.arange(z.shape[0]), self.token_max] @ self.encoder.text_projection
        out = out / out.norm(dim=-1, keepdim=True)
        return out

    def txt_inference(self, txt):
        self.encoder.eval()
        text_tokens = self.tokenizer(txt)
        with torch.no_grad():
            text_tokens = text_tokens.to(self.device)
            text_embeddings = self.encoder.encode_text(text_tokens).float()
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def get_embedding(self, txt):
        self.encoder.eval()
        text_tokens = self.tokenizer(txt)
        with torch.no_grad():
            text_tokens = text_tokens.to(self.device)
            text_embeddings = self.encoder.encode_text(text_tokens).float()
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def img_inference(self, img):
        self.encoder.eval()
        with torch.no_grad():
            img_embeddings = self.encoder(img.type(self.dtype)).float()
            img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        return img_embeddings

    def encod(self, img):
        self.encoder.eval()
        with torch.no_grad():
            img_embeddings = self.encoder(img.type(self.dtype)).float()
            img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        return img_embeddings

    def custom_multi_head_attention(self, x):
        model = self.encoder.attnpool
        num_heads = model.num_heads
        in_proj_bias = torch.cat([model.q_proj.bias, model.k_proj.bias, model.v_proj.bias])
        q_proj_weight = model.q_proj.weight
        k_proj_weight = model.k_proj.weight
        v_proj_weight = model.v_proj.weight
        tgt_len, bsz, embed_dim = x.size()
        scaling = float(embed_dim // num_heads) ** -0.5

        q = F.linear(x, q_proj_weight, in_proj_bias[:embed_dim])
        k = F.linear(x, k_proj_weight, in_proj_bias[embed_dim:2 * embed_dim])
        v = F.linear(x, v_proj_weight, in_proj_bias[2 * embed_dim:])
        head_dim = embed_dim // num_heads
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        attn_output_weights = F.softmax(torch.bmm(q * scaling, k.transpose(1, 2)), dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        return attn_output, None


def save_to_hdf5(filename, dataloader, model, device, batch_size=16):
    import h5py
    with h5py.File(filename, "w") as f:
        first_batch, dataset = True, None
        with torch.no_grad():
            for batch_x, _, _ in dataloader:
                z = model.get_feature(batch_x.to(device)).cpu()
                if first_batch:
                    dataset = f.create_dataset(
                        "data", shape=(0, *z.shape[1:]), maxshape=(None, *z.shape[1:]),
                        dtype="float32", compression="gzip", chunks=(batch_size, *z.shape[1:]))
                    first_batch = False
                dataset.resize((dataset.shape[0] + z.shape[0]), axis=0)
                dataset[-z.shape[0]:] = z.numpy()


def load_hdf5_in_batches(filename, device, batch_size=16):
    import h5py
    with h5py.File(filename, "r") as f:
        dataset = f["data"]
        num_samples = dataset.shape[0]
        for i in range(0, num_samples, batch_size):
            batch = torch.tensor(dataset[i: i + batch_size]).to(device)
            yield batch
