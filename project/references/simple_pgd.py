from copy import deepcopy as dc

import torch
import torch.nn.functional as F


class PGD():  # (self.cfg, self.att_bnd, self.att_stp, self.att_itr, device, self.att_mode, bias_txts, debias_txt, self.l_scale, self.P)
    def __init__(self, cfg, bound, step, iters, device, att_mode, bias_txt, debias_txt, l_scale, bias_cb=None,
                 mix_cb=None):
        self.bound, self.step, self.iter, self.cfg = bound, step, iters, cfg
        self.device, self.mode, self.l_scale = device, att_mode, l_scale
        self.bias_txt = bias_txt
        self.debias_txt = debias_txt
        self.bias_cb = bias_cb
        self.mix_cb = mix_cb
        if bias_cb is not None:
            self.bias_len = len(bias_cb)
        if mix_cb is not None:
            self.mix_len = len(mix_cb)
        torch.manual_seed(cfg.seed)
        self.my_pgd_set()
        self.sub = torch.tensor(0, device=self.device).half()

    def my_pgd_set(self):
        self.norm, self.rand, self.discrete = False, True, True

    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        return x_adv.clone().detach().requires_grad_(True)

    def perturb_bafa(self, z, target_model, target_y=None, mix=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        batch = z_adv.size(0)
        # if self.rand:
        #     rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        #     rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        #     z_adv = self.clamper(self.inverse_normalize(z_adv) + rand_perturb, self.inverse_normalize(z_nat),
        #                          bound=self.bound, inverse_normalized=True)  # .half()
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        # # out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)  <- check
        # bias_logit = self.l_scale * out_feat.float() @ self.bias_txt.t()
        # debias_logit = self.l_scale * out_feat.float() @ self.debias_txt.t()
        # bias_predict, debias_predict = torch.argmax(bias_logit, dim=-1), torch.argmax(debias_logit, dim=-1)
        for i in range(self.iter):  # TODO argmax need to debias Y. So before argmax please use P0 Sigma-1
            adv_feat = target_model(z_adv.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            if self.cfg.att_mode == 'bafa':
                loss = F.cross_entropy(adv_bias, target_y) - F.cross_entropy(adv_debias, target_y)
            elif self.cfg.att_mode == 'txt_guidance':
                logits = torch.stack([
                    (adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),  # land | water
                ]).view(batch, -1)
                loss = F.cross_entropy(logits, target_y)  # landbird is landbackground, waterbird is water background
                loss -= F.mse_loss(ori_feat, adv_debias) * 10 / self.l_scale
            else:
                loss = F.cross_entropy(adv_bias, target_y)
            # loss = F.cross_entropy(adv_bias, target_y) - F.cross_entropy(adv_debias, target_y)
            loss.backward(retain_graph=True)

            grad_sign = z_adv.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv = self.clamper(z_adv_new, z_nat, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv.grad = None
        return z_adv.detach().to(self.device).half()

    def perturb_bafa_nolabel(self, z, target_model):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        # if self.rand:
        #     rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        #     rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        #     z_adv = self.clamper(self.inverse_normalize(z_adv) + rand_perturb, self.inverse_normalize(z_nat),
        #                          bound=self.bound, inverse_normalized=True)  # .half()

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            batch = adv_feat.size(0)
            label = torch.zeros(batch).type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        for i in range(self.iter):
            adv_feat = target_model(z_adv2.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            label = torch.ones(batch).type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            loss.backward(retain_graph=True)

            grad_sign = z_adv2.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv2) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv2 = self.clamper(z_adv_new, z_nat2, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv2.grad = None

        return z_adv1.detach().to(self.device).half(), z_adv2.detach().to(self.device).half()

    def perturb_bafa_nolabel_mix(self, z, target_model, mix=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        batch = ori_feat.size(0)
        label = torch.zeros(batch).type(torch.LongTensor).to(self.device)

        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            if mix is not None:
                logits = torch.stack([
                    (adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),  # land | water
                ]).view(batch, -1)
                entropy_loss = F.cross_entropy(logits, label)
                loss += entropy_loss
                loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        label = torch.ones(batch).type(torch.LongTensor).to(self.device)
        for i in range(self.iter):
            adv_feat = target_model(z_adv2.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            if mix is not None:
                logits = torch.stack([
                    (adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),  # land | water
                ]).view(batch, -1)
                loss += F.cross_entropy(logits, label)
                loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            loss.backward(retain_graph=True)

            grad_sign = z_adv2.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv2) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv2 = self.clamper(z_adv_new, z_nat2, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv2.grad = None

        return z_adv1.detach().to(self.device).half(), z_adv2.detach().to(self.device).half()

    def perturb_bafa_nolabel_use_predict(self, z, target_model, mix=None, batch_y=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        batch = ori_feat.size(0)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            # loss = F.cross_entropy(adv_bias, batch_y) - F.cross_entropy(adv_debias, batch_y)
            if mix is not None:
                logits = torch.stack([(adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),
                                      ]).view(batch, -1)  # land | water
                loss = F.cross_entropy(logits, batch_y)
            loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def perturb_bafa_nolabel_label(self, z, target_model, label, bias_label=None, keep=1):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        ori_loss = F.cross_entropy(ori_feat, label)
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            label = label.type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            adv_bias_txt = self.l_scale * adv_feat @ self.bias_cb.T
            adv_bias_logit = torch.stack(
                (adv_bias_txt[:, :self.bias_len // 2].max(1)[0], adv_bias_txt[:, self.bias_len // 2:].max(1)[0]), dim=1)
            ce_loss = F.cross_entropy(adv_bias_logit, bias_label)
            loss = ce_loss * 0.1 + F.cross_entropy(adv_bias, label) * 0.9 - keep * torch.abs(
                F.cross_entropy(adv_debias, label) - ori_loss)
            # loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            # print(round(float(ce_loss), 4),round(float(F.cross_entropy(adv_bias, label) ), 4), round(float(F.cross_entropy(adv_debias, label) - ori_loss), 4))
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bdfa_mix(self, z, target_model, label, mix_label=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        ori_loss = F.cross_entropy(ori_feat, label)
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            label = label.type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            adv_mix_txt = self.l_scale * adv_feat @ self.mix_cb.T
            adv_mix_logit = torch.where(
                (mix_label // 2).unsqueeze(1) == 0,
                torch.stack((adv_mix_txt[:, :self.mix_len // 4].max(1)[0],
                             adv_mix_txt[:, self.mix_len // 4:self.mix_len // 2].max(1)[0]), dim=1),
                torch.stack((adv_mix_txt[:, self.mix_len // 2:self.mix_len * 3 // 4].max(1)[0],
                             adv_mix_txt[:, self.mix_len * 3 // 4:].max(1)[0]), dim=1))
            ce_loss = F.cross_entropy(adv_mix_logit, mix_label % 2)
            loss = ce_loss * 0.05 + F.cross_entropy(adv_bias, label) * 0.95 - torch.abs(
                F.cross_entropy(adv_debias, label) - ori_loss)

            # loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            # print(round(float(ce_loss), 4), round(float(F.cross_entropy(adv_bias, label)), 4),
            #       round(float(F.cross_entropy(adv_debias, label) - ori_loss), 4))
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bdfa_bm(self, z, target_model, label, label_type='base'):
        target_model.zero_grad()
        label = label.type(torch.LongTensor).to(self.device)
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)  # img emb
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        bias_label = torch.argmax(self.l_scale * out_feat @ self.bias_txt.T, dim=-1)
        ori_loss = F.cross_entropy(ori_feat, label)
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            # ce_loss = F.cross_entropy(adv_bias, label) if label_type == 'base' else F.cross_entropy(adv_bias, bias_label)
            # preserve_loss = torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)  # be lower
            # loss = ce_loss - preserve_loss
            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            loss.backward(retain_graph=True)
            # print('bm : ', round(float(ce_loss), 4), round(float(preserve_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bdfa_tg(self, z, target_model, label, label_type='base'):
        target_model.zero_grad()
        label = label.type(torch.LongTensor).to(self.device)
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)  # img emb
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T
        bias_logit_mean = torch.stack(
            (bias_feat[:, :self.bias_len // 2].mean(1), bias_feat[:, self.bias_len // 2:].mean(1)), dim=1)
        bias_logit_max = torch.stack(
            (bias_feat[:, :self.bias_len // 2].max(1)[0], bias_feat[:, self.bias_len // 2:].max(1)[0]), dim=1)
        bias_label = torch.argmax(bias_logit_mean, dim=-1)
        bias_label_max = torch.argmax(bias_logit_max, dim=-1)
        ori_loss = F.cross_entropy(ori_feat, label)  # original binary entropy
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            adv_bias_txt = self.l_scale * adv_feat @ self.bias_cb.T
            adv_bias_logit = torch.stack(
                (adv_bias_txt[:, :self.bias_len // 2].mean(1), adv_bias_txt[:, self.bias_len // 2:].mean(1)), dim=1)
            if label_type == 'base':
                ce_loss = F.cross_entropy(adv_bias_logit, label)  # be bigger
            elif label_type == 'max':
                ce_loss = F.cross_entropy(adv_bias_logit, bias_label_max)
            else:
                ce_loss = F.cross_entropy(adv_bias_logit, bias_label)
            preserve_loss = torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)  # be lower
            loss = ce_loss - preserve_loss
            loss.backward(retain_graph=True)

            # print('tg : ', round(float(ce_loss), 4), round(float(preserve_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bdfa_tg_t(self, z, target_model, keep=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        with torch.no_grad():
            out_feat = target_model(z)
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T
        bias_logit = torch.stack(
            (bias_feat[:, :self.bias_len // 2].max(1)[0], bias_feat[:, self.bias_len // 2:].max(1)[0]), dim=1)
        keep_ = self.l_scale * out_feat @ keep[0] if keep is not None else None
        cls_logit_d = self.l_scale * out_feat @ self.debias_txt.T
        bias_predict = torch.argmax(bias_logit, dim=-1)
        z_adv1, z_nat1 = dc(z_adv), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            bias_feat = self.l_scale * adv_feat @ self.bias_cb.T
            adv_bias = torch.stack(
                (bias_feat[:, :self.bias_len // 2].max(1)[0], bias_feat[:, self.bias_len // 2:].max(1)[0]), dim=1)
            bias_loss = F.cross_entropy(adv_bias, bias_predict)
            adv_cls_d = self.l_scale * adv_feat @ self.debias_txt.T
            keep_loss = F.mse_loss(adv_cls_d, cls_logit_d)
            if keep is not None:
                keep_loss += F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_)
            loss = bias_loss - keep_loss
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    loss.float().backward(retain_graph=True)
            except RuntimeError:
                print(target_model.parameters().__next__().grad)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        return z_adv1.detach().to(self.device).half()

    def bdfa_bias(self, z, target_model, label, label_type='base'):
        target_model.zero_grad()
        label = label.type(torch.LongTensor).to(self.device)
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)  # img emb
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T
        bias_logit = torch.stack(
            (bias_feat[:, :self.bias_len // 2].mean(1), bias_feat[:, self.bias_len // 2:].mean(1)), dim=1)
        bias_label = torch.argmax(bias_logit, dim=-1)

        bias_m_label = torch.argmax(self.l_scale * out_feat @ self.bias_txt.T, dim=-1)
        ori_loss = F.cross_entropy(ori_feat, label)  # original binary entropy
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            adv_bias_txt = self.l_scale * adv_feat @ self.bias_cb.T
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_bias_logit = torch.stack(
                (adv_bias_txt[:, :self.bias_len // 2].mean(1), adv_bias_txt[:, self.bias_len // 2:].mean(1)), dim=1)
            if label_type == 'base':
                ce_loss_bt = F.cross_entropy(adv_bias_logit, label) * 0.2
                ce_loss_bm = F.cross_entropy(adv_bias, label) * 0.8
                ce_loss = ce_loss_bt + ce_loss_bm
            else:
                ce_loss_bt = F.cross_entropy(adv_bias_logit, bias_label) * 0.2
                ce_loss_bm = F.cross_entropy(adv_bias, bias_m_label) * 0.8
                ce_loss = ce_loss_bt + ce_loss_bm
            preserve_loss = torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)  # be lower
            loss = ce_loss - preserve_loss
            loss.backward(retain_graph=True)

            # print('bias : ', round(float(ce_loss_bt), 4), round(float(ce_loss_bm), 4), round(float(preserve_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
            if not i:
                track_ce_bt, track_ce_bm, track_pr = ce_loss_bt.item(), ce_loss_bm.item(), preserve_loss.item()
        track_ce_bt, track_ce_bm, track_pr = ce_loss_bt.item() - track_ce_bt, ce_loss_bm.item() - track_ce_bm, preserve_loss.item() - track_pr
        # print(track_ce_bt, track_ce_bm, track_pr)
        return z_adv1.detach().to(self.device).half()

    def bdfa_bias_k(self, z, target_model, label, keep, label_type='base'):
        target_model.zero_grad()
        label = label.type(torch.LongTensor).to(self.device)
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)  # img emb
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)

        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T
        keep_feat = self.l_scale * out_feat @ keep[0]
        bias_logit = torch.stack(
            (bias_feat[:, :self.bias_len // 2].mean(1), bias_feat[:, self.bias_len // 2:].mean(1)), dim=1)
        bias_label = torch.argmax(bias_logit, dim=-1)
        bias_m_label = torch.argmax(self.l_scale * out_feat @ self.bias_txt.T, dim=-1)

        ori_loss = F.cross_entropy(ori_feat, label)  # original binary entropy
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)

        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            adv_bias_txt = self.l_scale * adv_feat @ self.bias_cb.T
            adv_keep = self.l_scale * adv_feat @ keep[0]

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_bias_logit = torch.stack(
                (adv_bias_txt[:, :self.bias_len // 2].mean(1), adv_bias_txt[:, self.bias_len // 2:].mean(1)), dim=1)
            if label_type == 'base':
                ce_loss_bt = F.cross_entropy(adv_bias_logit, label) * 0.2
                ce_loss_bm = F.cross_entropy(adv_bias, label) * 0.8
                ce_loss = ce_loss_bt + ce_loss_bm
            else:
                ce_loss_bt = F.cross_entropy(adv_bias_logit, bias_label) * 0.2
                ce_loss_bm = F.cross_entropy(adv_bias, bias_m_label) * 0.8
                ce_loss = ce_loss_bt + ce_loss_bm
            # preserve_loss1 = torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)
            preserve_loss1 = F.mse_loss(ori_feat, adv_debias)
            preserve_loss2 = F.mse_loss(keep_feat, adv_keep)  # * 10  # be lower
            loss = ce_loss - (preserve_loss1 + preserve_loss2) * 10
            loss.backward(retain_graph=True)

            # print('bias : ', round(float(ce_loss_bt), 4), round(float(ce_loss_bm), 4), round(float(preserve_loss1), 4),
            #       round(float(preserve_loss2), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        return z_adv1.detach().to(self.device).half()

    def bdfa_bias_k_bw(self, z, target_model, keep):
        target_model.zero_grad()
        # label = label.type(torch.LongTensor).to(self.device)
        adv_loss, adv_loss_keep, keep_loss = self.sub, self.sub, self.sub
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        z_adv = self.clamper(z_adv + rand_perturb, z_nat, bound=self.bound, inverse_normalized=True)
        # img emb
        with torch.no_grad():
            out_feat = target_model(z)
        cls_d = self.l_scale * out_feat @ self.debias_txt.T
        cls_b = self.l_scale * out_feat @ self.bias_txt.T
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T

        keep_ = self.l_scale * out_feat @ keep[0]
        debias_cls_predict = torch.argmax(cls_d, dim=-1)
        ori_loss = F.cross_entropy(cls_d, debias_cls_predict)

        case = 3
        if case == 3:
            n_bias_logit = torch.stack(
                [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in range(4)],
                dim=1).view(-1, 2)
            bias_predict = torch.argmax(n_bias_logit, dim=-1)
        else:
            bias_logit = torch.stack(
                (bias_feat[:, :self.bias_len // 2].max(1)[0], bias_feat[:, self.bias_len // 2:].max(1)[0]), dim=1)
            bias_predict = torch.argmax(bias_logit, dim=-1)

        bias_cls_predict = torch.argmax(cls_b, dim=-1)
        set_ = debias_cls_predict == bias_cls_predict
        # ori_loss = F.cross_entropy(cls_logit_d[set_], bias_cls_predict[set_])
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_cls_d = self.l_scale * adv_feat @ self.debias_txt.T
            adv_cls_b = self.l_scale * adv_feat @ self.bias_txt.T
            bias_feat = self.l_scale * adv_feat @ self.bias_cb.T

            if case == 3:
                adv_bias = torch.stack(
                    [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in
                     range(4)],
                    dim=1).view(-1, 2)
            else:
                adv_bias = torch.stack(
                    (bias_feat[:, :self.bias_len // 2].max(1)[0], bias_feat[:, self.bias_len // 2:].max(1)[0]), dim=1)
            if case == 1:
                if any(~set_):
                    adv_loss = F.cross_entropy(adv_cls_b[~set_], bias_cls_predict[~set_]) * sum(~set_) / len(z_adv1)
                    adv_loss_keep = F.cross_entropy(adv_cls_d[~set_], debias_cls_predict[~set_]) * sum(~set_) / len(
                        z_adv1)
                # adv_loss = F.cross_entropy(adv_cls_b[~set_], bias_cls_predict[~set_]) * sum(~set_)/len(z_adv1)
            elif case == 2:
                if any(set_):
                    adv_loss = F.cross_entropy(adv_cls_b[set_], bias_cls_predict[set_]) * sum(set_) / len(z_adv1)
                    adv_loss_keep = F.cross_entropy(adv_cls_d[set_], debias_cls_predict[set_]) * sum(set_) / len(
                        z_adv1)
            else:
                adv_loss = F.cross_entropy(adv_cls_b, bias_cls_predict)
                adv_loss_keep = F.cross_entropy(adv_cls_d, debias_cls_predict)
                # adv_loss_keep += F.mse_loss(adv_cls_d, cls_d)
            bias_loss = F.cross_entropy(adv_bias, bias_predict)
            keep_loss = F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_)
            loss = (adv_loss - adv_loss_keep + bias_loss - keep_loss) / 2
            loss.backward(retain_graph=True)

            # print(round(float(adv_loss), 4), round(float(adv_loss_keep), 4), ' |  ', round(float(bias_loss), 4),
            #       round(float(keep_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bafa_mix(self, z, target_model, keep):
        target_model.zero_grad()
        adv_loss, adv_loss_keep, keep_loss = self.sub, self.sub, self.sub
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        z_adv = self.clamper(z_adv + rand_perturb, z_nat, bound=self.bound, inverse_normalized=True)

        with torch.no_grad():
            out_feat = target_model(z)
        cls_d = self.l_scale * out_feat @ self.debias_txt.T
        cls_b = self.l_scale * out_feat @ self.bias_txt.T
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T

        keep_ = self.l_scale * out_feat @ keep[0]
        debias_cls_predict = torch.argmax(cls_d, dim=-1)
        ori_loss = F.cross_entropy(cls_d, debias_cls_predict)

        n_bias_logit = torch.stack(
            [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in range(4)],
            dim=1).view(-1, 2)
        bias_predict = torch.argmax(n_bias_logit, dim=-1)

        bias_cls_predict = torch.argmax(cls_b, dim=-1)
        set_ = debias_cls_predict == bias_cls_predict
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_cls_d = self.l_scale * adv_feat @ self.debias_txt.T
            adv_cls_b = self.l_scale * adv_feat @ self.bias_txt.T
            bias_feat = self.l_scale * adv_feat @ self.bias_cb.T

            adv_bias = torch.stack(
                [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in range(4)],
                dim=1).view(-1, 2)

            adv_loss = F.cross_entropy(adv_cls_b, bias_cls_predict)
            adv_loss_keep = F.cross_entropy(adv_cls_d, debias_cls_predict)
            bias_loss = F.cross_entropy(adv_bias, bias_predict)
            keep_loss = F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_)
            loss = (adv_loss - adv_loss_keep) / 2 + (bias_loss - keep_loss)
            loss.backward(retain_graph=True)

            # print(round(float(adv_loss), 4), round(float(adv_loss_keep), 4), ' |  ', round(float(bias_loss), 4),
            #       round(float(keep_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bafa_tm(self, z, target_model, keep):

        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        with torch.no_grad():
            out_feat = target_model(z)
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T

        keep_ = self.l_scale * out_feat @ keep[0]
        n_bias_logit = torch.stack(
            [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in range(4)],
            dim=1).view(-1, 2)
        bias_predict = torch.argmax(n_bias_logit, dim=-1)

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            bias_feat = self.l_scale * adv_feat @ self.bias_cb.T

            adv_bias = torch.stack(
                [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in range(4)],
                dim=1).view(-1, 2)
            bias_loss = F.cross_entropy(adv_bias, bias_predict)
            keep_loss = F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_)
            loss = (bias_loss - keep_loss * 10)
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv1 + grad_sign * self.step, z_nat1, bound=self.bound)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        return z_adv1.detach().to(self.device).half()

    def bafa_tm_set(self, z, target_model, keep, att_bnd=None, att_stp=None, att_itr=None, samples_num=1):
        att_bnd = att_bnd if att_bnd is not None else self.bound
        att_stp = att_stp if att_stp is not None else self.step
        att_itr = att_itr if att_itr is not None else self.iter

        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        with torch.no_grad():
            out_feat = target_model(z)
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T
        keep_ = self.l_scale * out_feat @ keep[0]
        n_bias_logit = torch.stack(
            [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in range(4)],
            dim=1).view(-1, 2)

        z_adv_list, z_adv_list2 = [], []
        for num in range(samples_num):
            target_model.zero_grad()
            if num:
                z_new = clamper(z_adv + rand_perturb, z_nat, bound=att_bnd)
            z_adv1 = dc(z_adv) if not num else (z_new + rand_perturb).detach().requires_grad_(True)
            z_nat1 = dc(z_nat) if not num else (z_new + rand_perturb).detach()
            for i in range(att_itr):
                adv_feat = target_model(z_adv1.half())
                bias_feat = self.l_scale * adv_feat @ self.bias_cb.T
                adv_bias = torch.stack(
                    [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in
                     range(4)],
                    dim=1).view(-1, 2)
                bias_loss = F.cross_entropy(adv_bias, torch.tensor([0] * len(n_bias_logit), device='cuda'))
                keep_loss = F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_) * 10
                loss = bias_loss - keep_loss

                loss.backward(retain_graph=True)
                z_adv1 = self.clamper(z_adv1 + z_adv1.grad.data.detach().sign() * att_stp, z_nat1, bound=att_bnd)
                target_model.zero_grad()
                z_adv1.grad = None
            z_adv_list.append(z_adv1.detach().to(self.device))

            target_model.zero_grad()
            z_adv1 = dc(z_adv) if not num else (z_new + rand_perturb).detach().requires_grad_(True)
            z_nat1 = dc(z_nat) if not num else (z_new + rand_perturb).detach()
            for i in range(att_itr):
                adv_feat = target_model(z_adv1.half())
                bias_feat = self.l_scale * adv_feat @ self.bias_cb.T
                adv_bias = torch.stack(
                    [torch.cat((bias_feat[:, i].unsqueeze(1), bias_feat[:, i + 4].unsqueeze(1)), dim=1) for i in
                     range(4)],
                    dim=1).view(-1, 2)
                bias_loss = F.cross_entropy(adv_bias, torch.tensor([1] * len(n_bias_logit), device='cuda'))
                keep_loss = F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_) * 10
                loss = bias_loss - keep_loss
                loss.backward(retain_graph=True)
                z_adv1 = self.clamper(z_adv1 + z_adv1.grad.data.detach().sign() * att_stp, z_nat1, bound=att_bnd)
                target_model.zero_grad()
                z_adv1.grad = None
            z_adv_list2.append(z_adv1.detach().to(self.device))
            rand_perturb_dist = torch.distributions.uniform.Uniform(-att_bnd, att_bnd)
            rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        re_adv1 = torch.cat(z_adv_list, dim=0).half().detach().to(self.device)
        re_adv2 = torch.cat(z_adv_list2, dim=0).half().detach().to(self.device)
        return re_adv1, re_adv2

    def bafa_bm(self, z, target_model, keep):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        # rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        # rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        # z_adv = self.clamper(z_adv + rand_perturb, z_nat, bound=self.bound, inverse_normalized=True)

        with torch.no_grad():
            out_feat = target_model(z)
        cls_d = self.l_scale * out_feat @ self.debias_txt.T
        cls_b = self.l_scale * out_feat @ self.bias_txt.T

        # keep_ = self.l_scale * out_feat @ keep[0]
        debias_cls_predict = torch.argmax(cls_d, dim=-1)
        bias_cls_predict = torch.argmax(cls_b, dim=-1)

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_cls_d = self.l_scale * adv_feat @ self.debias_txt.T
            adv_cls_b = self.l_scale * adv_feat @ self.bias_txt.T

            adv_loss = F.cross_entropy(adv_cls_b, bias_cls_predict)
            adv_loss_keep = F.cross_entropy(adv_cls_d, debias_cls_predict)
            loss = adv_loss - adv_loss_keep
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        return z_adv1.detach().to(self.device).half()

    def bafa_bm_iccv(self, z, target_model, step=1e-4, bound=0.5, iter=10):
        #  1e-3 0.1, , 20
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        with torch.no_grad():
            out_feat = target_model(z)
        cls_d = self.l_scale * out_feat @ self.debias_txt.T
        cls_b = self.l_scale * out_feat @ self.bias_txt.T

        # keep_ = self.l_scale * out_feat @ keep[0]
        debias_cls_predict = torch.argmax(cls_d, dim=-1)
        bias_cls_predict = torch.argmax(cls_b, dim=-1)

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(iter):
            adv_feat = target_model(z_adv1.half())
            adv_cls_d = self.l_scale * adv_feat @ self.debias_txt.T
            adv_cls_b = self.l_scale * adv_feat @ self.bias_txt.T

            adv_loss = F.cross_entropy(adv_cls_b, bias_cls_predict)
            adv_loss_keep = F.cross_entropy(adv_cls_d, debias_cls_predict)
            loss = adv_loss - adv_loss_keep
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = self.clamper(z_adv1 + grad_sign * step, z_nat1, bound=bound)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        return z_adv1.detach().to(self.device).half()

    def bdfa_bm_t(self, z, target_model, keep):
        target_model.zero_grad()
        adv_loss, adv_loss_keep, keep_loss = self.sub, self.sub, self.sub
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)

        cls_logit_d = self.l_scale * out_feat @ self.debias_txt.T
        cls_logit_b = self.l_scale * out_feat @ self.bias_txt.T
        keep_ = self.l_scale * out_feat @ keep[0]
        debias_cls_predict = torch.argmax(cls_logit_d, dim=-1)
        bias_cls_predict = torch.argmax(cls_logit_b, dim=-1)
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_cls_d = self.l_scale * adv_feat @ self.debias_txt.T
            adv_cls_b = self.l_scale * adv_feat @ self.bias_txt.T
            adv_loss = F.cross_entropy(adv_cls_b, debias_cls_predict)
            adv_loss_keep = F.cross_entropy(adv_cls_d, debias_cls_predict)
            keep_loss = F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_)
            loss = adv_loss - adv_loss_keep - keep_loss * 0.1
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bdfa_bias_tg(self, z, target_model, keep):
        target_model.zero_grad()
        # label = label.type(torch.LongTensor).to(self.device)
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)  # img emb

        cls_logit_d = self.l_scale * out_feat @ self.debias_txt.T
        cls_logit_b = self.l_scale * out_feat @ self.bias_txt.T
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T
        bias_logit = torch.stack(
            (bias_feat[:, :self.bias_len // 2].max(1)[0], bias_feat[:, self.bias_len // 2:].max(1)[0]), dim=1)
        keep_ = self.l_scale * out_feat @ keep[0]

        debias_cls_predict = torch.argmax(cls_logit_d, dim=-1)
        bias_cls_predict = torch.argmax(cls_logit_b, dim=-1)
        bias_predict = torch.argmax(bias_logit, dim=-1)
        set_ = debias_cls_predict == bias_cls_predict
        ori_loss = F.cross_entropy(cls_logit_d[set_], bias_cls_predict[set_])
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())

            adv_cls_d = self.l_scale * adv_feat @ self.debias_txt.T
            adv_cls_b = self.l_scale * adv_feat @ self.bias_txt.T
            bias_feat = self.l_scale * adv_feat @ self.bias_cb.T
            adv_bias = torch.stack(
                (bias_feat[:, :self.bias_len // 2].max(1)[0], bias_feat[:, self.bias_len // 2:].max(1)[0]), dim=1)

            adv_loss = F.cross_entropy(adv_cls_b[set_], bias_cls_predict[set_])
            adv_loss_keep = F.mse_loss(adv_cls_d[set_], cls_logit_d[set_])
            bias_loss = F.cross_entropy(adv_bias, bias_predict)
            keep_loss = F.mse_loss(self.l_scale * adv_feat @ keep[0], keep_)
            loss = adv_loss - adv_loss_keep + bias_loss - keep_loss
            loss.backward(retain_graph=True)

            print(round(float(adv_loss), 4), round(float(adv_loss_keep), 4), ' |  ', round(float(bias_loss), 4),
                  round(float(keep_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bdfa_all(self, z, target_model, label, label_type='base'):
        target_model.zero_grad()
        label = label.type(torch.LongTensor).to(self.device)
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)  # img emb
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        bias_feat = self.l_scale * out_feat @ self.bias_cb.T
        bias_logit = torch.stack(
            (bias_feat[:, :self.bias_len // 2].mean(1), bias_feat[:, self.bias_len // 2:].mean(1)), dim=1)
        bias_label = torch.argmax(bias_logit, dim=-1)
        mix_feat = self.l_scale * out_feat @ self.mix_cb.T
        mix_label = torch.argmax(mix_feat, dim=-1) // (self.mix_len // 4)

        bias_m_label = torch.argmax(self.l_scale * out_feat @ self.bias_txt.T, dim=-1)
        ori_loss = F.cross_entropy(ori_feat, label)  # original binary entropy
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            adv_bias_txt = self.l_scale * adv_feat @ self.bias_cb.T
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_bias_logit = torch.stack(
                (adv_bias_txt[:, :self.bias_len // 2].mean(1), adv_bias_txt[:, self.bias_len // 2:].mean(1)), dim=1)
            adv_mix_txt = self.l_scale * adv_feat @ self.mix_cb.T
            if label_type == 'base':
                ce_loss_bt = F.cross_entropy(adv_bias_logit, label) * 0.2
                ce_loss_bm = F.cross_entropy(adv_bias, label) * 0.6
                adv_mix_logit = torch.where(label.unsqueeze(1) == 0,
                                            torch.stack((adv_mix_txt[:, :self.mix_len // 4].max(1)[0],
                                                         adv_mix_txt[:, self.mix_len // 4:self.mix_len // 2].max(1)[0]),
                                                        dim=1),
                                            torch.stack(
                                                (adv_mix_txt[:, self.mix_len // 2:self.mix_len * 3 // 4].max(1)[0],
                                                 adv_mix_txt[:, self.mix_len * 3 // 4:].max(1)[0]), dim=1))
                ce_loss_mix = F.cross_entropy(adv_mix_logit, label) * 0.2
                ce_loss = ce_loss_bt + ce_loss_bm + ce_loss_mix
            else:
                ce_loss_bt = F.cross_entropy(adv_bias_logit, bias_label) * 0.2
                ce_loss_bm = F.cross_entropy(adv_bias, bias_m_label) * 0.6
                adv_mix_logit = torch.where((mix_label // 2).unsqueeze(1) == 0,
                                            torch.stack((adv_mix_txt[:, :self.mix_len // 4].max(1)[0],
                                                         adv_mix_txt[:, self.mix_len // 4:self.mix_len // 2].max(1)[0]),
                                                        dim=1),
                                            torch.stack(
                                                (adv_mix_txt[:, self.mix_len // 2:self.mix_len * 3 // 4].max(1)[0],
                                                 adv_mix_txt[:, self.mix_len * 3 // 4:].max(1)[0]), dim=1))
                ce_loss_mix = F.cross_entropy(adv_mix_logit, mix_label % 2) * 0.2
                ce_loss = ce_loss_bt + ce_loss_bm + ce_loss_mix
            preserve_loss = torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)  # be lower
            loss = ce_loss - preserve_loss
            loss.backward(retain_graph=True)

            # print('bias : ', round(float(ce_loss_bt), 4), round(float(ce_loss_bm), 4), round(float(preserve_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
            if not i:
                track_ce_bt, track_ce_bm, track_mix, track_pr = ce_loss_bt.item(), ce_loss_bm.item(), ce_loss_mix.item(), preserve_loss.item()
        track_ce_bt, track_ce_bm, track_mix, track_pr = ce_loss_bt.item() - track_ce_bt, ce_loss_bm.item() - track_ce_bm, ce_loss_mix.item() - track_mix, preserve_loss.item() - track_pr
        print(track_ce_bt, track_ce_bm, track_mix, track_pr)
        return z_adv1.detach().to(self.device).half()

    def bdfa_mtg(self, z, target_model, label, label_type='base'):
        target_model.zero_grad()
        label = label.type(torch.LongTensor).to(self.device)
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        out_feat = target_model(z)  # img emb
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        mix_feat = self.l_scale * out_feat @ self.mix_cb.T
        mix_label = torch.argmax(mix_feat, dim=-1) // (self.mix_len // 4)

        ori_loss = F.cross_entropy(ori_feat, label)  # original binary entropy
        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            adv_mix_txt = self.l_scale * adv_feat @ self.mix_cb.T
            if label_type == 'base':
                adv_mix_logit = torch.where(label.unsqueeze(1) == 0,
                                            torch.stack((adv_mix_txt[:, :self.mix_len // 4].max(1)[0],
                                                         adv_mix_txt[:, self.mix_len // 4:self.mix_len // 2].max(1)[0]),
                                                        dim=1),
                                            torch.stack(
                                                (adv_mix_txt[:, self.mix_len // 2:self.mix_len * 3 // 4].max(1)[0],
                                                 adv_mix_txt[:, self.mix_len * 3 // 4:].max(1)[0]), dim=1))
                ce_loss = F.cross_entropy(adv_mix_logit, label)  # be bigger
            else:
                adv_mix_logit = torch.where((mix_label // 2).unsqueeze(1) == 0,
                                            torch.stack((adv_mix_txt[:, :self.mix_len // 4].max(1)[0],
                                                         adv_mix_txt[:, self.mix_len // 4:self.mix_len // 2].max(1)[0]),
                                                        dim=1),
                                            torch.stack(
                                                (adv_mix_txt[:, self.mix_len // 2:self.mix_len * 3 // 4].max(1)[0],
                                                 adv_mix_txt[:, self.mix_len * 3 // 4:].max(1)[0]), dim=1))
                ce_loss = F.cross_entropy(adv_mix_logit, mix_label % 2)
            preserve_loss = torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)  # be lower
            loss = ce_loss - preserve_loss
            loss.backward(retain_graph=True)

            # print('mtg : ', round(float(ce_loss), 4), round(float(preserve_loss), 4))
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def bdfa_new(self, z, target_model, label, bias_txt=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        bias_txt, label = bias_txt.type(torch.LongTensor).to(self.device), label.type(torch.LongTensor).to(self.device)
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_loss = F.cross_entropy(self.l_scale * out_feat @ self.debias_txt.T, label)
        z_adv1, z_adv2, z_adv3, z_nat1, z_nat2, z_nat3 = dc(z_adv), dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat), dc(
            z_nat)
        # tpp tnp
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            label = label.type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            loss = F.cross_entropy(adv_bias, label) - torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)

            loss.backward(retain_graph=True)
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        # tpp tpn
        for i in range(self.iter):
            adv_feat = target_model(z_adv2.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            label = label.type(torch.LongTensor).to(self.device)
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            adv_bias_txt = self.l_scale * adv_feat @ self.bias_cb.T

            adv_bias_logit = torch.stack(
                (adv_bias_txt[:, :self.bias_len // 2].max(1)[0], adv_bias_txt[:, self.bias_len // 2:].max(1)[0]), dim=1)
            ce_loss = F.cross_entropy(adv_bias_logit, bias_txt)
            loss = ce_loss - torch.abs(F.cross_entropy(adv_debias, label) - ori_loss)
            loss.backward(retain_graph=True)
            grad_sign = z_adv2.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv2) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv2 = self.clamper(z_adv_new, z_nat2, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv2.grad = None
        # tpp tnn
        for i in range(self.iter):
            adv_feat = target_model(z_adv3.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            label = label.type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            adv_bias_txt = self.l_scale * adv_feat @ self.bias_cb.T
            adv_bias_logit = torch.stack(
                (adv_bias_txt[:, :self.bias_len // 2].max(1)[0], adv_bias_txt[:, self.bias_len // 2:].max(1)[0]), dim=1)
            ce_loss = F.cross_entropy(adv_bias_logit, bias_txt)
            loss = ce_loss * 0.1 + F.cross_entropy(adv_bias, label) * 0.9 - torch.abs(
                F.cross_entropy(adv_debias, label) - ori_loss)

            loss.backward(retain_graph=True)
            grad_sign = z_adv3.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv3) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv3 = self.clamper(z_adv_new, z_nat3, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv3.grad = None

        return z_adv1.detach().to(self.device).half(), z_adv2.detach().to(self.device).half(), z_adv3.detach().to(
            self.device).half()

    def base_perturb(self, x, y, target_y=None, model=None, bound=None, step=None, iters=None, x_nat=None, device=None,
                     **kwargs):
        criterion = self.CE
        model = model or self.model
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iter
        device = device or self.device

        model.zero_grad()
        if x_nat is None:
            x_nat = self.inverse_normalize(x.detach().clone().to(device))
        else:
            x_nat = self.inverse_normalize(x_nat.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
        for i in range(iters):
            adv_pred = model(x_adv)  # 64 256 -> 64 10
            loss = criterion(adv_pred, y)
            loss.backward(retain_graph=True)

            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step  # a sign( d( L (x_adv)))
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
            model.zero_grad()

        return x_adv.detach().to(device)

    def normalize(self, x):
        return x

    def inverse_normalize(self, x):
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255


def perturb_bafa_txt(z, target_model, mix, t_, test=None):
    target_model.zero_grad()
    device = z.device
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)

    z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
    att_bnd, att_stp, att_itr = 0.1, 5e-4, 10
    with torch.no_grad():
        ori = target_model(z, t_)
    # check = ori @ test[0]
    for i in range(att_itr):
        adv_feat = target_model(z_adv1.half(), t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = torch.stack([
            (adv_feat[0] @ mix[:4].t()).mean(), (adv_feat[0] @ mix[4:].t()).mean(),
            (adv_feat[1] @ mix[:4].t()).mean(), (adv_feat[1] @ mix[4:].t()).mean()
        ]).view(2, 2)
        loss = F.cross_entropy(logits,
                               torch.tensor([0, 1], device='cuda'))  # - F.mse_loss(adv_feat @ test[0], check) * 10
        loss.backward(retain_graph=True)

        grad_sign = z_adv1.grad.data.detach().sign()
        z_adv_new = z_adv1 + grad_sign * att_stp  # a sign( d( L (x_adv)))
        z_adv1 = clamper(z_adv_new, z_nat1, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv1.grad = None
        # print(adv_feat @ test[0], sum(torch.abs(adv_feat @ test[0] - check)))
        # print(adv_feat @ ori.t())
        # print(adv_feat @ test.t())

    return z_adv1.detach().to(device).half()


def get_proj_matrix(embeddings: torch.Tensor):
    device = embeddings.device
    U, S, Vh = torch.linalg.svd(embeddings, full_matrices=False)
    basis = Vh.T
    proj = torch.inverse(basis.T @ basis)  # (n_components, n_components)
    proj = basis @ proj  # (embedding_dim, n_components)
    proj = proj @ basis.T  # (embedding_dim, embedding_dim)
    proj_remove = torch.eye(proj.shape[0], device=device) - proj
    return proj, proj_remove


def perturb_bafa_txt_c(z, target_model, mix, t_, test=None, label=[0, 0], att_bnd=0.4, att_stp=1e-2, att_itr=20):
    target_model.zero_grad()
    device = z.device
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)
    mix = F.normalize(mix, dim=-1)
    # a, b = get_proj_matrix(mix.float())
    z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
    with torch.no_grad():
        ori = target_model(z, t_)
    keep = 100 * ori @ test[0]  # keep info
    for i in range(att_itr):
        # F.normalize(target_model(z_adv1) dim=-1)
        adv_feat = target_model(z_adv1, t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = torch.stack([
            torch.mean(adv_feat[0] @ mix[:len(mix) // 2].t()), torch.mean(adv_feat[0] @ mix[len(mix) // 2:].t()),
            torch.mean(adv_feat[1] @ mix[:len(mix) // 2].t()), torch.mean(adv_feat[1] @ mix[len(mix) // 2:].t())
        ]).view(2, 2)
        loss = F.cross_entropy(100 * logits, torch.tensor(label, device='cuda')) - F.mse_loss(100 * adv_feat @ test[0],
                                                                                              keep)
        loss.backward(retain_graph=True)

        grad_sign = z_adv1.grad.data.detach().sign()
        z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv1.grad = None
        # print(F.cross_entropy(logits, torch.tensor([0, 1], device='cuda')), F.mse_loss(adv_feat @ test[0], keep))
        # print(adv_feat @ test[0], sum(torch.abs(adv_feat @ test[0] - check)))
        # print(logits)
        # print(adv_feat @ test.t())

    return z_adv1.detach().to(device).half()


def perturb_bafa_txt_multi(z, target_model, mix, t_, test=None, label=[0, 0], att_bnd=0.5, att_stp=1e-2, att_itr=20,
                           num_samples=50, bias_model=None):
    target_model.zero_grad()
    device, mix = z.device, F.normalize(mix, dim=-1)
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)
    with torch.no_grad():
        ori = F.normalize(target_model(z.half(), t_), dim=-1)
        bias_feat = F.normalize(bias_model(z.half(), t_), dim=-1)
        keep_anchor = 100 * ori @ test[0]  # keep info
    z_adv_list = []
    for num in range(num_samples):
        z_adv1 = dc(z_adv) if not num else (z_adv + rand_perturb).detach().requires_grad_(True)
        z_nat1 = dc(z_nat) if not num else (z_nat + rand_perturb).detach()
        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)

            logits = torch.stack([
                torch.mean(adv_feat[0] @ mix[:len(mix) // 2].t()), torch.mean(adv_feat[0] @ mix[len(mix) // 2:].t()),
                torch.mean(adv_feat[1] @ mix[:len(mix) // 2].t()), torch.mean(adv_feat[1] @ mix[len(mix) // 2:].t())
            ]).view(2, 2)
            loss = F.cross_entropy(100 * logits, torch.tensor(label, device='cuda')) - F.mse_loss(
                100 * adv_feat @ test[0], keep_anchor)
            # ce_loss = F.relu(F.cross_entropy(100 * adv_feat @ bias_feat.T, torch.tensor(label).to(device)) - F.cross_entropy(100 * adv_feat @ ori.T, torch.tensor(label).to(device)))
            # if abs(ce_loss)>0.1:
            #     print(ce_loss, loss - ce_loss)
            # loss += ce_loss
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        rand_perturb_dist = torch.distributions.uniform.Uniform(-0.5, 0.5)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(device)
        z_adv_list.append(z_adv1.detach().to(device))

    return torch.cat(z_adv_list, dim=1).half().detach().to(device)


def perturb_bafa_txt_multi_all(z, target_model, mix, t_, test=None, att_bnd=0.1, att_stp=1e-2, att_itr=20, random=0.1,
                               num_samples=50, keep_weight=10, type=1):
    target_model.zero_grad()
    device, mix, z_len = z.device, F.normalize(mix, dim=-1), z.size(1)
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)
    if test is not None:
        keep = test[:5]
        with torch.no_grad():
            ori = F.normalize(target_model(z.half(), t_), dim=-1)
            keep_anchor = 100 * ori @ keep.T  # keep info

    z_adv_list, z_adv_list2 = [], []
    for num in range(num_samples):
        if not num:
            z_adv1, z_nat1 = dc(z_adv), dc(z_nat)
        else:
            z_new = clamper(z_adv + rand_perturb, z_nat, bound=att_bnd)
            z_adv1, z_nat1 = z_new.detach().requires_grad_(True), z_new.detach()
        target_model.zero_grad()
        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            att_loss = F.cross_entropy(100 * logits, torch.tensor([0] * (len(mix) // 2) * z_len, device='cuda'))
            if test is not None:
                keep_loss = F.mse_loss(100 * adv_feat @ keep.T, keep_anchor) * keep_weight
            loss = att_loss - keep_loss if test is not None else att_loss
            # print(F.cross_entropy(100 * logits, torch.tensor([0] * (len(mix) // 2) * len(keep_anchor), device='cuda')),
            #       F.mse_loss(100 * adv_feat @ keep.T, keep_anchor) * 10)
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        z_adv_list.append(z_adv1.detach().to(device))
        target_model.zero_grad()
        z_adv1 = dc(z_adv) if not num else z_new.detach().requires_grad_(True)
        z_nat1 = dc(z_nat) if not num else z_new.detach()

        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            att_loss = F.cross_entropy(100 * logits, torch.tensor([1] * (len(mix) // 2) * z_len, device='cuda'))
            if test is not None:
                keep_loss = F.mse_loss(100 * adv_feat @ keep.T, keep_anchor) * keep_weight
            loss = att_loss - keep_loss if test is not None else att_loss
            # print(F.cross_entropy(100 * logits, torch.tensor([1]* (len(mix)//2) * len(keep_anchor), device='cuda')), F.mse_loss(100 * adv_feat @ test.T, keep_anchor))
            loss.backward(retain_graph=True)
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        rand_perturb_dist = torch.distributions.uniform.Uniform(-random, random)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(device)
        z_adv_list2.append(z_adv1.detach().to(device))

    return torch.cat(z_adv_list, dim=1).half().detach().to(device), torch.cat(z_adv_list2, dim=1).half().detach().to(
        device)


def perturb_bafa_txt_multi_test(z, target_model, mix, t_, keep=None, att_bnd=0.1, att_stp=1e-2, att_itr=20, random=0.1,
                               num_samples=50, keep_weight=10, type=1):
    target_model.zero_grad()
    device, mix, z_len = z.device, F.normalize(mix, dim=-1), z.size(1)
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)
    if keep is not None:
        with torch.no_grad():
            ori = F.normalize(target_model(z.half(), t_), dim=-1)
            # keep_anchor = 100 * ori @ keep.T  # keep info

    z_adv_list, z_adv_list2 = [], []
    for num in range(num_samples):
        if not num:
            z_adv1, z_nat1 = dc(z_adv), dc(z_nat)
        else:
            z_new = clamper(z_adv + rand_perturb, z_nat, bound=att_bnd)
            z_adv1, z_nat1 = z_new.detach().requires_grad_(True), z_new.detach()
        target_model.zero_grad()
        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            att_loss = F.cross_entropy(100 * logits, torch.tensor([0] * (len(mix) // 2) * z_len, device='cuda'))
            keep_loss = 100 * ((adv_feat - ori) @ keep.T).pow(2).mean()
            loss = att_loss * (1 - keep_weight) - keep_loss * keep_weight
            # print( '0 ; ',round(att_loss.item(),5) , 'att loss ',round(keep_loss.item(),5), 'keep weight')
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        z_adv_list.append(z_adv1.detach().to(device))
        target_model.zero_grad()
        z_adv1 = dc(z_adv) if not num else z_new.detach().requires_grad_(True)
        z_nat1 = dc(z_nat) if not num else z_new.detach()

        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            att_loss = F.cross_entropy(100 * logits, torch.tensor([1] * (len(mix) // 2) * z_len, device='cuda'))
            keep_loss = 100 *((adv_feat - ori) @ keep.T).pow(2).mean()
            loss = att_loss * (1 - keep_weight) - keep_loss * keep_weight
            # print( '1 ; ',round(att_loss.item(),5) , 'att loss ',round(keep_loss.item(),5), 'keep weight')
            loss.backward(retain_graph=True)
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        rand_perturb_dist = torch.distributions.uniform.Uniform(-random, random)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(device)
        z_adv_list2.append(z_adv1.detach().to(device))

    return torch.cat(z_adv_list, dim=1).half().detach().to(device), torch.cat(z_adv_list2, dim=1).half().detach().to(
        device)


def perturb_bafa_txt_multi_ablation_lb_ls(z, target_model, mix, t_, keep=None, att_bnd=0.1, att_stp=1e-2, att_itr=20,
                                         random=0.1, num_samples=50, keep_weight=10, use_ls=True):
    target_model.zero_grad()
    device, mix, z_len = z.device, F.normalize(mix, dim=-1), z.size(1)
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)
    with torch.no_grad():
        ori = F.normalize(target_model(z.half(), t_), dim=-1)

    z_adv_list, z_adv_list2 = [], []
    for num in range(num_samples):
        if not num:
            z_adv1, z_nat1 = dc(z_adv), dc(z_nat)
        else:
            z_new = clamper(z_adv + rand_perturb, z_nat, bound=att_bnd)
            z_adv1, z_nat1 = z_new.detach().requires_grad_(True), z_new.detach()
        target_model.zero_grad()
        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            att_loss = F.cross_entropy(100 * logits, torch.tensor([0] * (len(mix) // 2) * z_len, device='cuda'))

            if use_ls and keep is not None:
                keep_loss = 100 * ((adv_feat - ori) @ keep.T).pow(2).mean()
                loss = att_loss * (1 - keep_weight) - keep_loss * keep_weight
            else:
                loss = att_loss
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        z_adv_list.append(z_adv1.detach().to(device))
        target_model.zero_grad()
        z_adv1 = dc(z_adv) if not num else z_new.detach().requires_grad_(True)
        z_nat1 = dc(z_nat) if not num else z_new.detach()

        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            att_loss = F.cross_entropy(100 * logits, torch.tensor([1] * (len(mix) // 2) * z_len, device='cuda'))

            if use_ls and keep is not None:
                keep_loss = 100 * ((adv_feat - ori) @ keep.T).pow(2).mean()
                loss = att_loss * (1 - keep_weight) - keep_loss * keep_weight
            else:
                loss = att_loss
            loss.backward(retain_graph=True)
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        rand_perturb_dist = torch.distributions.uniform.Uniform(-random, random)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(device)
        z_adv_list2.append(z_adv1.detach().to(device))

    return torch.cat(z_adv_list, dim=1).half().detach().to(device), torch.cat(z_adv_list2, dim=1).half().detach().to(
        device)


# def bafa_img_iccv(z, target_model, bias_txt, debias_txt, bound=0.1, step=1e-2, iters=20, l_scale=1.0, lambda_=1):
#     device = z.device
#     z_nat = z.detach().clone().to(device).detach()
#     z_adv = z.detach().clone().requires_grad_(True).to(device)
#     with torch.no_grad():
#         out_feat = target_model(z)
#         cls_d = l_scale * out_feat @ debias_txt.T
#         # cls_b = l_scale * out_feat @ bias_txt.T
#         debias_cls_predict = torch.argmax(l_scale * cls_d, dim=-1)
#         # bias_cls_predict = torch.argmax(l_scale * cls_b, dim=-1)
#
#     for i in range(iters):
#         adv_feat = target_model(z_adv)
#         adv_loss = F.cross_entropy(l_scale * adv_feat @ bias_txt.T, debias_cls_predict)
#         adv_loss_keep = F.cross_entropy(l_scale * adv_feat @ debias_txt.T, debias_cls_predict)
#         loss = adv_loss - lambda_ *  adv_loss_keep # lambda_ * adv_loss_keep
#         loss.backward(retain_graph=True)
#         if not i:
#             f_adv_loss, f_adv_keep = adv_loss, adv_loss_keep
#
#         grad_ = z_adv.grad.data.detach()#.sign()
#         attack_norm = 'linf'
#         if attack_norm == "linf":
#             step_dir = grad_.sign()
#         elif attack_norm == "l2":
#             step_dir = grad_ / (grad_.norm(p=2) + 1e-8)
#
#         z_adv = clamper(z_adv + step_dir * step, z_nat, bound = bound)
#
#         target_model.zero_grad()
#         z_adv.grad = None
#     # print('adv_loss_keep loss :', round((adv_loss-f_adv_loss).item(), 5), 'adv_loss_keep loss : ', round((adv_loss_keep-f_adv_keep).item(), 5))
#
#     return z_adv.detach().to(device)

def bafa_img_iccv(z, target_model, bias_txt, debias_txt, bound=0.1, step=1e-2, iters=20, l_scale=1.0, lambda_=1, rand_eps=0.0):
    device, dtype = z.device, z.dtype
    z_nat = z.detach().clone().to(device).detach()

    if rand_eps > 0:
        rand_dist = torch.distributions.uniform.Uniform(-rand_eps, rand_eps)
        rand_perturb = rand_dist.sample(sample_shape=z.shape).to(device=device, dtype=dtype)
        z_init = clamper(z_nat + rand_perturb, z_nat, bound=bound)
    else:
        z_init = z_nat

    z_adv = z_init.clone().detach().requires_grad_(True).to(device=device, dtype=dtype)

    with torch.no_grad():
        out_feat = target_model(z_nat)
        cls_d = l_scale * out_feat @ debias_txt.T
        debias_cls_predict = torch.argmax(l_scale * cls_d, dim=-1)

    for i in range(iters):
        adv_feat = target_model(z_adv)
        adv_loss = F.cross_entropy(l_scale * adv_feat @ bias_txt.T, debias_cls_predict)
        adv_loss_keep = F.cross_entropy(l_scale * adv_feat @ debias_txt.T, debias_cls_predict)
        loss = adv_loss - lambda_ * adv_loss_keep
        loss.backward(retain_graph=True)

        grad_ = z_adv.grad.data.detach()
        if i == 0:
            f_adv_loss, f_adv_keep = adv_loss, adv_loss_keep

        attack_norm = 'linf'
        if attack_norm == "linf":
            step_dir = grad_.sign()
        elif attack_norm == "l2":
            step_dir = grad_ / (grad_.norm(p=2) + 1e-8)

        z_adv = clamper(z_adv + step_dir * step, z_nat, bound=bound)
        target_model.zero_grad()
        z_adv.grad = None

    return z_adv.detach().to(device=device, dtype=dtype)


def perturb_bafa_txt_multi_all_noli(z, target_model, mix, t_, att_bnd=0.1, att_stp=1e-2, att_itr=20, random=0.1,
                               num_samples=50):
    target_model.zero_grad()
    device, mix, z_len = z.device, F.normalize(mix, dim=-1), z.size(1)
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)
    z_adv_list, z_adv_list2 = [], []
    for num in range(num_samples):
        if not num:
            z_adv1, z_nat1 = dc(z_adv), dc(z_nat)
        else:
            z_new = clamper(z_adv + rand_perturb, z_nat, bound=att_bnd)
            z_adv1, z_nat1 = z_new.detach().requires_grad_(True), z_new.detach()
        target_model.zero_grad()
        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            loss = F.cross_entropy(100 * logits, torch.tensor([0] * (len(mix) // 2) * z_len, device='cuda'))
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        z_adv_list.append(z_adv1.detach().to(device))
        target_model.zero_grad()

        z_adv1 = dc(z_adv) if not num else z_new.detach().requires_grad_(True)
        z_nat1 = dc(z_nat) if not num else z_new.detach()
        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            logits = torch.stack([
                torch.cat((adv_feat[i] @ a.unsqueeze(1), adv_feat[i] @ b.unsqueeze(1)), dim=0)
                for i in range(len(adv_feat)) for a, b in zip(mix[:len(mix) // 2], mix[len(mix) // 2:])], dim=0)
            loss = F.cross_entropy(100 * logits, torch.tensor([1] * (len(mix) // 2) * z_len, device='cuda'))
            loss.backward(retain_graph=True)
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        rand_perturb_dist = torch.distributions.uniform.Uniform(-random, random)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(device)
        z_adv_list2.append(z_adv1.detach().to(device))
    return torch.cat(z_adv_list, dim=1).half().detach().to(device), torch.cat(z_adv_list2, dim=1).half().detach().to(device)


def perturb_bafa_bm_multi(z, target_model, bias_cb, debias_cb, t_, att_bnd=0.1, att_stp=1e-3, att_itr=10,
                          random=0.001, num_samples=50):
    target_model.zero_grad()
    device, z_len = z.device, z.size(1)
    ori = F.normalize(target_model(z.half(), t_), dim=-1)
    bias_cb = F.normalize(bias_cb, dim=-1)
    debias_cb = F.normalize(debias_cb, dim=-1)
    cls_d = 100 * ori @ debias_cb.T
    cls_b = 100 * ori @ bias_cb.T
    debias_cls_predict = torch.argmax(cls_d, dim=-1)
    bias_cls_predict = torch.argmax(cls_b, dim=-1)

    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)
    z_adv_list, z_adv_list2 = [], []
    for num in range(num_samples):
        if not num:
            z_adv1, z_nat1 = dc(z_adv), dc(z_nat)
        else:
            z_new = clamper(z_adv + rand_perturb, z_nat, bound=att_bnd)
            z_adv1, z_nat1 = z_new.detach().requires_grad_(True), z_new.detach()
        target_model.zero_grad()
        for i in range(att_itr):
            adv_feat = F.normalize(target_model(z_adv1.half(), t_), dim=-1)
            adv_cls_d = 100 * adv_feat @ debias_cb.T
            adv_cls_b = 100 * adv_feat @ bias_cb.T
            adv_loss = F.cross_entropy(adv_cls_b, bias_cls_predict)
            adv_loss_keep = F.cross_entropy(adv_cls_d, debias_cls_predict)
            loss = adv_loss - adv_loss_keep
            # print(adv_loss , adv_loss_keep)
            loss.backward(retain_graph=True)
            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv1 = clamper(z_adv1 + grad_sign * att_stp, z_nat1, bound=att_bnd)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None
        rand_perturb_dist = torch.distributions.uniform.Uniform(-random, random)
        rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(device)
        z_adv_list.append(z_adv1.detach().to(device))
    return torch.cat(z_adv_list, dim=1).half().detach().to(device)


def perturb_bdfa_txt_c(z, target_model, mix, t_, test=None):
    target_model.zero_grad()
    device = z.device
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)

    z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
    att_bnd, att_stp, att_itr = 0.1, 1e-3, 20
    with torch.no_grad():
        ori = target_model(z, t_)
    keep = ori @ test[0]  # keep info
    for i in range(att_itr):
        adv_feat = target_model(z_adv1.half(), t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = torch.stack([
            (adv_feat[0] @ mix[:4].t()).mean(), (adv_feat[0] @ mix[4:].t()).mean(),
            (adv_feat[1] @ mix[:4].t()).mean(), (adv_feat[1] @ mix[4:].t()).mean()
        ]).view(2, 2)
        loss = F.cross_entropy(logits * 100, torch.tensor([1, 0], device='cuda')) - F.mse_loss(adv_feat @ test[0] * 100,
                                                                                               keep * 100)
        loss.backward(retain_graph=True)

        grad_sign = z_adv1.grad.data.detach().sign()
        z_adv_new = z_adv1 + grad_sign * att_stp  # a sign( d( L (x_adv)))
        z_adv1 = clamper(z_adv_new, z_nat1, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv1.grad = None
        # print(F.cross_entropy(logits, torch.tensor([0, 1], device='cuda')), F.mse_loss(adv_feat @ test[0], keep))
        # print(adv_feat @ test[0], sum(torch.abs(adv_feat @ test[0] - check)))
        # print(logits)
        # print(adv_feat @ test.t())

    return z_adv1.detach().to(device).half()


def perturb_bafa_img2txt(z, target_model, img_set, ori_bias_imgs, mix, t_):
    target_model.zero_grad()
    device = z.device
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)

    z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
    att_bnd, att_stp, att_itr, batch = 0.3, 1e-3, 5, img_set.size(0)

    label1 = torch.zeros(batch).type(torch.LongTensor).to(z.device)
    label2 = torch.ones(batch).type(torch.LongTensor).to(z.device)
    with torch.no_grad():
        ori_em = target_model(z, t_)
        ori_em = ori_em / ori_em.norm(dim=-1, keepdim=True)

    for i in range(att_itr):
        adv_feat = target_model(z_adv1.half(), t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = 100 * img_set @ adv_feat.t()
        ori_logits = 100 * ori_bias_imgs @ adv_feat.t()
        # loss = F.cross_entropy(ori_logits, label1) - F.cross_entropy(logits, label1)  # all middle this right, nobias_p left
        logits = torch.stack([
            (adv_feat[0] @ mix[:4].t()).mean(), (adv_feat[0] @ mix[4:].t()).mean(),
            (adv_feat[1] @ mix[:4].t()).mean(), (adv_feat[1] @ mix[4:].t()).mean()
        ]).view(2, 2)
        loss = F.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))
        loss -= F.mse_loss(adv_feat, ori_em)
        loss.backward(retain_graph=True)

        grad_sign = z_adv1.grad.data.detach().sign()
        z_adv_new = z_adv1 + grad_sign * att_stp  # a sign( d( L (x_adv)))
        z_adv1 = clamper(z_adv_new, z_nat1, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv1.grad = None

    for i in range(att_itr):
        adv_feat = target_model(z_adv2.half(), t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = 100 * img_set @ adv_feat.t()
        ori_logits = 100 * ori_bias_imgs @ adv_feat.t()
        # loss = F.cross_entropy(ori_logits, label2) - F.cross_entropy(logits, label2)
        logits = torch.stack([
            (adv_feat[0] @ mix[:4].t()).mean(), (adv_feat[0] @ mix[4:].t()).mean(),
            (adv_feat[1] @ mix[:4].t()).mean(), (adv_feat[1] @ mix[4:].t()).mean()
        ]).view(2, 2)
        loss = F.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))
        loss -= F.mse_loss(adv_feat, ori_em)
        loss.backward(retain_graph=True)

        grad_sign = z_adv2.grad.data.detach().sign()
        z_adv_new = z_adv2 + grad_sign * att_stp  # a sign( d( L (x_adv)))
        z_adv2 = clamper(z_adv_new, z_nat2, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv2.grad = None

    return z_adv1.detach().to(device).half(), z_adv2.detach().to(device).half()


def clamper(x_adv, x_nat, bound=None, metric="inf"):
    if metric == "inf":
        clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
    else:
        clamp_delta = x_adv - x_nat
        for batch_index in range(clamp_delta.size(0)):
            image_delta = clamp_delta[batch_index]
            image_norm = image_delta.norm(p=metric, keepdim=False)
            if image_norm > bound:
                clamp_delta[batch_index] /= image_norm
                clamp_delta[batch_index] *= bound
    x_adv = x_nat + clamp_delta
    return x_adv.clone().detach().requires_grad_(True)


def simclr_loss(features1, features2, temperature=0.1):
    """
    Computes the SimCLR loss between two sets of features.

    Args:
    features1: torch.Tensor - Tensor of shape (batch_size, feature_dim)
    features2: torch.Tensor - Tensor of shape (batch_size, feature_dim)
    temperature: float - Temperature parameter for scaling the logits

    Returns:
    loss: torch.Tensor - Computed SimCLR loss
    """
    # Normalize the features
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)

    # Concatenate features for the similarity matrix computation
    features = torch.cat([features1, features2], dim=0)

    # Compute the similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Create labels for the contrastive learning
    batch_size = features1.shape[0]
    labels = torch.arange(batch_size, device=features1.device)
    labels = torch.cat([labels, labels], dim=0)

    # Mask to remove self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # Compute the positive pair loss
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)],
                          dim=0)
    positives = positives.view(2 * batch_size, 1)

    # Compute the cross-entropy loss
    loss = -torch.log(torch.exp(positives) / torch.exp(similarity_matrix).sum(dim=1, keepdim=True)).mean()
    return loss


def get_knn_avg_dist(features1: torch.Tensor, features2: torch.Tensor, knn: int = 10,
                     **_: torch.Tensor) -> torch.Tensor:
    # get the top-k nearest neighbors
    scores = features1 @ features2.T
    topk_distances = scores.topk(int(knn), dim=1, largest=True, sorted=True)[0]
    # get the average distance
    average_dist = topk_distances.mean(dim=1)
    return average_dist


def bias_ce_loss(adv_pred, device, bias_label=True):
    if len(adv_pred.shape) == 3:
        batch, bias_num, label = adv_pred.shape
        cross_entropy_results = torch.zeros(batch, bias_num, device=device, dtype=torch.float)
        # if attack for 1, else training for 0
        y_ = torch.ones(batch, device=device, dtype=torch.long) if bias_label else torch.zeros(batch, device=device,
                                                                                               dtype=torch.long)
        for bias_ in range(bias_num):
            # Select the logits for the i-th kind of output feature
            logits_i = adv_pred[:, bias_, :]

            # Compute cross-entropy for the selected logits
            cross_entropy_results[:, bias_] = F.cross_entropy(logits_i, y_, reduction='none')
    else:
        batch, label = adv_pred.shape
        max_indices = torch.argmax(adv_pred, dim=1).long()
        one_hot_vectors = torch.eye(6, device=device)[max_indices]

        cross_entropy_results = F.cross_entropy(adv_pred, y_, reduction='none')

    return cross_entropy_results


def orthogonal_projection(basis):
    proj = torch.inverse(torch.matmul(basis.T, basis))
    proj = torch.matmul(basis, proj)
    proj = torch.matmul(proj, basis.T)
    proj = torch.eye(basis.shape[0]).to(basis.device) - proj
    return proj

# # Each pair of logits is treated as mix binary classification problem
# logits_pairs = torch.cat((adv_mix_logit[group][:, :5].unsqueeze(2), adv_mix_logit[group][:, 5:].unsqueeze(2)),
#                          dim=2)  # Shape (128, 5, 2)
# logits_flat = logits_pairs.view(-1, 2)  # Shape (640, 2)
# labels_expanded = target_y[group].repeat_interleave(5)
# new_loss = F.cross_entropy(logits_flat, labels_expanded)
# loss += new_loss
