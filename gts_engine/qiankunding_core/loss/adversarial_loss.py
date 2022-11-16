
import torch
import torch.nn.functional as F


class AdversarialLoss(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # * divergence function
        self.divergence = getattr(self, args.divergence)

    def __call__(self, model, logits, train_inputs):
        # * get disturbed inputs
        inputs_embeds = model.bert.embeddings.word_embeddings(train_inputs['input_ids'])
        noise = inputs_embeds.clone().detach().normal_(
            0, 1).requires_grad_(True) * self.args.noise_var
        
        # * adv loop
        for i in range(self.args.adv_nloop):
            inputs_embeds = inputs_embeds.detach() + noise
            adv_logits = model(attention_mask=train_inputs['attention_mask'],
                               token_type_ids=train_inputs['token_type_ids'],
                               inputs_embeds=inputs_embeds)

            adv_loss = self.divergence(adv_logits,
                                    logits.detach(),
                                    reduction='batchmean')


            # * now we need to find the best noise according to gradient
            # * theoretically we need the max, to be more efficient, we
            # * approximate with it by gradient assent
            noise_grad = torch.autograd.grad(outputs=adv_loss, inputs=noise, retain_graph=True)[0]
            noise = noise + noise_grad * self.args.adv_step_size
        
        # * normalization 这里的noise之后好像都没用到, 故注释掉
        #  noise = self.adv_project(noise,
        #                           norm_type=self.args.project_norm_type,
        #                           eps=self.args.noise_gamma)
        adv_loss = self.divergence(adv_logits, logits)

        return adv_loss


    @staticmethod
    def adv_project(grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    @staticmethod
    def kl(input, target, reduction="sum"):
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                        F.softmax(target, dim=-1, dtype=torch.float32),
                        reduction=reduction)
        return loss


    @staticmethod
    def sym_kl(input, target, reduction="sum"):
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
        return loss


    @staticmethod
    def js(input, target, reduction="sum"):
        input = input.float()
        target = target.float()
        m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + \
            F.softmax(input.detach(), dim=-1, dtype=torch.float32)
        m = 0.5 * m
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), m, reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)

        return loss
