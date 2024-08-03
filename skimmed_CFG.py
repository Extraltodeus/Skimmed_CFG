import torch

@torch.no_grad()
def skimmed_CFG(x_orig, cond, uncond, cond_scale, skimming_scale):
    denoised = x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))

    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()

    deviation_influence = (denoised.sign() == (denoised - x_orig).sign())
    outer_influence = matching_pred_signs & matching_diff_after & deviation_influence

    low_cfg_denoised_outer = x_orig - ((x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond)))
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (low_cfg_denoised_outer_difference[outer_influence] / cond_scale)

    return cond

class CFG_skimming_single_scale_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        step_step = 10
        return {"required": {"model": ("MODEL",),
                             "Skimming_CFG": ("FLOAT", {"default": 7,  "min": 0.0, "max": 7.0,  "step": 1/step_step, "round": 1/100}),
                             "razor_skim" : ("BOOLEAN", {"default": False})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Pre CFG"
    def patch(self, model, Skimming_CFG, razor_skim):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig     = args['input']
            if not torch.any(conds_out[1]):
                return conds_out
            conds_out[1] = skimmed_CFG(x_orig, conds_out[1], conds_out[0], cond_scale, Skimming_CFG if not razor_skim else 0)
            conds_out[0] = skimmed_CFG(x_orig, conds_out[0], conds_out[1], cond_scale, Skimming_CFG)
            return conds_out
        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )
