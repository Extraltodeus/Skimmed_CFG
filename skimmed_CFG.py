import torch

@torch.no_grad()
def get_skimming_mask(x_orig, cond, uncond, cond_scale, return_denoised=False, disable_flipping_filter=False):
    denoised = x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = (denoised.sign() == (denoised - x_orig).sign())
        outer_influence = matching_pred_signs & matching_diff_after & deviation_influence
    if return_denoised:
        return outer_influence, denoised
    else:
        return outer_influence

@torch.no_grad()
def skimmed_CFG(x_orig, cond, uncond, cond_scale, skimming_scale, disable_flipping_filter=False):
    outer_influence, denoised = get_skimming_mask(x_orig, cond, uncond, cond_scale, True, disable_flipping_filter)
    low_cfg_denoised_outer = x_orig - ((x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond)))
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (low_cfg_denoised_outer_difference[outer_influence] / cond_scale)
    return cond

class CFG_skimming_single_scale_pre_cfg_node:
    @classmethod
    def INPUT_TYPES(s):
        step_step = 2
        return {"required": {"model": ("MODEL",),
                             "Skimming_CFG": ("FLOAT", {"default": 7,  "min": 0.0, "max": 7.0,  "step": 1/step_step, "round": 1/100}),
                             "full_skim_negative" : ("BOOLEAN", {"default": False}),
                             "disable_flipping_filter" : ("BOOLEAN", {"default": False})
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Pre CFG"
    def patch(self, model, Skimming_CFG, full_skim_negative, disable_flipping_filter):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig     = args['input']
            if not torch.any(conds_out[1]):
                return conds_out
            conds_out[1] = skimmed_CFG(x_orig, conds_out[1], conds_out[0], cond_scale, Skimming_CFG if not full_skim_negative else 0, disable_flipping_filter)
            conds_out[0] = skimmed_CFG(x_orig, conds_out[0], conds_out[1], cond_scale, Skimming_CFG, disable_flipping_filter)
            return conds_out
        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class skimReplacePreCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig     = args['input']

            if not torch.any(conds_out[1]):
                return conds_out

            cond = conds_out[0]
            uncond = conds_out[1]

            skim_mask = get_skimming_mask(x_orig, cond, uncond, cond_scale)
            uncond[skim_mask] = cond[skim_mask]

            skim_mask = get_skimming_mask(x_orig, uncond, cond, cond_scale)
            uncond[skim_mask] = cond[skim_mask]

            return [cond,uncond]

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )


class SkimmedCFGLinInterpCFGPreCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        step_step = 2
        return {"required": {
                                "model": ("MODEL",),
                                "Skimming_CFG": ("FLOAT", {"default": 5.0,  "min": 0.0, "max": 7.0,  "step": 1/step_step, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, Skimming_CFG):

        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig     = args['input']

            if not torch.any(conds_out[1]):
                return conds_out

            fallback_weight = (Skimming_CFG - 1) / (cond_scale - 1)

            skim_mask = get_skimming_mask(x_orig, conds_out[0], conds_out[1], cond_scale)
            conds_out[1][skim_mask] = conds_out[0][skim_mask] * (1 - fallback_weight) + conds_out[1][skim_mask] * fallback_weight

            skim_mask = get_skimming_mask(x_orig, conds_out[1], conds_out[0], cond_scale)
            conds_out[1][skim_mask] = conds_out[0][skim_mask] * (1 - fallback_weight) + conds_out[1][skim_mask] * fallback_weight

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )

class SkimmedCFGLinInterpDualScalesCFGPreCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        step_step = 2
        return {"required": {
                                "model": ("MODEL",),
                                "Skimming_CFG_positive": ("FLOAT", {"default": 5.0,  "min": 0.0, "max": 7.0,  "step": 1/step_step, "round": 1/100}),
                                "Skimming_CFG_negative": ("FLOAT", {"default": 5.0,  "min": 0.0, "max": 7.0,  "step": 1/step_step, "round": 1/100}),
                              }
                              }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Pre CFG"

    def patch(self, model, Skimming_CFG_positive, Skimming_CFG_negative):

        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out  = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig     = args['input']

            if not torch.any(conds_out[1]):
                return conds_out

            fallback_weight_positive = (Skimming_CFG_positive - 1) / (cond_scale - 1)
            fallback_weight_negative = (Skimming_CFG_negative - 1) / (cond_scale - 1)

            skim_mask = get_skimming_mask(x_orig, conds_out[1], conds_out[0], cond_scale)
            conds_out[1][skim_mask] = conds_out[0][skim_mask] * (1 - fallback_weight_negative) + conds_out[1][skim_mask] * fallback_weight_negative

            skim_mask = get_skimming_mask(x_orig, conds_out[0], conds_out[1], cond_scale)
            conds_out[1][skim_mask] = conds_out[0][skim_mask] * (1 - fallback_weight_positive) + conds_out[1][skim_mask] * fallback_weight_positive

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (m, )
