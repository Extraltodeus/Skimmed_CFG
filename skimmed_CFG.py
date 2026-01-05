import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

MAX_SCALE = 10
STEP_STEP = 2


@torch.no_grad()
def get_skimming_mask(
    x_orig,
    cond,
    uncond,
    cond_scale,
    return_denoised=False,
    disable_flipping_filter=False,
):
    denoised = x_orig - (
        (x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    )

    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        outer_influence = (
            matching_pred_signs & matching_diff_after & deviation_influence
        )

    if return_denoised:
        return outer_influence, denoised
    else:
        return outer_influence


@torch.no_grad()
def skimmed_CFG(
    x_orig, cond, uncond, cond_scale, skimming_scale, disable_flipping_filter=False
):
    outer_influence, denoised = get_skimming_mask(
        x_orig, cond, uncond, cond_scale, True, disable_flipping_filter
    )
    low_cfg_denoised_outer = x_orig - (
        (x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (
        low_cfg_denoised_outer_difference[outer_influence] / cond_scale
    )
    return cond


@torch.no_grad()
def interpolated_scales(
    x_orig, cond, uncond, cond_scale, small_scale, squared=False, root_dist=False
):
    deltacfg_normal = x_orig - cond_scale * cond - (cond_scale - 1) * uncond
    deltacfg_small = x_orig - small_scale * cond - (small_scale - 1) * uncond
    absdiff = (deltacfg_normal - deltacfg_small).abs()

    # Fix division by zero
    diff_range = absdiff.max() - absdiff.min()
    if diff_range > 0:
        absdiff = (absdiff - absdiff.min()) / diff_range
    else:
        absdiff = torch.zeros_like(absdiff)

    if squared:
        absdiff = absdiff**2
    elif root_dist:
        absdiff = absdiff**0.5

    new_scale = (small_scale - 1) / (cond_scale - 1) if cond_scale > 1 else 0.0
    smaller_uncond = cond * (1 - new_scale) + uncond * new_scale
    new_uncond = smaller_uncond * (1 - absdiff) + uncond * absdiff
    return new_uncond


class CFG_Skimming_Single_Scale_Pre_CFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CFG_Skimming_Single_Scale_Pre_CFG",
            display_name="Skimmed CFG",
            category="model_patches/Pre CFG",
            description="Apply skimming CFG with a single scale. Skims 'bad' values to a fallback scale.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "skimming_cfg",
                    default=7.0,
                    min=0.0,
                    max=MAX_SCALE,
                    step=1.0 / STEP_STEP,
                    tooltip="The fallback scale for the 'bad' values. Set to -1 to use the current CFG scale.",
                ),
                io.Boolean.Input(
                    "full_skim_negative",
                    default=False,
                    tooltip="If enabled, fully skim negative conditioning (set to 0).",
                ),
                io.Boolean.Input(
                    "disable_flipping_filter",
                    default=False,
                    tooltip="Disable the flipping filter for skimming detection.",
                ),
                io.Float.Input(
                    "start_at_percentage",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Start applying skimming at this percentage of the denoising process (0 = start, 1 = end).",
                ),
                io.Float.Input(
                    "end_at_percentage",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Stop applying skimming at this percentage of the denoising process (0 = start, 1 = end).",
                ),
                io.Float.Input(
                    "flip_at_percentage",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Flip the flipping filter at this percentage. Set to 0 to disable.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        skimming_cfg: float,
        full_skim_negative: bool,
        disable_flipping_filter: bool,
        start_at_percentage: float,
        end_at_percentage: float,
        flip_at_percentage: float,
    ) -> io.NodeOutput:
        model_sampling = model.get_model_object("model_sampling")
        start_at_sigma = model_sampling.percent_to_sigma(start_at_percentage)
        end_at_sigma = model_sampling.percent_to_sigma(end_at_percentage)
        flip_at_sigma = model_sampling.percent_to_sigma(flip_at_percentage)

        if 1 > flip_at_percentage > 0:
            print(f" \033[92mFlip at sigma: {round(flip_at_sigma, 2)}\033[0m")

        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args["input"]
            sigma = args["sigma"][0].item()

            # Fix: Use >= instead of > for proper boundary checking
            if (
                not torch.any(conds_out[1])
                or sigma <= end_at_sigma
                or sigma >= start_at_sigma
            ):
                return conds_out

            practical_scale = cond_scale if skimming_cfg < 0 else skimming_cfg

            flip_filter = disable_flipping_filter
            if flip_at_percentage > 0 and sigma > flip_at_sigma:
                flip_filter = not disable_flipping_filter

            conds_out[1] = skimmed_CFG(
                x_orig,
                conds_out[1],
                conds_out[0],
                cond_scale,
                practical_scale if not full_skim_negative else 0,
                flip_filter,
            )
            conds_out[0] = skimmed_CFG(
                x_orig,
                conds_out[0],
                conds_out[1],
                cond_scale - 1,
                practical_scale,
                flip_filter,
            )
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return io.NodeOutput(m)


class SkimFlipPreCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkimFlipPreCFG",
            display_name="Skimmed CFG - Timed flip",
            category="model_patches/Pre CFG",
            description="Flip the skimming filter at a specific point in the denoising process.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "flip_at",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=1.0 / 20,
                    tooltip="Relative to the step progression. Completely at 0 will give smoother results. Completely at one will give noisier results. The influence is more important from 0% to 30%.",
                ),
                io.Boolean.Input(
                    "reverse",
                    default=False,
                    tooltip="If turned on you will obtain a composition closer to what you would normally get with no modification.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, flip_at: float, reverse: bool) -> io.NodeOutput:
        # Use the main node with specific parameters
        return CFG_Skimming_Single_Scale_Pre_CFG.execute(
            model=model,
            skimming_cfg=-1.0,
            full_skim_negative=True,
            disable_flipping_filter=reverse,
            start_at_percentage=0.0,
            end_at_percentage=1.0,
            flip_at_percentage=flip_at,
        )


class ConstantSkimPreCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConstantSkimPreCFG",
            display_name="Skimmed CFG - Clean Skim",
            category="model_patches/Pre CFG",
            description="Apply constant skimming CFG. Can be enabled/disabled.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="Enable constant skimming CFG.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, enabled: bool) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)

        return CFG_Skimming_Single_Scale_Pre_CFG.execute(
            model=model,
            skimming_cfg=-1.0,
            full_skim_negative=True,
            disable_flipping_filter=False,
            start_at_percentage=0.0,
            end_at_percentage=1.0,
            flip_at_percentage=0.0,
        )


class SkimReplacePreCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkimReplacePreCFG",
            display_name="Skimmed CFG - replace",
            category="model_patches/Pre CFG",
            description="Replace negative conditioning with positive where skimming mask is detected.",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args["input"]

            if not torch.any(conds_out[1]):
                return conds_out

            cond = conds_out[0]
            uncond = conds_out[1]

            skim_mask = get_skimming_mask(x_orig, cond, uncond, cond_scale)
            uncond[skim_mask] = cond[skim_mask]

            skim_mask = get_skimming_mask(x_orig, uncond, cond, cond_scale - 1)
            uncond[skim_mask] = cond[skim_mask]

            # Fix: Return consistent format
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return io.NodeOutput(m)


class SkimmedCFG_LinInterp_CFG_PreCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkimmedCFG_LinInterp_CFG_PreCFG",
            display_name="Skimmed CFG - linear interpolation",
            category="model_patches/Pre CFG",
            description="Apply skimming CFG with linear interpolation between scales.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "skimming_cfg",
                    default=5.0,
                    min=0.0,
                    max=MAX_SCALE,
                    step=1.0 / STEP_STEP,
                    tooltip="The fallback CFG scale for linear interpolation.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, skimming_cfg: float) -> io.NodeOutput:
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args["input"]

            if not torch.any(conds_out[1]):
                return conds_out

            # Fix: Prevent division by zero
            if cond_scale <= 1:
                return conds_out

            fallback_weight = (skimming_cfg - 1) / (cond_scale - 1)

            skim_mask = get_skimming_mask(
                x_orig, conds_out[0], conds_out[1], cond_scale
            )
            conds_out[1][skim_mask] = (
                conds_out[0][skim_mask] * (1 - fallback_weight)
                + conds_out[1][skim_mask] * fallback_weight
            )

            skim_mask = get_skimming_mask(
                x_orig, conds_out[1], conds_out[0], cond_scale
            )
            conds_out[1][skim_mask] = (
                conds_out[0][skim_mask] * (1 - fallback_weight)
                + conds_out[1][skim_mask] * fallback_weight
            )

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return io.NodeOutput(m)


class SkimmedCFG_LinInterp_DualScales_CFG_PreCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SkimmedCFG_LinInterp_DualScales_CFG_PreCFG",
            display_name="Skimmed CFG - linear interpolation dual scales",
            category="model_patches/Pre CFG",
            description="Apply skimming CFG with linear interpolation using separate scales for positive and negative conditioning.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "skimming_cfg_positive",
                    default=5.0,
                    min=0.0,
                    max=MAX_SCALE,
                    step=1.0 / STEP_STEP,
                    tooltip="The fallback CFG scale for positive conditioning.",
                ),
                io.Float.Input(
                    "skimming_cfg_negative",
                    default=5.0,
                    min=0.0,
                    max=MAX_SCALE,
                    step=1.0 / STEP_STEP,
                    tooltip="The fallback CFG scale for negative conditioning.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(
        cls, model, skimming_cfg_positive: float, skimming_cfg_negative: float
    ) -> io.NodeOutput:
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args["input"]

            if not torch.any(conds_out[1]):
                return conds_out

            # Fix: Prevent division by zero
            if cond_scale <= 1:
                return conds_out

            fallback_weight_positive = (skimming_cfg_positive - 1) / (cond_scale - 1)
            fallback_weight_negative = (skimming_cfg_negative - 1) / (cond_scale - 1)

            skim_mask = get_skimming_mask(
                x_orig, conds_out[1], conds_out[0], cond_scale
            )
            conds_out[1][skim_mask] = (
                conds_out[0][skim_mask] * (1 - fallback_weight_negative)
                + conds_out[1][skim_mask] * fallback_weight_negative
            )

            skim_mask = get_skimming_mask(
                x_orig, conds_out[0], conds_out[1], cond_scale
            )
            conds_out[1][skim_mask] = (
                conds_out[0][skim_mask] * (1 - fallback_weight_positive)
                + conds_out[1][skim_mask] * fallback_weight_positive
            )

            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return io.NodeOutput(m)


class DifferenceCFG_PreCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DifferenceCFG_PreCFG",
            display_name="Skimmed CFG - Difference CFG",
            category="model_patches/Pre CFG",
            description="Apply CFG based on difference between reference and current CFG scales.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "reference_cfg",
                    default=5.0,
                    min=0.0,
                    max=MAX_SCALE,
                    step=1.0 / STEP_STEP,
                    tooltip="The reference CFG scale to compare against.",
                ),
                io.Combo.Input(
                    "method",
                    options=[
                        "linear_distance",
                        "squared_distance",
                        "root_distance",
                        "absolute_sum",
                    ],
                    default="linear_distance",
                    tooltip="The method to calculate the difference.",
                ),
                io.Float.Input(
                    "end_at_percentage",
                    default=0.80,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Relative to the step progression. 0 means disabled, 1 means active until the end.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        reference_cfg: float,
        method: str,
        end_at_percentage: float,
    ) -> io.NodeOutput:
        model_sampling = model.get_model_object("model_sampling")
        end_at_sigma = model_sampling.percent_to_sigma(end_at_percentage)
        print(
            f" \033[92mDifference CFG method: {method} / Reference Scale: {reference_cfg} / End at percent/sigma: {round(end_at_percentage, 2)}/{round(end_at_sigma, 2)}\033[0m"
        )

        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out = args["conds_out"]
            cond_scale = args["cond_scale"]
            x_orig = args["input"]
            sigma = args["sigma"][0]

            if not torch.any(conds_out[1]) or sigma <= end_at_sigma:
                return conds_out

            if method == "absolute_sum":
                ref_norm = (
                    conds_out[0] * reference_cfg - conds_out[1] * (reference_cfg - 1)
                ).norm(p=1)
                cfg_norm = (
                    conds_out[0] * cond_scale - conds_out[1] * (cond_scale - 1)
                ).norm(p=1)

                # Fix: Prevent division by zero
                if cfg_norm == 0:
                    return conds_out

                new_scale = cond_scale * ref_norm / cfg_norm

                # Fix: Prevent division by zero
                if cond_scale <= 1:
                    return conds_out

                fallback_weight = (new_scale - 1) / (cond_scale - 1)
                conds_out[1] = (
                    conds_out[0] * (1 - fallback_weight)
                    + conds_out[1] * fallback_weight
                )
            elif method in ["linear_distance", "squared_distance", "root_distance"]:
                conds_out[1] = interpolated_scales(
                    x_orig,
                    conds_out[0],
                    conds_out[1],
                    cond_scale,
                    reference_cfg,
                    method == "squared_distance",
                    method == "root_distance",
                )
            return conds_out

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return io.NodeOutput(m)


class SkimmedCFGExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CFG_Skimming_Single_Scale_Pre_CFG,
            SkimFlipPreCFG,
            ConstantSkimPreCFG,
            SkimReplacePreCFG,
            SkimmedCFG_LinInterp_CFG_PreCFG,
            SkimmedCFG_LinInterp_DualScales_CFG_PreCFG,
            DifferenceCFG_PreCFG,
        ]


async def comfy_entrypoint() -> SkimmedCFGExtension:
    return SkimmedCFGExtension()
