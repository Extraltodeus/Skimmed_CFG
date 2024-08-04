from .skimmed_CFG import *

NODE_CLASS_MAPPINGS = {
    "Skimmed CFG": CFG_skimming_single_scale_pre_cfg_node,
    "Skimmed CFG - replace": skimReplacePreCFGNode,
    "Skimmed CFG - linear interpolation": SkimmedCFGLinInterpCFGPreCFGNode,
    "Skimmed CFG - linear interpolation dual scales": SkimmedCFGLinInterpDualScalesCFGPreCFGNode,
    
}
