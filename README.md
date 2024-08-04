# Skimmed_CFG
A powerful anti-burn allowing much higher CFG scales for latent diffusion models (for ComfyUI)

CFG below at: 6/8/12/16/24/32, skimming scale at 6

![6_8_12_16_24_32](https://github.com/user-attachments/assets/6eb4beb6-0579-4f3e-a85e-e23b6472ebae)


Simply plug after the model loader (same for all the fours nodes):

![image](https://github.com/user-attachments/assets/b188947c-6226-42ff-b868-e6a44bbfe590)

## nodes: 
- Skimmed CFG: My version first version of this, works like a charm!
- Skimmed CFG - replace: replace the values within the negative by those in the positive prediction, nullifying the effect of values targeted by the filter.
- Skimmed CFG - linear interpolation: instead of replacing, does a linear interpolation in between the values. Highly recommanded!
- Skimmed CFG - linear interpolation dual scales: Two scales. One named "positive" and one.. well "negative". The name is more related to a visualy intuitive relation rather than fully from the predictions. A higher positive will tend to go towards high saturations and vice versa with the other slider.

## Side-effects:

- better prompt adherence
- sharper images
- less mess / more randomness due to less conflicts in between the positive and negative predictions
- something something sometimes fused fingers with too low skimming CFG scale and too low amount of steps.


## Tips:

- The "Razor skim" toggle may give interesting results...
- The skimming scale is basically how much do you like them burned. 3 was the intended scale but suit to your needs.
- The SDE samplers can still burn a little but much less
- The SDE samplers can still totaly do nonsense with low steps
- A too low skimming scale may require to do more steps
- Recommanded skim: 2-3 for maximum antiburn, 5-7 for colorful/strong style. 4 is cruise scale.
- a good negative prompt is a style negative prompt
- to use super high scales it is not a bad idea to cut the negative before the end. You can find in [this repository](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI/tree/main) a node named "Support empty uncond". Plug it after the skimmed cfg node. Then menu>advanced>conditioning>ConditioningSetTimestepRange and set the and at ~65%. This will avoid potential artifacts.



# Pro tip:

[It would be actually nice to have some support!](https://www.patreon.com/extraltodeus) because like this I will continue to share my findings!

Did you know that my first activity is to write creative model merging functions?

While the code is too much of a mess to be shared, I do expose and share my models. You can find them in this [gallery](https://github.com/Extraltodeus/shared_models_galleries)! 😁



# Other examples with a CFG at 100

### Base image 🤭

![00417UI_00001_](https://github.com/user-attachments/assets/0b4a5ae2-4815-456f-a3ff-4280d311842b)

### Linear interpolation dual scale 10/0

![00409UI_00001_](https://github.com/user-attachments/assets/9bcf9121-4341-4948-94e7-2aebca50a4d3)

### Linear interpolation dual scale 5/0

![00416UI_00001_](https://github.com/user-attachments/assets/f51365fa-9553-43a1-a060-e2b545b1dc74)

### Linear interpolation scale 5

![00407UI_00001_](https://github.com/user-attachments/assets/ead094fd-a74c-4722-a393-63a9bf738b10)

### Replace

![00405UI_00001_](https://github.com/user-attachments/assets/8c14a7a7-6b04-4d4b-9264-4651b3134186)

### Skimmed CFG node skimming scale at 4

![00419UI_00001_](https://github.com/user-attachments/assets/0f84ce0a-5547-4594-aff4-9b67eeb3bdf2)

### Skimmed CFG node skimming scale at 4 with razor skim

![00420UI_00001_](https://github.com/user-attachments/assets/861b7c42-8f48-4123-904e-bd1ada973595)

### forgotten setting 😶

![00454UI_00001_](https://github.com/user-attachments/assets/b3f107e4-8ee4-4eb8-beb2-e15506e02283)


