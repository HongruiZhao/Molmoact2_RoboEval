* Overview: you will finetune MolmoAct 2 checkpoint on RoboEval dataset.
* Env to use: `conda activate molmoact2`.
* How to load RoboEval dataset: @scripts/roboeval_dataset_generation/check_roboeval_dataset.py.
* How to load MolmoAct 2 checkpoint: @scripts/sanity_check_libero.py.
* Where the new training scripts should go: @scripts/molmoact2_training.
* Other requiremnts for training:
    * Use built-in funtions of LeRobot library to conduct training as much as possible.
    * LoRA finetune both VLM and action expert.
    * Plot training loss vs step in wandb.
    * Training configuration should be stored/loaded as a json file. Don't use ArgumentParser.
    * stats:
        * Batch size of 32, 100k steps.
        * chunk size = 10.
        * output end effector pose.
        * bfloat16 model type.
        * num_flow_timesteps = 8.
    * For unmentioned details like learning rates, follow offical LeRobot MolmoAct2 training recipe: https://huggingface.co/docs/lerobot/molmoact2. 