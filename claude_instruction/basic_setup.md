* This is an empty repo, we want to write codes here to finetune a molmoact2 VLA (https://github.com/allenai/molmoact2) on demostration data from RoboEval (@/home/hongrui/codes/RoboEval).

* Basic env setup
    * Create a conda environment for it.
    * Install LeRobot 
        * with the MolmoAct2 optional dependencies (https://huggingface.co/docs/lerobot/molmoact2) into the conda environment (we use conda env instead of uv becasue I am not familiar with uv).
        * with LIBERO dependencies. 

* First step:
    * Download `MolmoAct2-LIBERO` checkpoint
    * Write a short sanity=check code to run MolmoAct2-LIBERO on a single task of LIBERO (user inputs task and task_ids), save video of all camera obervations provided to MolmoAct2. 

    