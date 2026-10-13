* Monitor the training that is running at tmux session `molmoact2_roboeval`.
    * Monitor every 2 hours. 
* If training crashes:
    * Use the latest checkpoint.
    * Find two GPUs that has at least 40GB VRAM left to resume the run.
    * If no GPU with enough VRAM are available, wait till they are available. Come back and check every 2 hours.