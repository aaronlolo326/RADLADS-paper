This config is for testing only ce loss with stage 3 using 2B 3B 4B data mix
Line 10 of step2.sh: without teacher
Line 12 of step2.sh: train stage 3 on top of stage 2's ckpt-final.pth of 20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT 

train command: from RADLADS-paper/, "bash scripts/20260115__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_stage3ce/step2.sh"
lm_eval command: from RADLADS-paper/, "bash scripts/20260115__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_dclm/lm_eval.sh"