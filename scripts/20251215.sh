dirs=(
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_conv0_rmsF_l2T
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_conv0_rmsT_l2T
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_convF_rmsF_l2T
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_convF_rmsT_l2T
)

for dir in "${dirs[@]}"; do
    bash "$dir/step0.sh"
    # bash "$dir/lm_eval.sh"
done

dirs=(
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_conv0_rmsF_l2T
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_conv0_rmsT_l2T
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_convF_rmsF_l2T
/home/hei/RADLADS-paper/scripts/20251215__qwen3-8b_gdn_convF_rmsT_l2T
)

for dir in "${dirs[@]}"; do
    bash "$dir/step1.sh"
    # bash "$dir/lm_eval.sh"
done