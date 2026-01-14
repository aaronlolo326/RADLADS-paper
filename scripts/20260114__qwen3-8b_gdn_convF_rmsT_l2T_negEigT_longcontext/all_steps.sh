# run from RADPADS-paper/
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

# bash scripts/$RUN_NAME/step0.sh
# bash scripts/$RUN_NAME/step1.sh
bash scripts/$RUN_NAME/torchrun_step2.sh