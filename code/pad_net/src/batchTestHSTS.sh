declare -a conditions=("l2_10k" "l1_10k" "ssim_10k" "msssim_10k" "msssimL1_10k" "msssimL2_10k")
for post_fix in "${conditions[@]}"
do
    python test.py -m ../data10k/solver_${post_fix}_iter_18000.caffemodel -i ../test/hsts -o ../test/hsts_${post_fix}
done

declare -a conditions=("msssimL2_10k_fine_tune" "msssimL2_10k_fine_tune_0.5")
# declare -a conditions=("msssimL2_10k_fine_tune_0.3" "msssimL2_10k_fine_tune_0.7" "msssimL2_10k_fine_tune_0.9")
for post_fix in "${conditions[@]}"
do
    python test.py -m ../data10k/solver_${post_fix}_iter_9000.caffemodel -i ../test/hsts -o ../test/hsts_${post_fix}
done