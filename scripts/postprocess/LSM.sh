### extrapolation
CUDA_VISIBLE_DEVICES=0 python extrapolation_test.py ./config/LSM/config_cylinder_rRE_rBC.yaml -c rRE | tee ./log/extp/LSM_extrapolation_rRE.out
CUDA_VISIBLE_DEVICES=0 python extrapolation_test.py ./config/LSM/config_ircylinder_irRE_irBC.yaml -c irRE | tee ./log/extp/LSM_extrapolation_irRE.out

### plotting

