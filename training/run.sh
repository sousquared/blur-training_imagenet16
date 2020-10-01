# python main.py --mode normal -e 60 -b 64 --lr 0.01 -n normal_b64_lr1e-2
python main.py --mode normal -e 60 -b 64 --lr 0.01 -n normal_norm-in


# blur-half-data
python main.py --mode blur-half-data -s 1 -n blur-half-data_s1_norm-in
python main.py --mode blur-half-data -s 2 -n blur-half-data_s2_norm-in
python main.py --mode blur-half-data -s 3 -n blur-half-data_s3_norm-in
python main.py --mode blur-half-data -s 4 -n blur-half-data_s4_norm-in


# blur-half-epochs
python main.py --mode blur-half-epochs -s 1 -n blur-half-epochs_s1_norm-in
python main.py --mode blur-half-epochs -s 2 -n blur-half-epochs_s2_norm-in
python main.py --mode blur-half-epochs -s 3 -n blur-half-epochs_s3_norm-in
python main.py --mode blur-half-epochs -s 4 -n blur-half-epochs_s4_norm-in


# blur-step
python main.py --mode blur-step -n blur-step_norm-in
# python main.py --mode blur-step -n blur-step_from_s5


# blur-step-cbt
## cbt-rate 0.9
#python main.py --mode blur-step-cbt --init-s 1 --cbt-rate 0.9 -n blur-step-cbt_decay9e-1_init-s1
#python main.py --mode blur-step-cbt --init-s 2 --cbt-rate 0.9 -n blur-step-cbt_decay9e-1_init-s2
#python main.py --mode blur-step-cbt --init-s 3 --cbt-rate 0.9 -n blur-step-cbt_decay9e-1_init-s3
#python main.py --mode blur-step-cbt --init-s 4 --cbt-rate 0.9 -n blur-step-cbt_decay9e-1_init-s4
## cbt-rate 0.8
#python main.py --mode blur-step-cbt --init-s 1 --cbt-rate 0.8 -n blur-step-cbt_decay8e-1_init-s1
#python main.py --mode blur-step-cbt --init-s 2 --cbt-rate 0.8 -n blur-step-cbt_decay8e-1_init-s2
#python main.py --mode blur-step-cbt --init-s 3 --cbt-rate 0.8 -n blur-step-cbt_decay8e-1_init-s3
#python main.py --mode blur-step-cbt --init-s 4 --cbt-rate 0.8 -n blur-step-cbt_decay8e-1_init-s4