## Ciao 数据集：不加任何社交图7
#python main.py --dataset Ciao        --gat_weight 0.1  Epoch 21 validation: MAE: 0.6339, RMSE: 0.8308, CL Loss: 1.9889, Best MAE: 0.6125, test_MAE: 0.6137, test_RMSE: 0.8067, test_CL Loss: 1.9701
#
### Ciao 数据集：使用随机社交图7
#python main.py --dataset Ciao    --gat_weight 0.75   MAE: 0.6339, RMSE: 0.7844,
# python main.py --dataset Ciao    --gat_weight 0.8   MAE: 0.6216, RMSE: 0.7721,
# python main.py --dataset Ciao    --gat_weight 0.85   MAE: 0.6229, RMSE: 0.7787,
# python main.py --dataset Ciao    --gat_weight 0.9   MAE: 0.5934, RMSE: 0.6932,
# python main.py --dataset Ciao    --gat_weight 0.95   MAE: 0.6032, RMSE: 0.7478,
## Ciao 数据集：不加任何社交图8
#python main.py --dataset Epinions     --gat_weight 0.75  --epoch 10    0.8308 1.1945
### Ciao 数据集：使用随机社交图8
#python main.py --dataset Epinions    --gat_weight 0.8  --epoch 10    0.8194  1.1647
#9
#python main.py --dataset Epinions    --gat_weight 0.85 --epoch 10  --device 1   0.8087  1.1386
#python main.py --dataset Epinions    --gat_weight 0.9  --epoch 10  --device 1   0.7889  1.0446
#10
#python main.py --dataset Epinions    --gat_weight 0.95  --epoch 10  --device 2   0.8003  1.1158

#
# Ciao 数据集：不加任何社交图0
#python main.py --dataset Ciao        --embed_dim 128   Epoch 7 validation: MAE: 0.6742, RMSE: 0.8616, CL Loss: 6.0173, Best MAE: 0.6742, test_MAE: 0.6503, test_RMSE: 0.8303, test_CL Loss: 6.1014
#
### Ciao 数据集：使用随机社交图0
#python main.py --dataset Ciao        --embed_dim 128 --use_social --random_social  Epoch 8 validation: MAE: 0.9251, RMSE: 1.0660, CL Loss: 7.1010, Best MAE: 0.7391, test_MAE: 0.8977, test_RMSE: 1.0360, test_CL Loss: 7.1423

## Epinions 数据集：不加任何社交图1
#python main.py --dataset Epinions    --device 1  --embed_dim 128

# Epinions 数据集：使用随机社交图2
#python main.py --dataset Epinions   --device 1   --embed_dim 128 --use_social --random_social

#3
#python main.py --dataset Ciao    --device 2    --use_social --embed_dim 32   Epoch 79 validation: MAE: 0.6860, RMSE: 0.8657, CL Loss: 2.1680, Best MAE: 0.6848, test_MAE: 0.6666, test_RMSE: 0.8452, test_CL Loss: 2.1481
#
#python main.py --dataset Ciao     --device 2      --use_social --embed_dim 64  Epoch 34 validation: MAE: 0.6942, RMSE: 0.8735, CL Loss: 2.2696, Best MAE: 0.6705, test_MAE: 0.6701, test_RMSE: 0.8453, test_CL Loss: 2.2865
#
#python main.py --dataset Ciao    --device 2       --use_social --embed_dim 256 [TRAIN] Epoch 48/80, Batch 100, Loss: 1.0096, CL Loss: 2.3877, MAE: 0.7019, RMSE: 0.8780, Avg Loss: 0.9250, Avg CL Loss: 1.8684, Avg MAE: 0.7472, Avg RMSE: 0.9595
#
### Epinions 数据集4
#python main.py --dataset Epinions  --device 3     --use_social --embed_dim 32
#5
#python main.py --dataset Epinions  --device 3     --use_social --embed_dim 64
#6
#python main.py --dataset Epinions  --device 3     --use_social --embed_dim 256

#11
#python main.py --dataset Ciao   --baseline_als  --epoch 10  --device 2
#12
#python main.py --dataset Epinions   --baseline_als  --epoch 10  --device 3
#13
python main.py --dataset Epinions   --baseline_nmf  --epoch 10  --device 3
#[ALS] MAE=3.2360, RMSE=3.3804

#[NMF-filled] MAE=0.7639, RMSE=1.2398
#Val MAE: 0.6876, Val RMSE: 0.9169
#[User‐CF K=20] MAE=0.8682, RMSE=1.1433