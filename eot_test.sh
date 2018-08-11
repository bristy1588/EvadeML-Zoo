COMMON_DETECTION='--detection FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=bit_depth_3&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=bit_depth_4&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=bit_depth_5&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=median_filter_3_3&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=non_local_means_color_11_3_2&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=non_local_means_color_11_3_4&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=non_local_means_color_13_3_4&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05;'

COMMON_CIFAR='python eot_break.py --dataset_name CIFAR-10 --model_name DenseNet
--nb_examples 200 --robustness none;FeatureSqueezing?squeezer=median_filter_2_2;FeatureSqueezing?squeezer=bit_depth_5;FeatureSqueezing?squeezer=non_local_means_color_13_3_2'

COMMON_IMAGENET='python eot_break.py --dataset_name ImageNet --model_name MobileNet
--nb_examples 200 --robustness none;FeatureSqueezing?squeezer=median_filter_2_2;FeatureSqueezing?squeezer=bit_depth_5;FeatureSqueezing?squeezer=non_local_means_color_11_3_4'


$COMMON_CIFAR $COMMON_DETECTION --result_folder "results/cifar/eot_small" --reg_lambda  200.0 --threshold 0.8 \
   --attacks "pgdli?epsilon=0.012&k=50&a=0.01;"

$COMMON_CIFAR $COMMON_DETECTION --result_folder "results/cifar/eot_large" --reg_lambda  50.0 --threshold 1.0 \
   --attacks "pgdli?epsilon=0.016&k=50&a=0.01;"

$COMMON_IMAGENET $COMMON_DETECTION --result_folder "results/imagenet/eot_small" --reg_lambda  50.0 --threshold 1.0 \
   --attacks "pgdli?epsilon=0.012&k=20&a=0.01;"

$COMMON_IMAGENET $COMMON_DETECTION --result_folder "results/imagenet/eot_large" -reg_lambda  50.0 --threshold 1.0 \
   --attacks "pgdli?epsilon=0.008&k=50&a=0.01;"
