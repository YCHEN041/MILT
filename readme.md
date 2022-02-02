User Instructions:

For eval/test, add flag --train=False; 
For test on test set add flag --test_set=True; (otherwise validation set)

Note that higher accuracy can be achieved by larger networks.

"""
Usage Instructions:

    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=140000 --norm=None --update_batch_size=10 --sghmc_num_sample=5 --sghmc_num_updates=5

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/ --num_filters=32 --sghmc_num_sample=5 --sghmc_num_updates=5

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/ --num_filters=32 --sghmc_num_sample=5 --sghmc_num_updates=5

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True --sghmc_num_sample=5 --sghmc_num_updates=5
"""