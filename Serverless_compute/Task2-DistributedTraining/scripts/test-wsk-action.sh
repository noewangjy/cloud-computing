SERVER_HOST=http://172.17.0.1:29500 
DATASET_URL=http://172.17.0.1:9000/mnist-dataset/dataset_dl_1.tar.gz
wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost $SERVER_HOST \
    --param update_intv 8 \
    --param dataset_url $DATASET_URL \
    --param device cpu