SERVER_HOST=http://172.17.0.1:29500
DATASET_URL=http://172.17.0.1:9000/mnist-dataset/dataset_dl_1.tar.gz
WORKER_HOST=http://localhost:8080
curl -X POST \
     -d '{"value":{"batch_sz_train": 32, "epoch_n": 32, "apihost": "'$SERVER_HOST'","update_intv": 8, "dataset_url": "'$DATASET_URL'","device": "cpu"}}' \
     -H 'Content-Type: application/json' $WORKER_HOST/run