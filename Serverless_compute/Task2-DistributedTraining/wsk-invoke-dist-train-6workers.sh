wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.36.193:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.36.193:8000/dataset_dl_1.tar.gz' \
    --param device cpu

wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.36.193:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.36.193:8000/dataset_dl_2.tar.gz' \
    --param device cpu

wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.36.193:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.36.193:8000/dataset_dl_3.tar.gz' \
    --param device cpu

wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.36.193:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.36.193:8000/dataset_dl_4.tar.gz' \
    --param device cpu

wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.36.193:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.36.193:8000/dataset_dl_5.tar.gz' \
    --param device cpu

wsk action invoke dist-train \
    --param batch_sz_train 32 \
    --param epoch_n 8 \
    --param apihost 'http://192.168.36.193:29500' \
    --param update_intv 8 \
    --param dataset_url 'http://192.168.36.193:8000/dataset_dl_6.tar.gz' \
    --param device cpu