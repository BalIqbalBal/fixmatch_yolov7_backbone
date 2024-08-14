## Instalasi Dependensi ##

Untuk menginstall seluruh dependensi yang digunakan untuk projek ini maka lakukan:

```pip install -r requirements.txt```

Apabila menggunakan conda maka gunakan:

```conda create --name <env> --file <this file> ```

note: Dependensi ini bukan merupakan dependensi minimal, terdapat beberapa package yang sebenarnya tidak diperlukan.

lalu install package tambahan dengan:

```pip install fixmatch_yolov7/packages/rhea_pkg ```

```pip install fixmatch_yolov7/packages/polyaxon_schemas_pkg```

```pip install fixmatch_yolov7/packages/polyaxon_client_pkg```

## Training ##

untuk mengatur config dataset silahkan ubah pengaturannya di file ./datasets/config.py

```python main.py --data-dir ./fixmatch_yolov7/data --batch-size 4 --device cpu --pbar --checkpoint-interval 5 --out-dir /out --num-labeled 0.5 --num-validation 0.1 --save```

