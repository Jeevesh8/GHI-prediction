# GHI-prediction

This repository implements some time-series prediction models (CNN-LSTM, Transformer, AR-Net, LSTM) for predicting GHI-values in various ways.

## Dataset

Datasets can be obtained from NSRDB-NREL website :- https://maps.nrel.gov/nsrdb-viewer/

## Preprocessing

Each data file must correspond to an year. You can choose the training , testing and validation files. Various averages are calculated over the data in the training files and subtracted from the original data, obtaining residuals which are encapsulate information relatively(relative to original data) independent of time based phenomenons like seasons, day and night etc. The model is trained on this data. The below command will perform pre-processing :-

```
python DataSet/MakingData.py --csv_prefix <common-prefix-for-all-downloaded-data-files> --tar_dir <directory-to-store-new-data-files>\
                             --tr_start_year <start-year-of-training> --tr_final_year <final-year-of-training>
                             --t_start_year <start-year-of-testing/validation> --t_final_year <final-year-of-testing/valiation>
```

## Training a Model

Run the train.py file to train the model.

```
python train.py --batch_size 256 --seq_len 256 --root_dir <directory-having-Data-files> --tr_start_year <compulsory>
                --tr_final_year <compulsory> --val_start_year <compulsory> -val_final_year  <compulsory>
                --loss <mse|qr_loss> --epochs 10 --lr 0.001 --model <ar_net|trfrmr|cnn_lstm> --ini_len <no-of-columns-in-data-file>
                --final_len 1 --steps 1 --param_file <file-path>
```

final_len denotes <no-of-numbers-to-be-predicted-by-model>.
steps denotes (number of rows to skip after seq_len rows before predicting final_len values -1)
param_file is used to give path name of file where model-parameters are to be stored/retrieved from.
Choose loss as 'mse' to train model to predict values of GHI. For predicting confidence intervals choose 'qr_loss' .
And provide gammas as :- 

```
  --gamma_list [interval_1_start, interval_2_start,...interval_n_start, interval_1_end, interval_2_end... interval_n_end]
```

Example :-

```
  --gamma_list [0.1,0.5,0.9,0.5]
```

