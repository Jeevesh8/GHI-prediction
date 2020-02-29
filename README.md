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

final_len denotes number of consecutive GHI predictions to be made by model in one forward pass.
steps denotes (number of rows to skip after seq_len rows before predicting final_len values -1)
param_file is used to give path name of file where model-parameters are to be stored/retrieved from.
Choose loss as 'mse' to train model to predict values of GHI. For predicting confidence intervals choose 'qr_loss' .
And provide gammas as :- 

```
  --gamma_list interval_1_start interval_2_start...interval_n_start interval_1_end interval_2_end... interval_n_end          ....(A)
```

Example :-

```
  --gamma_list 0.1 0.5 0.9 0.5
```

You can also specify 1 element list if we want to train entire model to predict one value only.

## Inference
The script Infer.py can be run in three ways :-

1.) To get average loss over a Dataset.

Example :-
```
python Infer.py --mode avg_loss --loss <rmse|mae|mbe|mape> --model <ar_net|trfrmr|cnn_lstm> --ini_len <same-as-in-train.py> --param_file
                <same-as-train.py> --steps <same-as-in-train.py> --final_len <same-as-in-train.py> --seq_len <same-as-in-train.py>
                --root_dir <dir-of-test-files> --test_start_year <int> --test_final_year <int>
```
2.) To get prediction at a particular time.

Example :-

```
python Infer.py --mode predict_next --model <ar_net|trfrmr|cnn_lstm> --ini_len <same-as-in-train.py> --steps <same-as-in-train.py>
                --final_len <same-as-in-train.py> --seq_len <same-as-in-train.py> --root_dir <dir-in-which-file-having-just-prev-values>
                --test_year <year-having-test-date> --date_lis Year Month Day Hour Minute
```
3.) To get predicted values by running models n times.

```
python Infer.py --mode predict_list --model <ar_net|trfrmr|cnn_lstm> --ini_len <same-as-in-train.py> --steps <same-as-in-train.py>
                --final_len <same-as-in-train.py> --seq_len <same-as-in-train.py> --root_dir <directory-having-test-file>
                --test_year <year-having-test-date> --times_to_run <no-of-times-to-run-the-model>
```

The last two modes can also be run for confidence interval predictions. In that case, we need to provide corresponding param_file. And also add ```--interval True``` in the command. It will give outputs in the form same as (A) if model predicts multiple intervals simultaneously. 

## Getting right values from shifted ones

The following command will allow you to shift back GHI-values(to counteract the effect of preprocessing). It can be used in two ways:-

1.) Provide dates and predictions, actual values of model(interval predictions should also be sent) 
```
python shift_ghi.py --ghi val1 val2 val3 val4 
                    --date_lis date_1_year date_1_month.. date1_minute date_2_year date_2_month...date_4_minute
```
2.) Provide a pickle file storing a dictionary with one key as 'times_lis' and other keys corresponding to predictions of intervals, actual values in ```Data<Year>.csv``` files, etc. All the lists in dic[key] and times_lis can be nested, but must contain as many values as the number of dates in times_lis.

```
python shift_ghi.py --ghi_time_file <pickle-file-path> --write_to <append|out-pickle-file-path>
```
