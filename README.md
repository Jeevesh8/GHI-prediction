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

final_len is equal to number of outputs of the model in case no interval is being predicted. If intervals are being predicted, the number of outputs of the model will automatically be changed to final_len* length(gamma_list).

steps denotes (number of rows to skip after seq_len rows before predicting final_len values)

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

You can also specify 1 element list if you want to train entire model to predict one side of interval only.

## Inference
The script Infer.py can be run in three ways :-

1.) To get average loss over a Dataset.

Example :-
```
python Infer.py --mode avg_loss --loss <mse|mae|mbe|mape|qr_loss> --model <ar_net|trfrmr|cnn_lstm> --ini_len <same-as-in-train.py> --param_file
                <same-as-train.py> --steps <same-as-in-train.py> --final_len <same-as-in-train.py> --seq_len <same-as-in-train.py>
                --root_dir <dir-of-test-files> --test_start_year <int> --test_final_year <int>
```

Also, you must specify ```--gamma_list``` if it was used during training. Optionally, you can also specify ```--mask_gamma_list``` to calculate q_risk over few intervals only. For example usage of this see "Example commands(2)" below.

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

The last two modes can also be run for confidence interval predictions. In that case, we need to provide corresponding param_file. It will give outputs in the form same as (A) if model predicts multiple intervals simultaneously. 

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
## Using smart_persistence.py (Baseline Model)

1.) Provides smart persistence [Pedro and Coimbra, 2012](https://www.sciencedirect.com/science/article/abs/pii/S0038092X12001429) predictions and accuracy metrics.

```
python smart_persistence.py --loss <mse|mae|mbe|mape> --tr_start_year <training-start-year>
                            --tr_final_year <training-final-year> --test_start_year <year>
                            --test_final_year <year> --root_dir <directory-having-data-files>
                            --steps <steps-b/w-consecutive-preds-for-test-data> --get_preds
```

Omitting the ```--get_preds``` flag will cause only the loss to print.

## Example Commands :- 

1.) Training :-
```
python train.py --root_dir '/content/drive/My Drive/SolarDataIndia/SolarData(In)' \
                --tr_start_year 0 --tr_final_year 12 --val_start_year 13 --val_final_year 14 \
                --model trfrmr --ini_len 15 --final_len 12 --optimizer RAdam \
                --param_file '/content/drive/My Drive/trfrmr_12_step_ahead.param' \
                --loss qr_loss --gamma_list 0.95 0.9 0.5 0.05 0.1 0.5 --batch_size 128
```

2.) Getting Q-Risk over test set:-
```
python Infer.py --model trfrmr --mask_gamma_list 1 1 1 0 0 0 --ini_len 15 --final_len 12 \ 
                --loss qr_loss --param_file '/content/drive/My Drive/trfrmr_12_step_ahead.param' \
                --root_dir '/content/drive/My Drive/SolarDataIndia/SolarData(In)' \
                --test_start_year 13 --test_final_year 14 --gamma_list 0.95 0.9 0.5 0.05 0.1 0.5
```
Q-Risk is calculated over upper limits of intervals only due to mask_gamma_list. 

3.) Getting predictions for 12 March , 2014 at 8:30 a.m. to 12 steps ahead :-
```
python Infer.py --mode predict_next --model trfrmr --ini_len 15 --final_len 12\
                 --param_file '/content/drive/My Drive/trfrmr_12_step_ahead.param'\
                 --root_dir '/content/drive/My Drive/SolarDataIndia/SolarData(In)' \
                 --test_year 14 --date_lis 2014 3 12 8 30 --gamma_list 0.95 0.9 0.5 0.05 0.1 0.5
```
The first 12 values are corresponding to gamma value of 0.95 for all the twelve GHI values ahead. The next 12 values for gamma 0.9 and so on..


4.) To repeat the above thing 10 times, each time beginning with 1 hour ahead

```
python Infer.py --mode predict_list --model trfrmr --ini_len 15 --final_len 12\
                 --param_file '/content/drive/My Drive/trfrmr_12_step_ahead.param'\
                 --root_dir '/content/drive/My Drive/SolarDataIndia/SolarData(In)' \
                 --test_year 14 --times_to_run 10 --gamma_list 0.95 0.9 0.5 0.05 0.1 0.5
```

**NOTE** :- Currently you can't predict GHI for more future steps(parallely) than you trained for as you'd need weather data of those steps and hence just using prediction for previous n steps to predict for n steps after those will not work. 

All the years, i.e., ```--tr_start_year, --val_final_year``` etc. are integers from 0 to n-1 where n is the number of years(or files) in your root directory. Each file must correspond to single year.
