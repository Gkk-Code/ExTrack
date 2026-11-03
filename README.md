# Evolving Historical Prompt Memory for Day-Night Object Tracking

### Data Preparation
**Put the tracking datasets in ./data. It should look like this:**
   ```
   ${extrack_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
        -- bdd100k_night
            |-- images
            |-- annotations
            ...
       
        -- shift_night
            |-- 0b3d-e686
            |-- 0b4d-d96f
            ...
       
   ```


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### Training
Dowmload the pretrained [[Baidu Drive]](https://pan.baidu.com/s/1pMc3SzshxhLTGTF99GrvMg?pwd=6wtc). 
Put the pretrained models in [./pretrained](pretrained), it should be like:

   ```
   ${ExTrack_ROOT}
    -- pretrained
        -- itpn
            |-- fast_itpn_base_clipl_e1600.pt
```
Then, run the following command:
```
python tracking/train.py --script extrack --config extrack_b224 --save_dir ./output --mode multiple --nproc_per_node 5
```



### Testing
Put the trained  weights on `<PATH_of_extrack>/output/checkpoints/train/extrack/extrack_b224`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- UAVDark135 or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py extrack extrack_b224 --dataset uavdark135 --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- DarkTrack2021
```
python tracking/test.py extrack extrack_b224 --dataset_name darktrack2021 --threads 0 --num_gpus 1
python tracking/analysis_results.py # need to modify tracker configs and names
```
- NAT2021
```
python tracking/test.py extrack extrack_b224 --dataset nat2021 --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- NAT2024-1
```
python tracking/test.py extrack extrack_b224 --dataset nat2021 --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
