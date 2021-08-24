# Run AlphaFolder on AWS Deep Learning EC2

## Launch AWS Deep Learning EC2 Instance

In this section, we will demonstrate step by step how to set up an AWS EC2 using one of pre-built Deep Learning AMI from AWS. It has most of the dependcies for AlphaFold installed on the AMI to save lots of time.

1. In the AWS region of you choice, launch a new EC2 instance with Deep Learning AMI by searching `Deep Learning AMI`. In the steps below, we will use a Deep Learning AMI based on Ubuntu 18.04.

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/ec2ami.png">

2. Choose p3.2xlarge with 1 GPU as the instance type.

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/p32xlarge.png">

3. Set the system volume to 200GB and add one new data volume of 3TB in size  

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/ebsvolume.png">

4. Make sure the security group settings allow you to access the EC2 instance and it could reach the internet to install AlphaFold and other packages. Launch the EC2.

5. Wait for the EC2 instance to become ready and SSH to the EC2 terminal.

6. (Optional) If you have internal security controls that is required, install them now.

## Install AlphaFold

1. SSH into the EC2 terminal. first update all packages to the latest on the EC2

```
sudo apt update
```

2. Mount the data volume to folder `/data`. 

First find the device list on the instance
```
lsblk
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/lsblk.png">

Make sure the volume is not mount on other device (use the deive name matching what you get from `lsblk`) which should only have data in the output
```
sudo file -s /dev/xvdb
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/sudofile.png">

Create a new file system and mount it to `/data` folder
```
sudo mkfs.xfs /dev/xvdb
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chown ubuntu:ubuntu -R /data
df -h /data
```

3. Install AlphaFold dependecies and other tools 

```
sudo apt install aria2 rsync git vim wget tmux tree -y
```

4. Create working folders, clone AlphaFold code from github and download the data using provided scripts in the background. We'll use the new volume exclusively.

```
cd /data
mkdir -p /data/af_download_data
mkdir -p /data/output/alphafold
mkdir -p /data/input
git clone https://github.com/deepmind/alphafold.git
```

5. AlphaFold needs multiple genetic (sequence) databases to run. We will download them using the provided script. 

``` 
nohup /data/alphafold/scripts/download_all_data.sh /data/af_download_data &
```

The whole download process could take over 10 hours, wait for it to finish ...

You could use the command below to monitor the download and unzip process
```
du -sh /data/af_download_data/*
```

Once all the download process finishes, you should have the following files in your `/data/af_download_data` folder  

```
$DOWNLOAD_DIR/                             # Total: ~ 2.2 TB (download: 438 GB)
    bfd/                                   # ~ 1.7 TB (download: 271.6 GB)
        # 6 files.
    mgnify/                                # ~ 64 GB (download: 32.9 GB)
        mgy_clusters_2018_12.fa
    params/                                # ~ 3.5 GB (download: 3.5 GB)
        # 5 CASP14 models,
        # 5 pTM models,
        # LICENSE,
        # = 11 files.
    pdb70/                                 # ~ 56 GB (download: 19.5 GB)
        # 9 files.
    pdb_mmcif/                             # ~ 206 GB (download: 46 GB)
        mmcif_files/
            # About 180,000 .cif files.
        obsolete.dat
    small_bfd/                             # ~ 17 GB (download: 9.6 GB)
        bfd-first_non_consensus_sequences.fasta
    uniclust30/                            # ~ 86 GB (download: 24.9 GB)
        uniclust30_2018_08/
            # 13 files.
    uniref90/                              # ~ 58 GB (download: 29.7 GB)
        uniref90.fasta
```

6. Update `/data/alphafold/docker/run_docker.py` to make the configuration matching the local path

```
vim /data/alphafold/docker/run_docker.py
```

With the folders we created in `/data` folder, the configurations will look like the following. If you have setup different folder structure in your instance, set it accordingly.
```
#### USER CONFIGURATION ####

# Set to target of scripts/download_all_databases.sh
DOWNLOAD_DIR = '/data/af_download_data'

# Name of the AlphaFold Docker image.
docker_image_name = 'alphafold'

# Path to a directory that will store the results.
output_dir = '/data/output/alphafold'
```

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/config.png">

7. Confirm the NVidia container kit is installed. You should see somthing like the screen below

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/nvidia.png">


8. Build AlphaFold docker image, make sure local path is `/data/alphafold` as there is a `.dockerignore` under that folder. You should see the new docker image after build.

```
cd /data/alphafold
docker build -f docker/Dockerfile -t alphafold .
docker images
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/dockerimage.png">

9. Use PIP to install all python dependencies
```
pip3 install -r /data/alphafold/docker/requirements.txt
```

10. Go to [CASP14 target list](https://www.predictioncenter.org/casp14/targetlist.cgi) and copy the sequence from the [plain text link for T1050](https://www.predictioncenter.org/casp14/target.cgi?target=T1050&view=sequence). 

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/casp14.png">

Copy it into new `.fasta` files in `/data/input` folder and save it.

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/fasta.png">

You can create a few more input `.fasta` files for testing.


## (Optional) Install CloudWatch monitoring for GPU

1. Create EC2 role and attach to EC2

2. Change the region in gpumon.py if your instance is NOT in us-east-1. Provide a new Namespace like `AlphaFold`

```
vim ~/tools/GPUCloudWatchMonitor/gpumon.py
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/gpumon.png">

3. Launch gpumon
```
source activate python3
python ~/tools/GPUCloudWatchMonitor/gpumon.py &
```

## Use AlphaFold for prediction

1. Use the following command to run prediction of protein sequence from `/data/input/T1050.fasta`

```
nohup python3 /data/alphafold/docker/run_docker.py --fasta_paths=/data/input/T1050.fasta --max_template_date=2020-05-14 &
```
