# alphafold


### Launch EC2 Instance

1. In the AWS region of you choice, launch a new EC2 instance with Deep learning AMI by `Deep learning AMI`. In this case, we are using a Deep learning AMI based on Amazon Linux 2.

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/ec2ami.png">

2. Choose p3.2xlarge with 1 GPU as the instance type

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/p32xlarge.png">

3. Set the system volume to 200GB and add one new data volume of 3TB in size  

<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/ebsvolume.png">

4. Make sure the security group settings allow you to access the EC2 instance and it could reach the internet to install AlphaFolder and other packages. Launch the EC2.

5. Wait for the EC2 instance to become ready and SSH to the EC2 terminal.


### install AlphaFold

1. Update all packages 
```
sudo yum update
```

2. Mount the data volume with the following commands

Find the device list on instance
```
lsblk
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/lsblk.png">

Make sure there is no mount on the device yet (use the deive name matching what you get from `lsblk`)
```
sudo file -s /dev/xvdb
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/sudofiles.png">

Create a file system and mount it to `/data` folder

```
sudo mkfs.xfs /dev/xvdb
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chown ec2-user:ec2-user -R /data
```

2. Install AlphaFold dependecies

```
sudo rpm -i http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum install aria2 rsync git vim wget tmux -y
```

3. Create working folders, clone AlphaFold code from github and download the data using provided scripts in the background. We'll use the new volume exclusively.

```
cd /data
mkdir -p /data/af_download_data
mkdir -p /data/output/alphafold
git clone https://github.com/deepmind/alphafold.git
pip3 install -r alphafold/docker/requirements.txt
nohup alphafold/scripts/download_all_data.sh /data/af_download_data &
```

The whole download process could take over a few hours, sit back and wait for it to finish ...

4. Change /data/alphafold/docker/run_docker.py

```
vim /data/alphafold/docker/run_docker.py
```
And make the cinfuguration to match the local folders

```
#### USER CONFIGURATION ####

# Set to target of scripts/download_all_databases.sh
DOWNLOAD_DIR = /data/af_download_data

# Name of the AlphaFold Docker image.
docker_image_name = 'alphafold'

# Path to a directory that will store the results.
output_dir = '/data/output/alphafold'
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/config.png">

5. Make sure the NVidia container kit is installed per this [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). You should see somthing like the screen below

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
<img src="https://raw.githubusercontent.com/pkuqiwang/alphafold/main/images/nvidia.png">


6. Buidl docker image
```
docker build -f docker/Dockerfile -t alphafold .
docker images
```

7. Download test files

```
wget -O T1050.fasta https://www.predictioncenter.org/casp14/target.cgi?target=T1050&view=sequence
```