# Dataset — CSE-CIC-IDS2018 (not included in this repository)

The dataset is **about 6.5 GB**, so it is **not shipped** with the source code.
Download it directly from the official public source using the steps below.
These are the same steps used for this project (Ubuntu 24.04 VM, Python virtual
environment `bigdata-env`).

| | |
|---|---|
| **Dataset** | CSE-CIC-IDS2018 — Communications Security Establishment (CSE) & Canadian Institute for Cybersecurity (CIC) |
| **Official page** | <https://www.unb.ca/cic/datasets/ids-2018.html> |
| **Download location (AWS S3, public)** | `s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/` |
| **Account needed?** | No — public bucket, accessed with `--no-sign-request` |
| **Total size** | ~6.5 GB (10 CSV files) |
| **Files used by the model** | 9 of the 10 (see table below) |

> **Windows users:** download the data **inside the Ubuntu machine/VM** where
> Hadoop and Spark run. Do not download on Windows and copy it across — the
> files are large and the project expects them at the Linux path
> `~/cybersecurity_project/data/CSE_CIC_IDS2018/`. (An Ubuntu VirtualBox VM was
> used for this project; WSL2 Ubuntu should also work but was not tested.)

---

## Expected folder layout after download

```
~/cybersecurity_project/
└── data/
    ├── README.md                     ← this file
    └── CSE_CIC_IDS2018/
        ├── Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
        ├── Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv   ← downloaded but NOT used
        ├── Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
        ├── Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
        └── Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
```

---

## Step 1 — Activate the project environment

```bash
source ~/bigdata-env/bin/activate        # adjust the path if your venv lives elsewhere
```

Your prompt should start with `(bigdata-env)`.

## Step 2 — Install the AWS CLI

Either option works; no AWS account or keys are needed.

```bash
# Option A — inside the virtual environment
pip install awscli

# Option B — system-wide on Ubuntu
sudo snap install aws-cli --classic
```

Check: `aws --version`

## Step 3 — List the files in the public bucket

```bash
aws s3 ls --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
```

Expected output (size in bytes):

```
352368373  Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
333723605  Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
382840456  Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
4054925350 Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
107842858  Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
375945899  Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
382636202  Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
358223333  Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
328893673  Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
209249758  Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
```

> The quotes around the S3 path are required because the folder name contains
> spaces. "Thuesday" is spelled that way in the original dataset.

## Step 4 — Download into the project data folder

```bash
mkdir -p ~/cybersecurity_project/data/CSE_CIC_IDS2018
cd ~/cybersecurity_project/data/CSE_CIC_IDS2018

aws s3 cp --no-sign-request --recursive \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" .
```

This downloads all 10 files (~6.5 GB). Make sure the VM has enough free disk
space; the files are stored again in HDFS in Step 6, so allow roughly twice
that, plus space for the cleaned Parquet output.

**To save ~3.8 GB**, you can skip the one file the model does not use:

```bash
aws s3 cp --no-sign-request --recursive \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" . \
  --exclude "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
```

## Step 5 — Verify the download

```bash
ls -lh
du -sh .
```

Expected (all 10 files):

```
total 6.5G
337M  Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
319M  Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
366M  Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
3.8G  Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
103M  Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
359M  Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
365M  Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
342M  Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
314M  Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
200M  Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
```

If a file is noticeably smaller than listed, the download was interrupted. Run
the Step 4 command again; it re-downloads the files.

## Step 6 — Copy the raw files into HDFS

HDFS must be running (`start-dfs.sh`; `jps` should show NameNode and DataNode).

```bash
hdfs dfs -mkdir -p /cybersecurity/CSE_CIC_IDS2018/raw
hdfs dfs -put ~/cybersecurity_project/data/CSE_CIC_IDS2018/*.csv /cybersecurity/CSE_CIC_IDS2018/raw/
```

Verify:

```bash
hdfs dfs -ls -h /cybersecurity/CSE_CIC_IDS2018/raw
hdfs dfs -du -h /cybersecurity/CSE_CIC_IDS2018/raw
```

Expected: `Found 10 items`, with sizes from 102.8 M (Thursday-01-03) to 3.8 G
(Thuesday-20-02).

> The message `WARN util.NativeCodeLoader: Unable to load native-hadoop library`
> is normal on this setup and can be ignored.

---

## Which files the model uses, and why one is excluded

| File | Size | Used? |
|---|---|---|
| Friday-02-03-2018 | 337 MB | ✅ |
| Friday-16-02-2018 | 319 MB | ✅ |
| Friday-23-02-2018 | 366 MB | ✅ |
| **Thuesday-20-02-2018** | **3.8 GB** | ❌ **excluded** |
| Thursday-01-03-2018 | 103 MB | ✅ |
| Thursday-15-02-2018 | 359 MB | ✅ |
| Thursday-22-02-2018 | 365 MB | ✅ |
| Wednesday-14-02-2018 | 342 MB | ✅ |
| Wednesday-21-02-2018 | 314 MB | ✅ |
| Wednesday-28-02-2018 | 200 MB | ✅ |

When all 10 files were read together (16,233,002 rows, 80 columns), Spark
reported `CSV header does not conform to the schema` for the
`Thuesday-20-02-2018` file (it found `Idle Mean` where `Label` was expected),
so its columns do not line up with the other nine files. To keep the data
consistent, this file is excluded. The list of files used is fixed in
`GOOD_FILES` in `scripts/common.py`, so the scripts ignore this file even if it
is in the folder.

**Nine-file subset (what the model is trained and tested on):**

| Item | Count |
|---|---|
| Rows read | 8,284,254 |
| Repeated header rows removed (`Label == "Label"`) | 59 |
| Cleaned rows | 8,284,195 |
| Columns | 80 (78 numeric features + `Timestamp` + `Label`) |
| Benign (label 0) | 6,112,151 |
| Attack (label 1, all other labels) | 2,172,044 |

Attack labels in the subset: Bot, DoS attacks-SlowHTTPTest, DoS attacks-Hulk,
SQL Injection, Brute Force-XSS, Brute Force-Web, DoS attacks-Slowloris,
DoS attacks-GoldenEye, SSH-Bruteforce, FTP-BruteForce, DDOS attack-LOIC-UDP,
DDOS attack-HOIC, Infilteration (spelled as in the dataset).

---

## Next step

Once the files are in place, continue with **Step 9** of the main
[`README.md`](../README.md) (run `scripts/data_preprocessing.py` first).

## Citation

> I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New
> Intrusion Detection Dataset and Intrusion Traffic Characterization,"
> *Proc. 4th Int. Conf. on Information Systems Security and Privacy (ICISSP)*,
> 2018, pp. 108–116.

Use of the dataset is subject to the terms on the official CIC dataset page.
