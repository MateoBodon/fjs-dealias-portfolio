# Cloud Runner — AWS EC2
_Last updated: 2025-11-05_

## Quick facts
- Region: `us-east-1`
- Instance: `i-075b6e3853fe2349e` (`ec2-3-236-225-54.compute-1.amazonaws.com`)
- Recommended type for heavy calibrations: `c7a.32xlarge` (128 vCPUs, ample memory). Override with `INSTANCE_TYPE=c7a.32xlarge` when running `scripts/aws_provision.sh`.
- SSH user / key: `ubuntu` with `~/.ssh/mateo-us-east-1-ec2-2025`
- IAM role on instance: `EC2AdminRole`
- Subnet: `subnet-0929c64978f562f59`
- Artifact bucket: `s3://fjs-artifacts-mab-prod/` (`docs/`, `reports/`, etc.)

Setup reports generated on 2025-11-05 were uploaded to:
- `s3://fjs-artifacts-mab-prod/docs/fjs-cloud-setup-pre-2025-11-05.md`
- `s3://fjs-artifacts-mab-prod/docs/fjs-ec2-setup-2025-11-05.md`

The bucket now enforces AES256 default encryption and a lifecycle policy that (i) aborts multipart uploads after 7 days, (ii) moves noncurrent versions to Glacier Instant Retrieval after 60 days and expires them after 365 days (keeping the latest three), and (iii) transitions current objects to Intelligent-Tiering after 30 days.

---

## Local prerequisites
1. **AWS CLI v2** installed on your workstation. Verify with:
   ```bash
   aws --version
   ```
2. **`fjs-prod` profile** configured in `~/.aws/credentials` and `~/.aws/config`. Example:
   ```ini
   [fjs-prod]
   aws_access_key_id = ...
   aws_secret_access_key = ...
   
   [profile fjs-prod]
   region = us-east-1
   output = json
   ```
3. **SSH key** present at `~/.ssh/mateo-us-east-1-ec2-2025` with permissions `chmod 400`.

Optional exports for convenience:
```bash
export AWS_PROFILE=fjs-prod
export AWS_REGION=us-east-1
export KEY_PATH="$HOME/.ssh/mateo-us-east-1-ec2-2025"
export INSTANCE_USER=ubuntu
export INSTANCE_DNS=ec2-3-236-225-54.compute-1.amazonaws.com
export BUCKET=fjs-artifacts-mab-prod
```

Confirm identity and bucket guardrails:
```bash
aws --region "$AWS_REGION" sts get-caller-identity
aws --region "$AWS_REGION" s3api get-bucket-encryption --bucket "$BUCKET"
aws --region "$AWS_REGION" s3api get-bucket-lifecycle-configuration --bucket "$BUCKET"
```

---

## Connecting to the EC2 runner
```bash
ssh -o StrictHostKeyChecking=accept-new \
    -i "$KEY_PATH" \
    "${INSTANCE_USER}@${INSTANCE_DNS}"
```

Notes:
- The host key is accepted automatically on first connect (`StrictHostKeyChecking=accept-new`).
- The AWS CLI v2 binary lives at `/usr/local/bin/aws` on the instance (installed from the official zip).
- `/data` is a writable directory on the root volume; use it for scratch space if needed.

---

## Python environment
- Micromamba root: `~/.local/share/mamba`
- Primary environment: `fjs` (Python 3.11) with NumPy, SciPy, pandas, pyarrow, scikit-learn, boto3, s3fs, dask, distributed, ray, tqdm, click, etc.
- Thread caps (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) are pinned to 1 in `~/.bashrc`.
- System telemetry tools (`sysstat`, `htop`) are installed automatically; per-run metrics are captured locally (see below).

Common workflows:
```bash
# Inspect environments
MICROMAMBA_ROOT_PREFIX="$HOME/.local/share/mamba" micromamba env list

# Run a command without mutating the parent shell
MICROMAMBA_ROOT_PREFIX="$HOME/.local/share/mamba" \
micromamba run -n fjs python -V

# Interactive shell (one-time per login)
eval "$(micromamba shell hook -s bash)"
micromamba activate fjs
```

The repository is cloned at `~/fjs-dealias-portfolio`. Pull latest changes before executing long jobs:
```bash
cd ~/fjs-dealias-portfolio
git fetch origin
git status
git pull --ff-only
```

---

## Running long evaluations
Example: daily evaluation with prewhitening and overlay diagnostics.
```bash
cd ~/fjs-dealias-portfolio
MICROMAMBA_ROOT_PREFIX="$HOME/.local/share/mamba" \
micromamba run -n fjs \
python experiments/eval/run.py \
  --returns-csv data/returns_daily.csv \
  --window 126 --horizon 21 \
  --shrinker rie \
  --out reports/rc-$(date +%Y%m%d)/
```

---

## Automation helpers (local workstation)
To remove manual SSH steps, the repository ships deterministic AWS wrappers:

- `scripts/aws_provision.sh` — launches (or resumes) the tagged EC2 runner with OpenBLAS pinned to one thread, installs micromamba, and prepares the `fjs` environment. Export the required AWS variables (`AWS_PROFILE`, `AWS_REGION`, `KEY_NAME`, `SUBNET_ID`, `SECURITY_GROUP_IDS`) before running.
- `scripts/aws_run.sh` — rsyncs the repo to the runner, executes a make target via `micromamba run`, writes `reports/runs/<run_id>/run.json` with hardware + BLAS metadata, and syncs `reports/` back under `reports/aws/<run_id>/` locally.
- Makefile namespaces: `make aws:rc-lite`, `make aws:rc`, `make aws:sweep-calibration`. These call `scripts/aws_run.sh` and accept the same environment overrides (e.g. `KEY_PATH`, `INSTANCE_DNS`, `AWS_REPORTS_DIR`).

Example local invocation:
```bash
export INSTANCE_DNS=ec2-3-236-225-54.compute-1.amazonaws.com
export KEY_PATH="$HOME/.ssh/mateo-us-east-1-ec2-2025"
CALIB_WORKERS=96 MONITOR_INTERVAL=5 make aws:rc-lite
```

Set `SKIP_INSTALL=1` to reuse the existing environment, or `SKIP_REPORT_SYNC=1` to leave artefacts on the runner only. The generated `reports/runs/<run_id>/run.json` mirrors the git SHA, target, timing, CPU/BLAS metadata, and dataset hashes for provenance.

### Telemetry & progress tracking

Every AWS run now streams structured telemetry via `tools/run_monitor.py`:

- `reports/runs/<run_id>/metrics.jsonl` – 5s samples of host CPU%, process CPU%, RSS/USS memory, IO deltas, thread counts, and progress snapshots.
- `reports/runs/<run_id>/metrics_summary.json` – aggregate statistics (avg/peak CPU, memory, IO totals) plus wall-clock start/end and exit status.
- `reports/runs/<run_id>/progress.jsonl` – parsed progress events (e.g., `calibration_progress` with live ETA).
- Console output includes `[monitor] … ETA` updates derived from progress payloads.

Tune sampling with `MONITOR_INTERVAL` (seconds). For example, `MONITOR_INTERVAL=2 make aws:calibrate-thresholds` increases granularity. All metadata is embedded in the resulting `run.json` for quick inspection.

Tips:
- Include `OMP_NUM_THREADS=1` (already enforced globally) if launching multi-process workloads.
- For heavier batches, consider staging inputs under `/data` and syncing outputs to S3 when finished.
- Upload fresh artefacts:
  ```bash
  aws s3 sync reports/rc-$(date +%Y%m%d)/ s3://$BUCKET/reports/rc-$(date +%Y%m%d)/
  ```

---

## Maintenance & troubleshooting
- **AWS CLI update (instance):**
  ```bash
  curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install --update
  rm -rf /tmp/aws /tmp/awscliv2.zip
  ```
- **Micromamba packages:** Use `micromamba run -n fjs pip install -U <package>` or recreate the env with `micromamba create -y -n fjs python=3.11`.
- **Bucket policy drift:** Reapply encryption/lifecycle with the JSON snippets in the setup script or run:
  ```bash
  aws s3api put-bucket-encryption --bucket "$BUCKET" \
    --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" \
    --lifecycle-configuration file:///tmp/fjs_lifecycle.json
  ```
  (See the repo history or `s3://$BUCKET/docs/` for the exact JSON used on 2025-11-05.)
- **Auditing:** The latest setup reports in S3 capture instance metadata, IAM role, installed packages, and smoke-test confirmations.

Keep this document current when instances, buckets, or IAM roles change so future cloud runs remain reproducible.
