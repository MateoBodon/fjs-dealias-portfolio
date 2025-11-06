#!/usr/bin/env bash
#
# Provision a deterministic EC2 runner for fjs-dealias-portfolio jobs.
# The script launches (or reuses) a tagged instance, installs system
# dependencies, pins BLAS/OpenMP thread counts to 1, and prepares the
# ubuntu user for rsync-based job execution.
#
# Required environment variables:
#   AWS_PROFILE   – AWS CLI profile with EC2 + IAM permissions
#   AWS_REGION    – Region to operate in (e.g. us-east-1)
#   KEY_NAME      – EC2 key pair name registered in the region
#   SUBNET_ID     – Subnet for the instance
#   SECURITY_GROUP_IDS – Space-separated list of SG IDs
# Optional environment variables (defaults shown):
#   RUNNER_NAME="fjs-dealias-runner"
#   INSTANCE_TYPE="c7i.2xlarge"
#   AMI_ID (Ubuntu 22.04 LTS)  – auto-resolved if empty
#   IAM_INSTANCE_PROFILE="EC2AdminRole"
#   VOLUME_SIZE_GB=200
#   TAG_ENVIRONMENT="research"
#   TAG_OWNER="$USER"
#   START_SHELL=0  – if set to 1, SSH into the instance after provisioning.
#
# Example:
#   AWS_PROFILE=fjs-prod AWS_REGION=us-east-1 \
#   KEY_NAME=mateo-us-east-1-ec2-2025 \
#   SUBNET_ID=subnet-0929c64978f562f59 \
#   SECURITY_GROUP_IDS="sg-0123456789abcdef0" \
#   ./scripts/aws_provision.sh

set -euo pipefail

if [[ -n "${DEBUG_PROVISION:-}" ]]; then
  set -x
fi

function require_env() {
  local var="$1"
  if [[ -z "${!var:-}" ]]; then
    echo "error: environment variable '$var' must be set" >&2
    exit 1
  fi
}

require_env "AWS_PROFILE"
require_env "AWS_REGION"
require_env "KEY_NAME"
require_env "SUBNET_ID"
require_env "SECURITY_GROUP_IDS"

RUNNER_NAME="${RUNNER_NAME:-fjs-dealias-runner}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7a.32xlarge}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-EC2AdminRole}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-200}"
TAG_ENVIRONMENT="${TAG_ENVIRONMENT:-research}"
TAG_OWNER="${TAG_OWNER:-$USER}"
START_SHELL="${START_SHELL:-0}"

trim() {
  local value="$1"
  echo -n "$value" | awk '{$1=$1;print}'
}

AWS() {
  aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"
}

resolve_ami() {
  if [[ -n "${AMI_ID:-}" ]]; then
    echo "$AMI_ID"
    return
  fi
  AWS ec2 describe-images \
    --owners "099720109477" \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate)[-1].ImageId' \
    --output text
}

INSTANCE_ID=""
PUBLIC_DNS=""

maybe_existing_runner() {
  AWS ec2 describe-instances \
    --filters "Name=tag:Name,Values=$RUNNER_NAME" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].{InstanceId:InstanceId,State:State.Name,PublicDns:PublicDnsName}' \
    --output json
}

existing="$(maybe_existing_runner)"
if [[ "$existing" != "[]" && "$existing" != "" ]]; then
  mapfile -t EXISTING_INFO < <(echo "$existing" | python3 - <<'PY'
import json,sys
data=json.load(sys.stdin)
if not data:
    sys.exit(1)
chosen=min(data, key=lambda item: {"pending":0,"running":1,"stopped":2,"stopping":3}.get(item["State"],9))
print(chosen["InstanceId"])
print(chosen["State"])
print(chosen.get("PublicDns",""))
PY
)
  INSTANCE_ID="${EXISTING_INFO[0]}"
  STATE="${EXISTING_INFO[1]}"
  PUBLIC_DNS="${EXISTING_INFO[2]}"
  if [[ "$STATE" == "running" ]]; then
    echo "Found existing runner '$RUNNER_NAME' ($INSTANCE_ID) in state 'running'."
    echo "Public DNS: ${PUBLIC_DNS:-<pending>}"
  else
    echo "Found runner '$RUNNER_NAME' ($INSTANCE_ID) in state '$STATE'; starting..."
    AWS ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
  fi
else
  AMI_ID="$(resolve_ami)"
  if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    echo "error: unable to resolve Ubuntu 22.04 AMI" >&2
    exit 1
  fi

  TMP_USER_DATA="$(mktemp)"
  cat >"$TMP_USER_DATA" <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail

exec >/var/log/cloud-init-output.log 2>&1

apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  git \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  libopenblas-pthread-dev \
  liblapack-dev \
  libatlas-base-dev \
  curl \
  rsync \
  unzip \
  htop \
  tmux \
  jq \
  sysstat

if update-alternatives --query libblas.so.3 >/dev/null 2>&1; then
  update-alternatives --set libblas.so.3 /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 || true
fi
if update-alternatives --query liblapack.so.3 >/dev/null 2>&1; then
  update-alternatives --set liblapack.so.3 /usr/lib/x86_64-linux-gnu/liblapack.so.3 || true
fi

THREAD_EXPORTS=$(cat <<'EOENV'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
EOENV
)

echo "$THREAD_EXPORTS" >> /etc/profile.d/fjs_threads.sh
chmod 0644 /etc/profile.d/fjs_threads.sh

HOME_BASHRC="/home/ubuntu/.bashrc"
if ! grep -q "OPENBLAS_NUM_THREADS" "$HOME_BASHRC"; then
  echo "" >> "$HOME_BASHRC"
  echo "# fjs-dealias deterministic thread caps" >> "$HOME_BASHRC"
  echo "$THREAD_EXPORTS" >> "$HOME_BASHRC"
fi

chown ubuntu:ubuntu "$HOME_BASHRC" /etc/profile.d/fjs_threads.sh

INSTALLER_DIR="/tmp/micromamba"
mkdir -p "$INSTALLER_DIR"
curl -sL https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xj -C "$INSTALLER_DIR" --strip-components=1 bin/micromamba
install -o root -g root -m 0755 "$INSTALLER_DIR/micromamba" /usr/local/bin/micromamba
rm -rf "$INSTALLER_DIR"

sudo -u ubuntu -H bash <<'EOSU'
set -euxo pipefail
export MICROMAMBA_ROOT_PREFIX="\$HOME/.local/share/mamba"
micromamba shell init -s bash -r "\$MICROMAMBA_ROOT_PREFIX" >> "\$HOME/.bashrc"
micromamba create -y -n fjs python=3.11
micromamba run -n fjs pip install --upgrade pip wheel setuptools
REPO_PATH="\$HOME/fjs-dealias-portfolio"
if [[ -f "\$REPO_PATH/pyproject.toml" ]]; then
  micromamba run -n fjs pip install -e "\$REPO_PATH[dev]" || micromamba run -n fjs pip install -e "\$REPO_PATH"
fi
EOSU

systemctl daemon-reload
EOF

  echo "Launching new EC2 runner '$RUNNER_NAME' (instance type: $INSTANCE_TYPE, ami: $AMI_ID)..."
  INSTANCE_ID="$(AWS ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --iam-instance-profile "Name=$IAM_INSTANCE_PROFILE" \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids $SECURITY_GROUP_IDS \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --user-data "file://$TMP_USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUNNER_NAME},{Key=Environment,Value=$TAG_ENVIRONMENT},{Key=Owner,Value=$TAG_OWNER}]" \
    --query 'Instances[0].InstanceId' \
    --output text)"
  rm -f "$TMP_USER_DATA"
  if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
    echo "error: failed to launch instance" >&2
    exit 1
  fi
fi

echo "Waiting for instance '$INSTANCE_ID' to reach 'running'..."
AWS ec2 wait instance-running --instance-ids "$INSTANCE_ID"

echo "Waiting for status checks..."
AWS ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"

PUBLIC_DNS="$(AWS ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' \
  --output text)"

if [[ -z "$PUBLIC_DNS" || "$PUBLIC_DNS" == "None" ]]; then
  echo "warning: public DNS not yet available; query with 'aws ec2 describe-instances --instance-ids $INSTANCE_ID'" >&2
else
  echo "Runner ready: $INSTANCE_ID ($PUBLIC_DNS)"
fi

if [[ "${START_SHELL}" == "1" && -n "$PUBLIC_DNS" && "$PUBLIC_DNS" != "None" ]]; then
  echo "Opening SSH shell..."
  ssh -o StrictHostKeyChecking=accept-new -i "${KEY_PATH:-$HOME/.ssh/${KEY_NAME}}" "ubuntu@${PUBLIC_DNS}"
fi
